"""大模型调用层：兼容 OpenAI Chat Completions 格式的任意供应商。

为什么用 requests 手写，而不用 openai SDK？
因为这一层只需要一个 HTTP POST。用 requests 能让你清楚看到：
请求体里到底有什么、返回的 JSON 长什么样、换供应商时到底要改哪些字段。

只要对方兼容 /v1/chat/completions（DeepSeek、通义、Kimi、智谱、Ollama、vLLM 都兼容），
改 .env 里的 LLM_BASE_URL 和 LLM_MODEL 就能切换，代码一行都不用动。
"""

import os
import time
import uuid

import requests

# 关键：必须显式 import，才能触发 config.py 里的 load_env_file()。
# .env 的加载是 import 时的副作用，不 import 就读不到 Key。
import rag.config  # noqa: F401
from rag.schema import SearchResult
from rag.tokenizer import tokenize

# 同一个进程内保持不变：多轮对话复用同一个 session，网关才能命中 prompt 缓存。
# 想跨进程复用（比如跑脚本对比效果），在 .env 里固定 LLM_SESSION_ID 即可。
SESSION_ID = os.getenv("LLM_SESSION_ID", "").strip() or uuid.uuid4().hex


class LLMError(RuntimeError):
    """调用大模型过程中的所有异常，统一用这个类型抛出，方便上层捕获。"""


# ----------------------------------------------------------------------
# 配置读取
# ----------------------------------------------------------------------
def get_llm_config() -> dict[str, str]:
    return {
        "api_key": os.getenv("LLM_API_KEY", "").strip(),
        "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
    }


def is_llm_configured() -> bool:
    """是否配置了 API Key。没有就走模拟模式。"""
    return bool(get_llm_config()["api_key"])


# 这些状态码是"服务端临时抽风"，重试有意义。
# 400/401/404 是"你请求写错了"，重试一百次也一样错，必须直接报出来。
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _post_with_retry(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: int,
    max_retries: int | None = None,
) -> requests.Response:
    """带指数退避的 POST。

    网关限流（429）和偶发过载很常见，直接失败会让用户以为是自己配错了。
    退避间隔 1s -> 2s -> 4s，避免瞬间重试把对方打得更死。
    """
    if max_retries is None:
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))

    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            # 网络层抖动（含 SSL 突然断开）同样值得重试一次
            last_exc = exc
            if attempt == max_retries:
                raise
        else:
            if response.status_code not in RETRYABLE_STATUS or attempt == max_retries:
                return response
            last_exc = None

        wait = 2**attempt
        print(f"  [重试] 第 {attempt + 1}/{max_retries} 次失败，{wait}s 后重试…")
        time.sleep(wait)

    # 理论上走不到这里，兜底保证函数一定有返回值或异常
    if last_exc:
        raise last_exc
    raise LLMError("请求失败且无响应")


def _hint_for_status(status: int) -> str:
    """把最常见的几个状态码翻译成人话，省得对着 JSON 猜。"""
    return {
        400: "请求体不合法。若提示 MissingSessionID / session，说明该网关要求额外的请求头。",
        401: "API Key 无效或已过期，检查 .env 里的 LLM_API_KEY 有没有多余空格。",
        403: "Key 有效但无权限访问该模型，确认账号已订阅或该模型对你开放。",
        404: "路径或模型名不对。注意 LLM_BASE_URL 应该以 /v1 结尾，且不要重复带 /chat/completions。",
        429: "限流或上游过载。稍等再试，或调大 LLM_MAX_RETRIES。",
    }.get(status, "参考响应体里的 message 字段。")


# ----------------------------------------------------------------------
# 真实大模型调用
# ----------------------------------------------------------------------
def chat(messages: list[dict[str, str]], temperature: float = 0.2, timeout: int = 60) -> str:
    """调用 /chat/completions，返回模型回复的纯文本。

    Args:
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        temperature: 0 最稳定，1 最发散。客服场景用 0.2 左右，压低随机性。
        timeout: 超时秒数。生产环境必须设，否则请求可能一直挂着。

    Raises:
        LLMError: 未配置 Key / 网络错误 / 非 200 响应 / 返回结构异常
    """
    config = get_llm_config()

    if not config["api_key"]:
        raise LLMError("未配置 LLM_API_KEY，无法调用真实大模型")

    url = f"{config['base_url']}/chat/completions"

    payload: dict = {
        "model": config["model"],
        "messages": messages,
        "temperature": temperature,
    }

    # 部分新模型（如 o1 系列）不再接受 max_tokens，所以默认不传，
    # 需要限制长度时在 .env 里显式配置 LLM_MAX_TOKENS
    max_tokens = os.getenv("LLM_MAX_TOKENS", "").strip()
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)

    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
        # 下面两个头标准 OpenAI 接口不需要，但 OpenCode Go 网关必须有：
        # 缺 x-opencode-session 会直接返回 400 MissingSessionID。
        # 用自定义 User-Agent 而不是 requests 默认值，也是网关的要求。
        "User-Agent": os.getenv("LLM_USER_AGENT", "ragxuexi/0.1"),
        "x-opencode-session": SESSION_ID,
    }

    try:
        response = _post_with_retry(url, payload, headers, timeout)
    except requests.Timeout as exc:
        raise LLMError(f"请求超时（{timeout}s），请检查网络或调大超时时间") from exc
    except requests.RequestException as exc:
        raise LLMError(f"网络错误：{exc}") from exc

    if response.status_code != 200:
        raise LLMError(
            f"接口返回 HTTP {response.status_code}：{response.text[:500]}\n"
            f"  排查方向：" + _hint_for_status(response.status_code)
        )

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMError(f"响应结构不符合预期：{response.text[:500]}") from exc


# ----------------------------------------------------------------------
# 模拟模式：没有 API Key 时用，保证检索链路可以先跑通
# ----------------------------------------------------------------------
def mock_answer(question: str, results: list[SearchResult]) -> str:
    """不调用大模型，直接从检索结果里"摘抄"答案。

    注意：这不是 RAG 的生成，只是一个占位实现，用来验证"检索"这一段是否正确。
    它不会组织语言、不会总结、不会处理多跳问题，答案质量远低于真实大模型。
    """
    if not results:
        return "【模拟模式】没有检索到任何资料。"

    question_tokens = set(tokenize(question))
    best_doc = results[0].document

    # 按段拆分（txt 里用单个换行分段），挑和问题重合词最多的两段
    paragraphs = [p.strip() for p in best_doc.text.split("\n") if p.strip()]

    def overlap(paragraph: str) -> int:
        return len(question_tokens & set(tokenize(paragraph)))

    ranked = sorted(paragraphs, key=overlap, reverse=True)
    picked = [p for p in ranked[:2] if overlap(p) > 0] or paragraphs[:2]

    body = "\n".join(f"- {p}" for p in picked)

    return (
        "【模拟模式 · 非真实大模型】以下内容是从检索结果中逐字摘抄的，"
        "只用于验证检索链路是否正常，不代表真实生成能力。\n\n"
        f"{body}\n\n"
        f"引用：[资料1]（{best_doc.title}）"
    )
