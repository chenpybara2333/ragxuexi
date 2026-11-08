"""一次 RAG 问答的完整执行过程，封装成一个可复用函数。

为什么要单独抽这一层？
    CLI 入口（main.py）和 Web 入口（server.py）用的是同一套检索与生成逻辑。
    如果各写一遍，改一次检索策略要改两处，两周后必然漂移。

    企业项目里这条边界叫「业务层 / 接口层分离」：
        pipeline.py  = 业务层，只关心"做什么"，不关心怎么输出
        main.py      = 终端适配器
        server.py    = HTTP 适配器
    以后加微信机器人、加定时任务，都只写一个新的适配器。

返回值是纯 dict（可直接 json.dumps），因为 Web 层需要把中间态全部吐给前端。
注意：这里故意把"检索到的候选文档"和"真正进入 Prompt 的文档"分开返回，
      前端才能看出 Top-K 到底是怎么截断的。
"""

import time
from typing import Any

from rag.config import MIN_SCORE
from rag.llm import LLMError, chat, get_llm_config, is_llm_configured, mock_answer
from rag.prompt import build_messages
from rag.tokenizer import tokenize

# 检索结果全为空时的固定回答。
# 关键点：这时候【不调用大模型】。没有资料还去问模型，它只会用训练时的通用常识编一段，
# 而用户会以为"这是你们平台的政策"——这是 RAG 系统最典型的事故来源。
NO_CONTEXT_ANSWER = (
    "知识库里没有找到与这个问题相关的内容，因此无法回答。\n\n"
    "可以试试：\n"
    "- 换个更具体的说法（例如把「坏了」说成「质量问题」）\n"
    "- 补充商品名、订单环节等关键词\n"
    "- 或者转接人工客服"
)


def _ms(start: float) -> float:
    """从 start 到现在经过的毫秒数，保留 1 位小数。"""
    return round((time.perf_counter() - start) * 1000, 1)


def run_rag(question: str, retriever: Any, top_k: int = 2) -> dict[str, Any]:
    """跑完一次完整 RAG，返回包含所有中间态的结果字典。

    retriever 只要求实现 search(query, top_k) / size / info() 三个成员，
    具体是 Chroma 还是别的实现，这里不关心（鸭子类型）。

    返回结构：
        question    原始问题
        retrieval   分词结果 + 全部候选文档（含是否入选）
        contexts    真正拼进 Prompt 的文档正文
        prompt      system / user 两段完整文本
        answer      回答文本、生成模式（llm / mock）、模型名、错误信息
        timings     各阶段耗时
    """
    total_start = time.perf_counter()

    # ================= 1. Retrieval =================
    retrieval_start = time.perf_counter()

    query_tokens = tokenize(question)

    # 要求检索器多返回 2 条：前端才能显示"分数够但排到 Top-K 之外"的证据。
    # 这就是 over_k 状态存在的前提 —— 只看前 top_k 条，永远看不到被截断的那部分。
    fetch_k = min(getattr(retriever, "size", top_k + 2), top_k + 2)

    # 支持"能带过程信息"的检索器（多查询改写）。普通检索器走 else 分支。
    # 用能力检测而不是类型判断：以后再加别的检索器，pipeline 不用改。
    rewrite_trace = None
    if hasattr(retriever, "search_with_trace"):
        all_results, rewrite_trace = retriever.search_with_trace(question, top_k=fetch_k)
    else:
        all_results = retriever.search(question, top_k=fetch_k)

    # 逐篇判定它的去向。三种状态必须区分清楚，否则排查时会误判：
    #   selected  -> 进了 Prompt
    #   low_score -> 分数不达标，被阈值拦掉（说明"根本没查到"，不是"没排上"）
    #   over_k    -> 分数够但排到 Top-K 之外（说明"查到了但名额不够"）
    candidates: list[dict[str, Any]] = []
    selected: list = []

    for rank, item in enumerate(all_results, start=1):
        if item.score <= MIN_SCORE:
            flag = "low_score"
        elif rank > top_k:
            flag = "over_k"
        else:
            flag = "selected"

        candidates.append(
            {
                "rank": rank,
                "doc_id": item.document.doc_id,
                "title": item.document.title,
                "score": round(item.score, 4),
                "chars": len(item.document.text),
                "selected": flag == "selected",
                "flag": flag,
            }
        )

        if flag == "selected":
            selected.append(item)

    contexts = [
        {
            "rank": rank,
            "title": item.document.title,
            "score": round(item.score, 4),
            "text": item.document.text,
        }
        for rank, item in enumerate(selected, start=1)
    ]

    retrieval_ms = _ms(retrieval_start)

    # ================= 2. Prompt 构建 =================
    # 拼 Prompt 本身几乎不耗时，但它是 RAG 最容易出问题的地方，所以单独返回全文
    messages = build_messages(question, selected)
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    # ================= 3. Generation =================
    generation_start = time.perf_counter()

    config = get_llm_config()
    error: str | None = None

    if not selected:
        # 召回为空 —— RAG 里最重要的分支之一。
        # 不调用模型，直接给出"答不了"。这一步省下的不只是 token 费，
        # 更关键的是不给模型编造的机会。
        mode = "empty"
        model_name = "（未调用模型）"
        answer_text = NO_CONTEXT_ANSWER
    elif is_llm_configured():
        mode = "llm"
        model_name = config["model"]
        try:
            answer_text = chat(messages)
        except LLMError as exc:
            # 调用失败时降级到模拟模式，保证前端不会白屏，同时把错误原样暴露出来
            mode = "mock"
            error = str(exc)
            answer_text = mock_answer(question, selected)
    else:
        mode = "mock"
        model_name = "（未调用模型）"
        answer_text = mock_answer(question, selected)

    generation_ms = _ms(generation_start)

    return {
        "question": question,
        "backend": retriever.info(),
        "query_rewrite": rewrite_trace,
        "retrieval": {
            "query_tokens": query_tokens,
            "top_k": top_k,
            "candidates": candidates,
        },
        "contexts": contexts,
        "prompt": {
            "system": system_prompt,
            "user": user_prompt,
            # Token 数取决于模型自己的分词器，这里只给字符数，不瞎猜
            "chars": len(system_prompt) + len(user_prompt),
        },
        "answer": {
            "text": answer_text,
            "mode": mode,
            "model": model_name,
            "error": error,
        },
        "timings": {
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": _ms(total_start),
        },
    }
