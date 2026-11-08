"""Prompt 构建：把「检索到的资料」和「用户问题」拼成发给大模型的消息。

RAG 之所以叫"增强生成"，重点就在这一步：
模型的参数没变，变的是它这一次看到的 Prompt 里多了外部知识。
"""

from rag.schema import SearchResult

# system 消息用来锁定角色和行为边界。
# 生产环境的 Prompt 一定要写清楚"不许编造""资料不足就承认"，
# 这是压制幻觉最有效、最便宜的手段（阶段 8 会讲更系统的防护）。
#
# 注意：角色设定要跟着语料走。之前这里是"电商平台客服助手"，
# 那是因为知识库是电商政策；现在语料换成了通用中文文本（百科/法律/评测等），
# 角色必须跟着改，否则模型会拿"客服口吻 + 电商规则"去套完全无关的内容。
SYSTEM_PROMPT = """你是一名严谨的知识库问答助手，依据检索到的资料回答问题。

回答规则：
1. 只依据【参考资料】中的信息回答，不要用你自己的常识补充细节。
2. 资料可能只是原文的一个片段，上下文不完整；如果它不足以回答问题，直接说明"根据现有资料无法确定"，并指出缺少什么信息。
3. 不要编造事实、数据、时间、人名等任何具体信息。
4. 回答简洁，分点作答。
5. 在答案最后单独一行标注引用的资料序号，格式如：引用：[资料1][资料3]
"""


def format_context(results: list[SearchResult]) -> str:
    """把检索结果拼成带编号和来源名的资料块。

    来源里带上 pid，是因为切片后一个文档会变成多段：
    只写"资料1"没法回溯原文，写上 pid + 段号，出问题能立刻定位到语料哪一条。
    """
    blocks: list[str] = []

    for index, result in enumerate(results, start=1):
        title = result.document.title
        doc_id = str(result.document.doc_id)

        # 标题里已经含 pid 就别重复拼（切片集合的 title 形如 "pid 2 · 第 2 段"）
        source = title if doc_id in title else f"{title}（语料 {doc_id}）"

        blocks.append(
            f"[资料{index}] 来源：{source}\n"
            f"{result.document.text}"
        )

    return "\n\n".join(blocks)


def build_messages(question: str, results: list[SearchResult]) -> list[dict[str, str]]:
    """构造符合 OpenAI Chat Completions 格式的 messages 列表。

    返回结构形如：
        [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."},
        ]
    """
    context = format_context(results)

    user_content = (
        f"【参考资料】\n{context}\n\n"
        f"【用户问题】\n{question}\n\n"
        "请依据上述参考资料回答，并在结尾标注引用的资料序号。"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
