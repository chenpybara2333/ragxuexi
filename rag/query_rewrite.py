"""查询改写 / 多查询生成。

解决什么问题？
    有些问题本身"太泛"，无论用什么检索算法都难以定位：
        "写文章时拿两样东西作对照，能起到什么效果？"
    这段问的是"对比/衬托手法的作用"，但用户用了"两样东西作对照"这种口语说法，
    和原文用词完全对不上。稠密向量也只能泛泛匹配到一批讲写作的段落。

    解法：先用模型把问题改写成几个更明确、措辞不同的版本，分别检索再融合。

两个必须守住的约束：
1. **不许添加原问题里没有的具体信息**。
   让模型"把问题写具体"，它很容易编出具体对象（"《XX》有声书"），
   一旦编错，检索会被带向完全无关的方向，比不改写更差。
   所以 prompt 里明确要求：只换措辞，不猜对象。
2. **原问题必须保留为一路**。
   改写可能变差（尤其短问题、专有名词多的问题）。
   把原问题当作其中一路，最坏情况也不比改写前差多少。
"""

import json

from rag.llm import LLMError, chat

PROMPT = """你在为一个检索系统扩展用户查询。

用户问题：{question}

请生成 {n} 个不同角度的检索查询，用来从知识库中找出能回答该问题的资料。

要求：
1. 每个查询都要保留原问题的真实意图。
2. 换用不同措辞、同义说法，或把口语表达换成更书面/更具体的表达。
3. **绝对不要添加原问题中没有的具体信息**。
   不要编造书名、型号、公司名、人名、数字。如果原问题很模糊，
   就换几种更明确的表达方式，而不是猜一个具体对象。
4. 每个查询独立完整，不要出现"它""这个""上述"之类的指代。
5. 只输出 JSON 字符串数组，不要解释、不要 Markdown 代码块：
["查询1", "查询2", "查询3"]
"""


def parse_variants(raw: str) -> list[str]:
    """从模型输出里抠出字符串数组。"""
    text = raw.strip()

    if "```" in text:
        for part in text.split("```"):
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("["):
                text = candidate
                break

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"找不到 JSON 数组：{raw[:150]}")

    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("输出的不是数组")

    return [str(x).strip() for x in data if str(x).strip()]


def rewrite(question: str, n: int = 3, temperature: float = 0.0) -> list[str]:
    """生成 n 个改写查询。失败时返回空列表（调用方应回退到原问题）。

    返回的列表**不含**原问题，由调用方自己决定是否保留原问题那一路。

    为什么默认 temperature=0？
        改写结果直接决定召回内容。温度一高，同一个问题每次跑出来的变体都不一样，
        评测指标就不可复现——同样一套题跑两遍，Recall 能差好几个点，
        你会分不清是"改动有效"还是"这次运气好"。
        温度设 0 不影响多样性：3 个变体是在同一次响应里生成的，本来就各不相同。
    """
    prompt = PROMPT.format(question=question, n=n)

    try:
        raw = chat([{"role": "user", "content": prompt}], temperature=temperature)
        variants = parse_variants(raw)
    except (LLMError, ValueError, json.JSONDecodeError):
        return []

    # 去掉与原问题重复、以及彼此重复的项
    seen = {question.strip()}
    out: list[str] = []

    for v in variants:
        key = v.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)

    return out[:n]
