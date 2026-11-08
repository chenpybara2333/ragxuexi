"""中文分词器：把一段中文文本切成一个个"词元"（token）。

为什么需要这一步？
TF-IDF、BM25、Embedding 之前的预处理，本质都是"把文本变成一串可计算的符号"。
英文天然有空格分隔，中文没有，所以必须自己切。

这里不引入 jieba 等分词库，而是用最经典的 **字符二元组（bi-gram）** 方案：

    "七天无理由退货"  ->  七天 / 天无 / 无理 / 理由 / 由退 / 退货

优点：不需要词典、不会切错、对未登录词（新词、品牌名）友好。
缺点：词元数量变多，且"无理"/"理由"这类无意义组合也会混进来。
企业里常用 jieba / HanLP 做词级切分，或用 BPE 子词切分（大模型就是这么做的），
我们在阶段 3 会碰到真正的 tokenizer。
"""

import re

# 连续的小写字母或数字，例如 "iphone15"、"5g"、"x1"
ASCII_WORD = re.compile(r"[a-z0-9]+")

# 连续的中日韩统一表意文字（即汉字区），例如 "退货政策"
CJK_BLOCK = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """把文本切成 token 列表。

    >>> tokenize("7天无理由退货")
    ['7', '天无', '无理', '理由', '由退', '退货']
    """
    # 统一小写，避免 "iPhone" 和 "iphone" 被当成两个不同的词
    text = text.lower()

    tokens: list[str] = []

    # 1) 英文/数字：整段作为一个 token（"iphone15" 不拆开，因为它本身就是一个型号）
    tokens.extend(ASCII_WORD.findall(text))

    # 2) 中文：按标点切出连续的汉字块，块内做二元组滑窗
    for block in CJK_BLOCK.findall(text):
        if len(block) == 1:
            # 单个汉字没法组成二元组，直接保留（否则这个字会彻底丢失）
            tokens.append(block)
        else:
            for i in range(len(block) - 1):
                tokens.append(block[i : i + 2])

    return tokens
