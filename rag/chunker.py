"""文本切分器：把长文档切成 chunk，带 overlap。

为什么不能直接按字符硬切？
    text[i:i+384] 会把一句话劈成两半："……签收后七日" | "内可以申请退货……"
    两个 chunk 的向量都是"半句话"，语义都糊。所以要先找语义边界。

实现思路（和 LangChain 的 RecursiveCharacterTextSplitter 同源，但按 token 计量）：
    1. 用模型自己的 tokenizer 把文本编码成 token 序列，同时拿到每个 token 对应的字符区间
    2. 按 token 数开窗（size=384，步长 = size - overlap = 320）
    3. 每个窗口的结尾，向回找最近的句末标点（。！？；换行），在句末切
    4. 找不到合适的句末（比如一个超长无标点段落），就按原窗口硬切，保证不死循环
    5. 下一个窗口从前一个窗口的结尾往回退 overlap 个 token，形成重叠区

为什么用 token 而不是字符计量？
    模型的上限是 512 token，不是 512 字符。"你好" 是 2 字符约 2 token，
    但一个英文单词可能 8 字符才 1-2 token。按字符切，中文和英文混排时长度完全失控。
"""

import re
from dataclasses import dataclass

SENTENCE_END = re.compile(r"[。！？；!?;\n]")


@dataclass
class Chunk:
    """一个切片。"""

    index: int          # 在原文中的序号，从 0 开始
    text: str           # 切片文本
    char_start: int     # 在原文中的起始字符位置
    char_end: int       # 结束字符位置（不含）
    token_count: int    # 实际 token 数


def _load_tokenizer():
    """加载和 embedding 模型配套的 tokenizer（必须同一个，计数才对得上）。"""
    from pathlib import Path

    from tokenizers import Tokenizer

    root = Path(__file__).resolve().parent.parent
    candidates = list(root.glob(".model_cache/**/tokenizer.json"))
    if not candidates:
        raise FileNotFoundError(
            "找不到 tokenizer.json，请先跑一次 embedding 下载模型（见 rag/embedder.py）"
        )
    return Tokenizer.from_file(str(candidates[0]))


_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = _load_tokenizer()
    return _TOKENIZER


def count_tokens(text: str) -> int:
    """用真实 tokenizer 数 token，不估算。"""
    return len(get_tokenizer().encode(text, add_special_tokens=False).ids)


def chunk_text(
    text: str,
    size: int = 384,
    overlap: int = 64,
    snap_window: int = 80,
) -> list[Chunk]:
    """把文本切成若干 Chunk。

    Args:
        text: 原始文本
        size: 每个 chunk 的目标 token 上限
        overlap: 相邻 chunk 的重叠 token 数
        snap_window: 向回找句末标点的最大字符搜索范围

    Returns:
        Chunk 列表；文本本身不超过 size 时返回单个 chunk
    """
    if size <= 0:
        raise ValueError("size 必须大于 0")
    if overlap >= size:
        raise ValueError("overlap 必须小于 size，否则窗口不会前进，会死循环")

    text = text.strip()
    if not text:
        return []

    tok = get_tokenizer()
    encoding = tok.encode(text, add_special_tokens=False)
    ids = encoding.ids
    offsets = encoding.offsets  # 每个 token 对应的 (char_start, char_end)

    if len(ids) <= size:
        return [Chunk(index=0, text=text, char_start=0, char_end=len(text), token_count=len(ids))]

    stride = size - overlap
    chunks: list[Chunk] = []
    start_tok = 0

    while start_tok < len(ids):
        end_tok = min(start_tok + size, len(ids))

        char_start = offsets[start_tok][0]
        char_end = offsets[end_tok - 1][1]

        # 最后一段直接收尾，不要为了凑句末再往回退
        if end_tok < len(ids):
            # 向回找最近的句末标点，避免把句子劈开
            search_from = max(char_start + 1, char_end - snap_window)
            last_end = -1
            for match in SENTENCE_END.finditer(text, search_from, char_end):
                last_end = match.end()

            if last_end > char_start:
                char_end = last_end
                # 把 token 指针退回到该字符位置，保证下一窗口从正确的地方开始
                end_tok = next(
                    (i for i in range(start_tok, end_tok) if offsets[i][1] > char_end),
                    end_tok,
                )

        piece = text[char_start:char_end]
        chunks.append(
            Chunk(
                index=len(chunks),
                text=piece,
                char_start=char_start,
                char_end=char_end,
                token_count=count_tokens(piece),
            )
        )

        if char_end >= len(text):
            break

        # 下一窗口：从当前结尾往回退 overlap 个 token 开始，形成重叠
        next_start = max(start_tok + 1, end_tok - overlap)
        # 保险：如果退回后位置没前进（极端短句密集的情况），强制前进一步
        if next_start <= start_tok:
            next_start = start_tok + stride
        start_tok = next_start

    return chunks
