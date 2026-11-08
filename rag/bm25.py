"""手写 BM25 检索器（稀疏检索）。

和阶段 1 的 TF-IDF 是同一族，但 BM25 在两处做了改进：
1. **词频饱和**：TF 用 f/(f+k1) 压成曲线，出现 50 次不等于重要 50 倍。
2. **长度归一化**：用 b 参数控制"长文档该被惩罚多少"。

为什么自己写而不是装 rank_bm25？
    核心逻辑不到 60 行，写出来才看得见 k1 / b 在做什么。
    而且我们需要按项目自己的中文 tokenizer 分词（字符 bigram），
    外部库的分词接口反而是个障碍。

分词用 rag.tokenizer.tokenize：
    中文切成字符二元组，英文/数字整段保留。
    "95017" 会作为一个整体 token —— 这正是我们想要的，稠密向量对数字串无感，而 BM25 能精确命中。
"""

import math
from collections import Counter, defaultdict

from rag.tokenizer import tokenize


class BM25Index:
    """内存倒排索引。数据量在十万级以下时，纯 Python 完全够用。"""

    name = "bm25"

    def __init__(
        self,
        docs: list[tuple[str, str]],
        k1: float = 1.2,
        b: float = 0.5,
    ) -> None:
        """建索引。

        Args:
            docs: [(chunk_id, text), ...]
            k1: 词频饱和系数。1.2 是通用默认值。
            b: 长度归一化强度。默认取 0.5 而不是 0.75——
               因为我们的 chunk 都被切到 384 token 附近，长度差异本来就小，
               用 0.75 会过度惩罚本来就不长的块。
        """
        self.k1 = k1
        self.b = b

        self.doc_ids: list[str] = []
        self.texts: dict[str, str] = {}
        self.doc_len: list[int] = []

        # term -> [(doc_index, term_freq), ...]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}

        self._build(docs)

    def _build(self, docs: list[tuple[str, str]]) -> None:
        df: Counter[str] = Counter()

        for idx, (chunk_id, text) in enumerate(docs):
            tokens = tokenize(text)

            self.doc_ids.append(chunk_id)
            self.texts[chunk_id] = text
            self.doc_len.append(len(tokens))

            for term, freq in Counter(tokens).items():
                self.postings[term].append((idx, freq))
                df[term] += 1

        total = max(1, len(self.doc_ids))
        self.avgdl = sum(self.doc_len) / total

        # 平滑 IDF：log(1 + (N - n + 0.5) / (n + 0.5))
        # +1 在外面保证 IDF 恒为正，罕见词不会被算成负权重
        for term, n in df.items():
            self.idf[term] = math.log(1 + (total - n + 0.5) / (n + 0.5))

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """返回 [(chunk_id, score), ...]，按分数降序。

        只累加出现在倒排表里的 term —— 没出现过的词贡献恒为 0，
        遍历它们纯属浪费。
        """
        scores: dict[int, float] = defaultdict(float)

        # query 里的重复 token 只算一次（BM25 标准做法，避免短查询被重复词放大）
        query_terms = {t for t in tokenize(query) if t in self.postings}

        for term in query_terms:
            idf = self.idf[term]
            for doc_idx, freq in self.postings[term]:
                dl = self.doc_len[doc_idx]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf * freq * (self.k1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [(self.doc_ids[idx], score) for idx, score in ranked]

    def info(self) -> dict:
        return {
            "name": self.name,
            "size": self.size,
            "vocab_size": len(self.postings),
            "avgdl": round(self.avgdl, 1),
            "k1": self.k1,
            "b": self.b,
            "model": "BM25（字符 bigram 分词，纯 Python）",
        }
