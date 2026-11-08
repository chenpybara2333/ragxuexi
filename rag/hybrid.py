"""混合检索：BM25 稀疏路 + 向量稠密路，用 RRF 融合。

为什么需要两路？
    稠密路擅长"意思像"（肚子大怎么减 → 产后恢复），
    稀疏路擅长"字面必须命中"（95017、百度文库、灵越7590）。
    两者的失败模式基本不重叠，所以并行跑、按名次合并，比赌任何一路都稳。

为什么用 RRF 而不是加权分数？
    余弦相似度固定 [0,1]，BM25 分数无上界（可能是 8，也可能是 45）。
    直接相加，BM25 会单方面主导；归一化又会被单条异常高分扭曲。
    RRF 只看名次，彻底绕开尺度问题，且 k=60 是通用值，不用调。

为什么在 chunk 级融合、最后才按 pid 去重？
    融合的前提是"两路说的是同一个东西"。chunk_id（pid::序号）是两路唯一共享的标识。
    如果先按 pid 去重再融合，一篇文档的多个片段会被压成一条，
    等于提前丢掉了"这一段比那一段更相关"的信息。
"""

from collections import defaultdict

from rag.bm25 import BM25Index
from rag.chroma_store import ChromaStore
from rag.embedder import embed_query
from rag.schema import Document, SearchResult


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion。

    score(d) = Σ 1 / (k + rank_i(d))

    k 越小，头部名次越强势（k=0 时第 1 名是第 10 名的 10 倍）；
    k 越大，越接近"数被几路提到过"。60 是论文推荐的通用值。
    """
    scores: dict[str, float] = defaultdict(float)

    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)

    return scores


def load_all_chunks(store: ChromaStore) -> list[tuple[str, str]]:
    """把集合里全部 chunk 拉成 [(chunk_id, text)]，用于建 BM25 索引。

    分页读，避免一次性把几万条文档塞进内存峰值。
    """
    total = store.size
    out: list[tuple[str, str]] = []

    for offset in range(0, total, 1000):
        res = store.collection.get(
            limit=min(1000, total - offset),
            offset=offset,
            include=["documents"],
        )
        out.extend(zip(res["ids"], res["documents"]))

    return out


def build_hybrid(store: ChromaStore, recall_n: int = 50, rrf_k: int = 60) -> "HybridRetriever":
    """从已有的 Chroma 集合构建混合检索器（BM25 索引在内存里现建）。

    成本参考：5,792 个 chunk 约 2 秒。数据量上到十万级时这会是启动瓶颈，
    届时应把 BM25 索引 pickle 到磁盘，或只在首次请求时懒加载。
    """
    return HybridRetriever(
        chroma=store,
        bm25=BM25Index(load_all_chunks(store)),
        recall_n=recall_n,
        rrf_k=rrf_k,
    )


class HybridRetriever:
    """BM25 + 向量 + RRF。接口与 ChromaStore 一致，pipeline 无感知。"""

    name = "hybrid"

    def __init__(
        self,
        chroma: ChromaStore,
        bm25: BM25Index,
        recall_n: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self.chroma = chroma
        self.bm25 = bm25
        self.recall_n = recall_n
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return self.chroma.size

    @property
    def is_chunked(self) -> bool:
        return self.chroma.is_chunked

    @property
    def collection(self):
        return self.chroma.collection

    # ------------------------------------------------------------------
    def _dense_chunks(self, query: str, n: int) -> list[tuple[str, float]]:
        """稠密路：返回 chunk 级 [(chunk_id, cosine_score)]，先不去重。"""
        result = self.chroma.collection.query(
            query_embeddings=[embed_query(query)],
            n_results=min(n, self.size),
            include=["distances"],
        )

        ids = result.get("ids") or [[]]
        distances = result.get("distances") or [[]]

        # 距离转相似度，只为日志可读；RRF 只吃名次，用不到这个分数
        return [(cid, 1.0 - float(d)) for cid, d in zip(ids[0], distances[0])]

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        dense = self._dense_chunks(query, self.recall_n)
        sparse = self.bm25.search(query, top_k=self.recall_n)

        if not dense and not sparse:
            return []

        fused = rrf_fuse(
            [[cid for cid, _ in dense], [cid for cid, _ in sparse]],
            k=self.rrf_k,
        )

        # 按 pid 聚合：同一篇文档只保留 RRF 分数最高的那一段
        best: dict[str, tuple[str, float]] = {}
        for chunk_id, score in fused.items():
            pid = chunk_id.split("::")[0]
            if pid not in best or score > best[pid][1]:
                best[pid] = (chunk_id, score)

        ordered = sorted(best.values(), key=lambda item: item[1], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for chunk_id, score in ordered:
            pid = chunk_id.split("::")[0]
            text = self.bm25.texts.get(chunk_id, "")

            # 标题带上段号，和 ChromaStore 的展示保持一致
            if "::" in chunk_id:
                seg = chunk_id.split("::", 1)[1]
                title = f"pid {pid} · 第 {int(seg) + 1} 段"
            else:
                title = f"pid {pid}"

            results.append(
                SearchResult(
                    document=Document(doc_id=pid, title=title, text=text),
                    score=score,
                )
            )

        return results

    # ------------------------------------------------------------------
    def info(self) -> dict:
        base = self.chroma.info()

        return {
            **base,
            # 名字必须唯一：一个库里可能同时有 dense 和 hybrid 两个后端
            "name": f"{self.chroma.collection.name}+bm25",
            "label": f"Hybrid · BM25 + 向量 + RRF(k={self.rrf_k})",
            "short": "混合",
            "retriever": "hybrid",
            "recall_n": self.recall_n,
            "rrf_k": self.rrf_k,
            "dense_model": base.get("model", ""),
            "sparse_model": f"BM25(k1={self.bm25.k1}, b={self.bm25.b}, 词表 {len(self.bm25.postings):,})",
            "model": f"bge-small-zh + BM25(k1={self.bm25.k1}, b={self.bm25.b})",
        }
