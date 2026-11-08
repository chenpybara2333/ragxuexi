"""多查询检索：改写出若干变体，各自检索，再用 RRF 二次融合。

结构上是"检索器的包装器"：里面套一个能用的检索器（比如 HybridRetriever），
对外仍然暴露 search / size / info，所以 server 和 pipeline 都不用改。

两级 RRF：
    第一级  每个变体内部：BM25 路 + 向量路 -> RRF（由被包装的 hybrid 完成）
    第二级  多个变体之间：各变体的有序 pid 列表 -> 再来一次 RRF

为什么第二级用 pid 级而不是 chunk 级？
    第一级出来已经是 pid 级结果。要在 chunk 级做二级融合，就得让 base 暴露
    未去重的候选，接口会复杂不少。而最终产物本来就是 pid 级，
    在 pid 级融合与 chunk 级融合的差异很小，不值得为此增加耦合。
"""

from collections import defaultdict

from rag.query_rewrite import rewrite


class MultiQueryRetriever:
    """把一个问题拆成多个查询，分别检索后融合。"""

    name = "multiquery"

    def __init__(self, base, n_variants: int = 3, rrf_k: int = 60) -> None:
        self.base = base
        self.n_variants = n_variants
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return self.base.size

    @property
    def is_chunked(self) -> bool:
        return getattr(self.base, "is_chunked", False)

    @property
    def collection(self):
        return getattr(self.base, "collection", None)

    # ------------------------------------------------------------------
    def search_with_trace(self, query: str, top_k: int = 3):
        """检索并返回 (结果, 过程信息)。

        过程信息里带改写出的变体，前端可以展示"它到底拿什么去搜的"——
        这一步不透明的话，检索效果变差时根本无从排查。
        """
        variants = rewrite(query, n=self.n_variants)

        # 原问题永远保留一路：改写可能更差，留个兜底
        queries = [query] + variants

        pool = max(top_k * 3, 10)
        rankings: list[list[str]] = []
        per_variant: list[dict] = []
        best: dict[str, object] = {}

        for q in queries:
            results = self.base.search(q, top_k=pool)

            pids = [r.document.doc_id for r in results]
            rankings.append(pids)
            per_variant.append(
                {
                    "query": q,
                    "top_pids": pids[:5],
                    "is_original": q == query,
                }
            )

            for r in results:
                pid = r.document.doc_id
                if pid not in best:
                    best[pid] = r

        # 二级 RRF：只看每个 pid 在各变体里的名次
        fused: dict[str, float] = defaultdict(float)
        for ranking in rankings:
            for rank, pid in enumerate(ranking, start=1):
                fused[pid] += 1.0 / (self.rrf_k + rank)

        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results = []
        for pid, score in ordered:
            if pid not in best:
                continue
            r = best[pid]
            # 复用原 SearchResult 的 document，只替换分数为融合分
            results.append(type(r)(document=r.document, score=score))

        return results, {
            "original": query,
            "variants": variants,
            "queries_used": queries,
            "per_variant": per_variant,
        }

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 3):
        results, _ = self.search_with_trace(query, top_k)
        return results

    # ------------------------------------------------------------------
    def info(self) -> dict:
        base = self.base.info()

        return {
            **base,
            "name": f"{getattr(self.collection, 'name', 'unknown')}+rewrite",
            "label": f"Multi-Query · {self.n_variants} 个变体 + RRF",
            "short": "改写",
            "retriever": "multiquery",
            "model": f"{base.get('model', '')} + LLM 改写",
            "n_variants": self.n_variants,
        }
