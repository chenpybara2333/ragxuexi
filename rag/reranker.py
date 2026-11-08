"""Cross-Encoder 重排。

和已有的两路检索有什么不同？
    BM25 和向量都是「双塔」：问题和文档各自编码，最后算一个相似度。
    好处是能预先建索引、检索快；坏处是问题和文档之间没有交互，
    细粒度的匹配（"这段到底有没有回答这个问题"）看不出来。

    Cross-Encoder 把（问题, 文档）拼在一起送进模型，逐对打分。
    精度高得多，但没法预建索引——每条候选都要现算一次，所以只能用在
    召回之后，对少量候选做精排。这是典型的「粗排 + 精排」两段式。

为什么它不算"LLM 改写"？
    它是判别式模型，输入一对文本、输出一个相关性分数，全程不生成任何文字。
    没有幻觉风险，延迟也稳定（本地 ONNX，无网络依赖）。

延迟参考（bge-reranker-base，CPU）：
    约 45-50 ms / 条。重排 20 条约 1 秒。
    所以候选数是个需要权衡的参数：开大了准，但慢。
"""

import math
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

MODEL_NAME = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
CACHE_DIR = Path(__file__).resolve().parent.parent / ".model_cache"

_encoder: TextCrossEncoder | None = None


def get_encoder() -> TextCrossEncoder:
    """懒加载。首次会下载模型（约 1.1 GB），之后走本地缓存。"""
    global _encoder
    if _encoder is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _encoder = TextCrossEncoder(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR))
    return _encoder


def sigmoid(x: float) -> float:
    """把原始 logit 压到 (0,1)，只为了前端显示好看，不影响排序。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def rerank(query: str, documents: list[str]) -> list[float]:
    """给若干文档打分，返回原始 logit（越大越相关）。顺序与输入一致。"""
    if not documents:
        return []
    return list(get_encoder().rerank(query, documents))


class RerankRetriever:
    """包装任意检索器：先粗排取 N 条，再用 Cross-Encoder 精排取 top_k。

    两级结构：
        base.search(query, top_n)  ->  N 条候选（BM25 + 向量融合的结果）
        cross-encoder 逐条打分      ->  按精排分重新排序，取 top_k
    """

    name = "rerank"

    def __init__(self, base, top_n: int = 20) -> None:
        self.base = base
        self.top_n = top_n

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
    def search(self, query: str, top_k: int = 3):
        candidates = self.base.search(query, top_k=self.top_n)
        if not candidates:
            return []

        docs = [c.document.text for c in candidates]
        raw = rerank(query, docs)

        order = sorted(range(len(candidates)), key=lambda i: raw[i], reverse=True)[:top_k]

        return [
            type(candidates[i])(
                document=candidates[i].document,
                # 原始 logit 可能是 -10 到 +5，压到 0-1 只为前端可读，排序已在上一步完成
                score=sigmoid(raw[i]),
            )
            for i in order
        ]

    # ------------------------------------------------------------------
    def info(self) -> dict:
        base = self.base.info()

        return {
            **base,
            "name": f"{getattr(self.collection, 'name', 'unknown')}+rerank",
            "label": f"Hybrid + 重排（top{self.top_n} 精排）",
            "short": "重排",
            "retriever": "rerank",
            "rerank_model": MODEL_NAME,
            "rerank_n": self.top_n,
            "model": f"{base.get('model', '')} + {MODEL_NAME.split('/')[-1]}",
        }
