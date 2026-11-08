"""Chroma 向量库封装。

和 TfidfRetriever 保持同一个调用接口（search / size / info），
这样 pipeline.py 不需要知道背后到底是 TF-IDF 还是向量库 —— 这叫「面向接口编程」，
换成 FAISS / Milvus / Qdrant 时，只换这个文件。
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings

from rag.embedder import embed_query
from rag.schema import Document, SearchResult

# 集合（collection）相当于关系库里的"表"，Chroma 里一个集合独立建一套 HNSW 索引
COLLECTION_NAME = "ecommerce_zh"


def _client(persist_dir: str | Path) -> "chromadb.ClientAPI":
    """统一的客户端工厂。

    坑：Chroma 对同一个持久化路径只允许一种 Settings。
    如果这里用 anonymized_telemetry=False，别处又用 allow_reset=True 打开同一目录，
    第二次创建会直接抛 "An instance of Chroma already exists ... with different settings"，
    而且往往被上层 try 吞掉，表现成"集合莫名其妙加载不出来"。
    所以全项目只走这一个入口。
    """
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )


class ChromaStore:
    """Chroma 持久化向量库的薄封装。"""

    name = "chroma"

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = COLLECTION_NAME,
        create: bool = False,
    ) -> None:
        """打开一个集合。

        Args:
            create: True 才允许"不存在就创建"。
                    只读场景务必保持 False —— 否则集合名写错时不会报错，
                    而是悄悄建一个空集合，你看到的是"检索 0 条"而不是"集合不存在"。
                    这个坑排查起来极费时间，所以默认值设成 False：
                    读的地方要么集合真的存在，要么立刻抛异常。
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 切片判定的缓存位（None = 还没查过）
        self._chunked: bool | None = None

        # PersistentClient = 数据落盘，进程退出不丢。统一走 _client()，避免 Settings 冲突。
        self.client = _client(self.persist_dir)

        # cosine 空间：bge 的向量已归一化，用余弦最自然。
        # 注意 metadata 里这个键只在【首次创建】集合时生效，改建后要删库重建。
        if create:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            try:
                self.collection = self.client.get_collection(name=collection_name)
            except Exception as exc:  # chromadb 的 NotFoundError，不同版本类型名不一致
                available = [c.name for c in self.client.list_collections()]
                raise FileNotFoundError(
                    f"集合 {collection_name!r} 不存在于 {self.persist_dir}。"
                    f"现有集合：{available or '（空库）'}。"
                    f"如果是首次灌库，请跑 scripts/ingest_chroma.py"
                ) from exc

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        """库里有多少条向量。"""
        return self.collection.count()

    @property
    def is_chunked(self) -> bool:
        """这个集合存的是切片还是整篇？

        判据：切片写入时 metadata 里会带 chunk 序号。
        结果缓存，避免每次 info() 都多打一次库。
        """
        if self._chunked is None:
            sample = self.collection.get(limit=1, include=["metadatas"])
            metas = sample.get("metadatas") or []
            self._chunked = bool(metas) and "chunk" in (metas[0] or {})
        return self._chunked

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """向量检索。

        流程：query 文本 -> 512 维向量 -> HNSW 近似最近邻 -> 按 pid 去重 -> top_k

        为什么要去重？
            一个文档被切成 5 片后，这 5 片的向量高度相似（还有 overlap 重叠部分）。
            不去重的话，前 3 名很可能全是同一篇文档的第 0/1/2 段，
            等于把宝贵的 Top-K 名额浪费在一篇文档上，多样性直接崩掉。
        """
        if self.size == 0:
            return []

        query_vector = embed_query(query)

        # 多取几倍，给去重留出余量：要 3 篇不同的文档，可能得先捞几十条。
        # 倍数太小会出现"要 3 条只回 2 条"——前几十个近邻全属于同一两篇文档。
        # 取得越多去重后越可能凑够，但 HNSW 的 top-k 开销也随 k 增长，这里取 20 倍并封顶 200。
        fetch_n = min(max(top_k * 20, 40), 200, self.size)

        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        ids = result.get("ids") or [[]]

        # 先按 pid 归并，每个 pid 只保留得分最高的那一片
        best: dict[str, tuple[str, str, dict, float]] = {}

        for row_id, text, meta, distance in zip(
            ids[0], documents[0], metadatas[0], distances[0]
        ):
            meta = meta or {}
            pid = str(meta.get("pid") or row_id)

            # Chroma 返回的是「距离」，越小越近。
            # cosine 距离 = 1 - 余弦相似度，所以相似度 = 1 - 距离。
            # 这一步很容易搞反，是接向量库时最常见的 bug：把距离当相似度排序。
            score = 1.0 - float(distance)

            if pid not in best or score > best[pid][3]:
                best[pid] = (pid, text or "", meta, score)

        ordered = sorted(best.values(), key=lambda item: item[3], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for pid, text, meta, score in ordered:
            # 切片要标出是第几段，否则前端看到的标题全是一样的
            if "chunk" in meta:
                title = f"pid {pid} · 第 {int(meta['chunk']) + 1} 段"
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
        """给前端展示的元信息。

        name 用集合名而不是 "chroma"，因为一个库里可能有多个集合（整篇/切分），
        前端要能区分它们，所以每个后端的标识必须唯一。
        """
        from rag.embedder import model_info

        chunked = self.is_chunked

        return {
            "name": self.collection.name,
            "label": f"Chroma · {self.collection.name}",
            "short": "切分" if chunked else "整篇",
            "chunked": chunked,
            "size": self.size,
            "persist_dir": str(self.persist_dir),
            "collection": self.collection.name,
            "space": "cosine",
            **model_info(),
        }


def list_collections(persist_dir: str | Path) -> list[dict]:
    """列出某个持久化目录下的所有集合，供服务启动时批量注册后端。"""
    path = Path(persist_dir)
    if not path.exists():
        return []

    client = _client(path)

    items = []
    for col in client.list_collections():
        meta = col.metadata or {}
        items.append(
            {
                "name": col.name,
                "count": col.count(),
                "note": meta.get("note", ""),
            }
        )
    return items
