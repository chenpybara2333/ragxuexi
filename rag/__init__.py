"""RAG 项目：检索增强生成流水线（当前阶段：向量检索）。

在线问答的数据流：
    bm25         问题文本      -> 倒排索引命中（字面精确，号码/型号靠它）
    embedder     问题文本      -> 512 维稠密向量（bge-small-zh-v1.5）
    chroma_store 查询向量      -> HNSW 近邻
    hybrid       两路结果      -> RRF 融合 + 按 pid 去重 -> list[SearchResult]
    prompt       SearchResult  -> OpenAI 格式的 messages
    llm          messages      -> 回答文本
    pipeline     把以上串起来，返回结构化中间态（CLI 与 Web 共用同一份逻辑）

离线灌库（scripts/ingest_chroma.py）：
    collection.tsv -> chunker 切分 -> embedder 向量化 -> chroma_store 落盘

注：阶段 1 的 TF-IDF 检索器（loader.py / retriever.py）已随 txt 知识库一起移除，
    需要回顾稀疏检索时看 git 历史。
"""

__all__ = [
    "schema",
    "chunker",
    "embedder",
    "chroma_store",
    "bm25",
    "hybrid",
    "tokenizer",
    "prompt",
    "llm",
    "config",
    "pipeline",
]
