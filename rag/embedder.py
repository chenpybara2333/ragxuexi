"""Embedding 封装：把文本变成稠密向量。

为什么自己算 embedding，不用 Chroma 自带的默认函数？
  1. Chroma 默认是 all-MiniLM-L6-v2，纯英文模型。我们这批语料是中文，
     用它检索中文等于用英文字典查中文词，效果会明显掉档。
  2. 更重要的是：embedding 是 RAG 里最该被"看见"的一环。
     藏在框架默认值里，你永远不知道向量是怎么来的、维度是多少、换模型会怎样。

模型：BAAI/bge-small-zh-v1.5
  512 维，CPU 可跑，中文检索里性价比最高的小模型之一。
  对比：bge-base-zh 768 维更准但慢 3 倍；OpenAI text-embedding-3-small 1536 维要钱。

关于 bge 的 query 指令前缀：
  bge v1.5 官方说明「检索场景不再需要 instruction」，所以这里不加前缀。
  如果你换回 v1.0，query 侧要加 "为这个句子生成表示以用于检索相关文章："，否则召回会掉。

下载：默认走 hf-mirror 镜像，因为 huggingface.co 在多数国内网络下不可达。
"""

import os

# 必须在 import huggingface_hub 之前设置，否则镜像不生效
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path

from fastembed import TextEmbedding

MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

# 模型文件缓存位置（首次约 100MB，之后复用）
CACHE_DIR = Path(__file__).resolve().parent.parent / ".model_cache"

_model: TextEmbedding | None = None


def get_model() -> TextEmbedding:
    """懒加载：第一次用到时才下载/加载模型，避免 import 就卡几秒。"""
    global _model
    if _model is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR))
    return _model


def embed_documents(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """批量把文档转成向量。

    返回 list[list[float]]，外层长度 = len(texts)，内层长度 = 模型维度（512）。
    注意：超过模型最大长度（512 token）的部分会被截断，这是短文本检索模型的天性，
    也是阶段 2 强调"切分"的原因之一。
    """
    if not texts:
        return []
    vectors = list(get_model().embed(texts, batch_size=batch_size))
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """把查询转成向量。单条，走 query_embed 以提高性能。"""
    vector = next(iter(get_model().query_embed(text)))
    return vector.tolist()


def model_info() -> dict:
    """给前端展示用的模型元信息。"""
    return {
        "model": MODEL_NAME,
        "dim": get_dim(),
        "provider": "fastembed (ONNX, CPU)",
    }


_dim_cache: int | None = None


def get_dim() -> int:
    """模型维度，通过一次真实推理拿到，不写死（换模型时不用改代码）。"""
    global _dim_cache
    if _dim_cache is None:
        _dim_cache = len(embed_query("维度探测"))
    return _dim_cache
