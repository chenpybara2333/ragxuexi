"""把 collection.tsv 灌进 Chroma 向量库。

用法：
    # 整篇入库（一条记录 = 一个向量）
    python scripts/ingest_chroma.py --limit 20000

    # 切分入库（一条记录 = 多个 chunk），推荐
    python scripts/ingest_chroma.py --limit 20000 --chunked

    # 全量 + 自定义集合名 + 重建
    python scripts/ingest_chroma.py --all --chunked --collection corpus_chunk --rebuild

为什么默认只灌 5 万条？
    230 万条 × 512 维 × 4 字节 ≈ 4.7 GB 光向量，CPU 上编码要按小时算。
    先用一个子集把"入库 -> 检索 -> 前端展示"整条链跑通，再决定要不要全量。

断点续传：
    进度记在 vector_store/.ingest_progress.json，按【已处理的文档序号】记，
    不是按集合里有多少条。因为切分模式下一条文档会产生多个 chunk，
    collection.count() 和"处理到第几条文档"完全对不上。
"""

import argparse
import json
import sys
import time
from pathlib import Path

# 让脚本能 import rag 包（scripts/ 在子目录里）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.chroma_store import ChromaStore  # noqa: E402
from rag.chunker import chunk_text  # noqa: E402
from rag.embedder import embed_documents  # noqa: E402

TSV_PATH = ROOT / "collection.tsv"
PERSIST_DIR = ROOT / "vector_store" / "chroma"
PROGRESS_FILE = ROOT / "vector_store" / ".ingest_progress.json"


def clean_text(raw: str) -> str:
    """清掉语料里的 HTML 残渣和多余空白。

    这批数据里混着 <br>、<img> 这类标签，直接拿去算 embedding
    会把"标签符号"也算进语义，纯属噪音。
    """
    text = raw.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    # 去掉 <img> <a href> 这类成对/自闭合标签
    out = []
    in_tag = False
    for ch in text:
        if ch == "<":
            in_tag = True
            continue
        if ch == ">":
            in_tag = False
            continue
        if not in_tag:
            out.append(ch)
    text = "".join(out)
    # 折叠所有空白
    return " ".join(text.split())


def iter_records(limit: int | None):
    """流式读取 tsv，逐条产出 (pid, text)。

    用流式而不是 pandas.read_csv：3.6 GB 一次性读进内存会直接打爆。
    """
    with open(TSV_PATH, encoding="utf-8", errors="replace") as f:
        header = f.readline()
        if not header.startswith("pid"):
            print(f"[警告] 表头看起来不对：{header[:40]!r}，仍按 pid\\ttext 解析")

        count = 0
        for line in f:
            if limit is not None and count >= limit:
                return
            pid, sep, text = line.rstrip("\n").partition("\t")
            if not sep or not pid.isdigit():
                continue
            cleaned = clean_text(text)
            if len(cleaned) < 10:      # 太短的没有检索价值
                continue
            count += 1
            yield pid, cleaned


# ----------------------------------------------------------------------
# 进度文件：{集合名: {"next_index": 文档序号, "mode": ..., "chunks": 累计块数}}
# ----------------------------------------------------------------------
def load_progress(collection: str) -> dict:
    if not PROGRESS_FILE.exists():
        return {}
    try:
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        return data.get(collection, {})
    except (json.JSONDecodeError, OSError):
        return {}


def save_progress(collection: str, payload: dict) -> None:
    data = {}
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    data[collection] = payload
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="collection.tsv -> Chroma")
    parser.add_argument("--limit", type=int, default=50000, help="最多灌多少条文档（默认 5 万）")
    parser.add_argument("--all", action="store_true", help="灌全量（忽略 --limit）")
    parser.add_argument("--batch", type=int, default=256, help="每批写入条数")
    parser.add_argument("--rebuild", action="store_true", help="先清空集合并重置进度再灌")
    parser.add_argument("--collection", help="集合名（默认按模式自动命名）")

    # 切分相关
    parser.add_argument("--chunked", action="store_true", help="按 chunk 切分入库，而不是整篇入库")
    parser.add_argument("--chunk-size", type=int, default=384, help="每个 chunk 的 token 上限")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="相邻 chunk 重叠的 token 数")
    args = parser.parse_args()

    if args.chunked and args.chunk_overlap >= args.chunk_size:
        print("[错误] --chunk-overlap 必须小于 --chunk-size，否则窗口不会前进")
        raise SystemExit(1)

    limit = None if args.all else args.limit

    if not TSV_PATH.exists():
        print(f"[错误] 找不到语料文件：{TSV_PATH}")
        raise SystemExit(1)

    # 集合名：不指定就按模式生成，避免两种形态写进同一个集合
    collection_name = args.collection or (
        f"corpus_chunk_{args.chunk_size}_{args.chunk_overlap}"
        if args.chunked
        else "ecommerce_zh"
    )

    # 灌库是写路径，允许"不存在就创建"
    store = ChromaStore(PERSIST_DIR, collection_name, create=True)

    if args.rebuild:
        store.client.delete_collection(collection_name)
        print(f"[重建] 已删除集合 {collection_name}")
        store = ChromaStore(PERSIST_DIR, collection_name, create=True)
        save_progress(collection_name, {})

    mode_desc = (
        f"切分（size={args.chunk_size} overlap={args.chunk_overlap}）"
        if args.chunked
        else "整篇"
    )
    print(f"[模式] {mode_desc}")
    print(f"[集合] {collection_name}（当前 {store.size:,} 条）")
    print(f"[开始] 目标 {limit if limit else '全量'} 篇文档，批大小 {args.batch}")
    print(f"[位置] {PERSIST_DIR}\n")

    # ---------- 续传 ----------
    progress = load_progress(collection_name)
    resume_at = int(progress.get("next_index", 0)) if store.size > 0 else 0

    if store.size > 0 and resume_at == 0:
        print("[警告] 集合非空但没有进度记录，将从头重跑（id 固定，upsert 会覆盖，不会重复）")
    if resume_at:
        print(f"[续传] 已完成 {resume_at:,} 篇文档，从第 {resume_at:,} 篇继续")

    started = time.perf_counter()
    docs_done = resume_at
    chunks_written = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    batch_meta: list[dict] = []
    last_report = time.perf_counter()

    def flush() -> None:
        nonlocal chunks_written
        if not batch_ids:
            return
        vectors = embed_documents(batch_texts)
        store.collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=vectors,
            metadatas=batch_meta,
        )
        chunks_written += len(batch_ids)
        batch_ids.clear()
        batch_texts.clear()
        batch_meta.clear()

        # 进度按文档记，并且必须包含已写入的块，防止中途崩溃丢进度
        save_progress(
            collection_name,
            {
                "next_index": docs_done,
                "mode": "chunked" if args.chunked else "whole",
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "chunks_written": chunks_written + int(progress.get("chunks_written", 0)),
            },
        )

    for index, (pid, text) in enumerate(iter_records(limit)):
        if index < resume_at:
            continue

        if args.chunked:
            pieces = chunk_text(text, size=args.chunk_size, overlap=args.chunk_overlap)
            for chunk in pieces:
                batch_ids.append(f"{pid}::{chunk.index}")
                batch_texts.append(chunk.text)
                batch_meta.append(
                    {
                        "pid": pid,
                        "chunk": chunk.index,
                        "chars": len(chunk.text),
                        "title": f"pid {pid}",
                    }
                )
        else:
            batch_ids.append(pid)
            batch_texts.append(text)
            batch_meta.append({"pid": pid, "title": f"pid {pid}"})

        docs_done = index + 1

        if len(batch_ids) >= args.batch:
            flush()

            now = time.perf_counter()
            if now - last_report >= 2:
                elapsed = now - started
                done = docs_done - resume_at
                rate = done / elapsed if elapsed > 0 else 0
                remain = (limit - docs_done) if limit else 0
                eta = remain / rate if rate > 0 and limit else 0
                print(
                    f"  文档 {docs_done:>7,} | 写入 {store.size:>8,} 条 | "
                    f"{rate:5.1f} 篇/秒 | 已用 {elapsed:5.0f}s | 剩余约 {eta:4.0f}s",
                    flush=True,
                )
                last_report = now

    flush()

    elapsed = time.perf_counter() - started
    docs_this_run = docs_done - resume_at

    print(f"\n[完成] 本次处理 {docs_this_run:,} 篇文档，写入 {chunks_written:,} 条")
    print(f"[结果] 集合 {collection_name} 共 {store.size:,} 条，耗时 {elapsed:.1f}s")

    if docs_this_run > 0:
        print(f"[膨胀] 平均每篇 {chunks_written / docs_this_run:.2f} 条")

    dim = len(embed_documents(["维度探测"])[0])
    print(f"[维度] 每条向量 {dim} 维，估算向量占用 {store.size * dim * 4 / 1024 / 1024:.1f} MB")
    print("\n服务启动时会自动注册这个集合，无需改代码。")


if __name__ == "__main__":
    main()
