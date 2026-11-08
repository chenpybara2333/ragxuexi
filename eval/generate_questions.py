"""从已入库的 chunk 反向生成评测问题。

思路：与其自己编题（编完还要去语料里找答案，容易对不上），
不如反过来——从库里抽一条真实 chunk，让模型基于它生成一个"用户会这么问"的问题。
这样 gold 标签天然准确：答案一定在这条 chunk 里，pid 已知。

关键约束（否则评测会失去意义）：
1. 问题不能照抄原文措辞。照抄的话 BM25 天然占优，测不出向量检索的真实水平。
2. 要标注问题类型：keyword（含专名/型号/数字）vs semantic（纯口语改写）。
   这两类的检索难度完全不同，混在一起统计会掩盖问题。
3. 纯过渡段、纯列表直接跳过——它们没有可问的事实。

用法：
    python eval/generate_questions.py --n 60
    python eval/generate_questions.py --n 10 --batch 5 --out eval/questions.try.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402

from rag.llm import LLMError, chat  # noqa: E402

PERSIST = ROOT / "vector_store" / "chroma"

PROMPT = """你是数据集构建助手。下面给你若干段真实语料，请为每一段生成一个【用户会问的问题】。

要求：
1. 问题的答案必须完整地存在于对应那段文字里。
2. 不要照抄原文措辞。用用户的口吻换一种说法。
   例如原文写"性能故障"，可以问"东西用着坏了"；原文写"配送时效"，可以问"多久能送到"。
3. 问题要具体，能唯一定位到这段内容；避免"这个怎么样"这类空泛提问。
4. 如果某段只是过渡句、纯列表、或者没有可回答的事实，跳过它（question 填 null）。
5. 给每个问题标注类型：
   - "keyword"：问题里包含具体的专有名词、型号、编号、人名、数字
   - "semantic"：纯口语化改写，不含专有名词

只输出 JSON 数组，不要任何解释、不要 Markdown 代码块：
[{{"idx": 0, "question": "...", "type": "keyword"}}, ...]

以下是语料：
{blocks}
"""


def load_chunks(collection: str) -> list[dict]:
    """把集合里的 chunk 全拉出来。"""
    client = chromadb.PersistentClient(
        path=str(PERSIST), settings=Settings(anonymized_telemetry=False)
    )
    col = client.get_collection(collection)

    total = col.count()
    out: list[dict] = []
    batch = 1000

    for offset in range(0, total, batch):
        res = col.get(
            limit=min(batch, total - offset),
            offset=offset,
            include=["documents", "metadatas"],
        )
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            meta = meta or {}
            out.append(
                {
                    "chunk_id": cid,
                    "pid": str(meta.get("pid", "")),
                    "chunk": meta.get("chunk"),
                    "text": doc or "",
                }
            )
    return out


def sample_chunks(chunks: list[dict], n: int, seed: int = 42) -> list[dict]:
    """按 pid 去重后随机抽样，保证来源多样。

    长度限制：太短没有可问的事实，太长的问题难以聚焦到唯一答案。
    """
    by_pid: dict[str, dict] = {}

    for c in chunks:
        pid = c["pid"]
        if not pid:
            continue
        length = len(c["text"])
        if not (200 <= length <= 1500):
            continue
        # 同一个 pid 只保留一条，且优先保留靠前的段（通常信息更完整）
        if pid not in by_pid:
            by_pid[pid] = c

    pool = list(by_pid.values())
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def parse_json_array(raw: str) -> list[dict]:
    """从模型输出里抠出 JSON 数组，容忍 ```json 包裹。"""
    text = raw.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("["):
                text = candidate
                break

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"输出里找不到 JSON 数组：{raw[:200]}")

    return json.loads(text[start : end + 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 RAG 评测问题集")
    parser.add_argument("--n", type=int, default=60, help="目标问题数")
    parser.add_argument("--batch", type=int, default=5, help="每次请求塞几条 chunk")
    parser.add_argument("--collection", default="corpus_chunk_384_64")
    parser.add_argument("--out", default=str(ROOT / "eval" / "questions.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[1/3] 读取集合 {args.collection} 的 chunk…")
    chunks = load_chunks(args.collection)
    print(f"      共 {len(chunks):,} 条")

    picked = sample_chunks(chunks, args.n, args.seed)
    print(f"[2/3] 按 pid 去重 + 长度过滤后抽样 {len(picked)} 条")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
        print(f"      已清空旧文件 {out_path.name}")

    print(f"[3/3] 调用模型生成问题（每批 {args.batch} 条）…\n")

    written = 0
    skipped = 0

    for start in range(0, len(picked), args.batch):
        group = picked[start : start + args.batch]
        blocks = "\n\n".join(
            f"[{i}] 语料（来源 pid {c['pid']}）：\n{c['text']}"
            for i, c in enumerate(group)
        )

        prompt = PROMPT.format(blocks=blocks)

        try:
            raw = chat([{"role": "user", "content": prompt}], temperature=0.7)
        except LLMError as exc:
            print(f"  批次 {start // args.batch + 1} 调用失败：{exc}")
            skipped += len(group)
            continue

        try:
            items = parse_json_array(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  批次 {start // args.batch + 1} 解析失败：{exc}")
            skipped += len(group)
            continue

        for item in items:
            idx = item.get("idx")
            question = (item.get("question") or "").strip()

            if not isinstance(idx, int) or not (0 <= idx < len(group)):
                continue
            if not question or question.lower() == "null" or len(question) < 5:
                skipped += 1
                continue

            src = group[idx]
            record = {
                "id": f"q{written:03d}",
                "question": question,
                "type": item.get("type", "semantic"),
                "gold_pid": src["pid"],
                "gold_chunk": src["chunk_id"],
                "source_len": len(src["text"]),
                "source_head": src["text"][:80],
            }

            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            written += 1
            print(f"  {record['id']} [{record['type']:<8}] {question}")

        print(f"      批次 {start // args.batch + 1}/{-(-len(picked) // args.batch)} 完成，累计 {written} 条", flush=True)

    print(f"\n[完成] 写入 {written} 条，跳过 {skipped} 条 → {out_path}")
    print("       下一步：python eval/run_eval.py")


if __name__ == "__main__":
    main()
