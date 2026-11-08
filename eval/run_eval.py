"""跑评测：对问题集里的每条问题做检索，算 Recall@K 与 MRR。

为什么要有这个脚本？
    没有基线数字，任何"我觉得更准了"都是感觉，不是结论。
    改切分、换模型、加 BM25，前后各跑一次，看数字说话。

指标说明：
    Recall@K  正确答案所在 pid 是否出现在前 K 条里。衡量"能不能捞到"。
    MRR@K     命中排名的倒数均值（第 1 名得 1，第 2 名得 0.5…）。衡量"排得够不够靠前"。
    两个都要看：Recall 高但 MRR 低，说明捞到了却排在末尾，进了 Prompt 也可能被忽略。

用法：
    python eval/run_eval.py
    python eval/run_eval.py --k 5 --collection corpus_chunk_384_64
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.chroma_store import ChromaStore  # noqa: E402

PERSIST = ROOT / "vector_store" / "chroma"


def load_questions(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[错误] 找不到问题集：{path}")
        print("       先生成：python eval/generate_questions.py --n 60")
        raise SystemExit(1)

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def ngrams(text: str, n: int = 5) -> set[str]:
    """字符 n-gram 集合，用来判断两段文字是否同源。"""
    s = "".join(text.split())
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def overlap_ratio(a: str, b: str, n: int = 5) -> float:
    """a 的 n-gram 有多大比例出现在 b 里。"""
    A, B = ngrams(a, n), ngrams(b, n)
    if not A:
        return 0.0
    return len(A & B) / len(A)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索评测")
    parser.add_argument("--file", default=str(ROOT / "eval" / "questions.jsonl"))
    parser.add_argument("--collection", default="corpus_chunk_384_64")
    parser.add_argument("--k", type=int, default=5, help="看前 K 条")
    parser.add_argument("--save", default=str(ROOT / "eval" / "report.md"))
    parser.add_argument("--show-fail", type=int, default=8, help="打印多少条失败样例")
    args = parser.parse_args()

    questions = load_questions(Path(args.file))
    store = ChromaStore(PERSIST, args.collection)

    print(f"[评测] 集合 {args.collection}｜{store.size:,} 条｜问题 {len(questions)} 条｜K={args.k}")
    print(f"[模型] {store.info()['model']}\n")

    records = []
    started = time.perf_counter()

    for q in questions:
        results = store.search(q["question"], top_k=args.k)
        pids = [r.document.doc_id for r in results]

        gold = str(q["gold_pid"])
        rank = pids.index(gold) + 1 if gold in pids else 0

        records.append(
            {
                **q,
                "rank": rank,
                "top_pids": pids,
                "top_scores": [round(r.score, 3) for r in results],
                "top_texts": [r.document.text for r in results],
            }
        )

    elapsed = time.perf_counter() - started

    # ---------- 同源检测：识别"标注粒度问题"造成的假失败 ----------
    # 语料是按 pid 逐条存的，同一篇长文会被切成连续 pid。
    # 问的是 1358，检索命中 1351（同一篇文章的上一段），pid 不同但内容同源。
    # 这不该算检索失败，否则基线会被系统性地压低。
    gold_texts: dict[str, str] = {}
    gold_ids = [r["gold_chunk"] for r in records if r.get("gold_chunk")]
    if gold_ids:
        fetched = store.collection.get(ids=gold_ids, include=["documents"])
        gold_texts = dict(zip(fetched["ids"], fetched["documents"]))

    near_miss = 0
    for r in records:
        if r["rank"] != 0:
            r["same_source"] = False
            continue

        gold_text = gold_texts.get(r.get("gold_chunk", ""), "")
        if not gold_text:
            r["same_source"] = False
            continue

        best = max((overlap_ratio(gold_text, t) for t in r["top_texts"]), default=0.0)
        r["same_source"] = best >= 0.5
        r["same_source_score"] = round(best, 2)
        near_miss += r["same_source"]

    # ---------- 汇总 ----------
    def summarize(rows: list[dict]) -> dict:
        n = len(rows) or 1
        return {
            "n": len(rows),
            "r1": sum(1 for r in rows if r["rank"] == 1) / n,
            "r3": sum(1 for r in rows if 0 < r["rank"] <= 3) / n,
            "rk": sum(1 for r in rows if 0 < r["rank"] <= args.k) / n,
            "mrr": sum((1 / r["rank"]) for r in rows if r["rank"]) / n,
        }

    overall = summarize(records)
    by_type: dict[str, dict] = {}
    for t in sorted({r.get("type", "unknown") for r in records}):
        by_type[t] = summarize([r for r in records if r.get("type", "unknown") == t])

    # ---------- 打印 ----------
    header = f"{'分组':<12}{'条数':>6}{'Recall@1':>11}{'Recall@3':>11}{f'Recall@{args.k}':>11}{'MRR':>9}"
    print(header)
    print("-" * len(header))
    print(
        f"{'总体':<12}{overall['n']:>6}{pct(overall['r1']):>11}{pct(overall['r3']):>11}"
        f"{pct(overall['rk']):>11}{overall['mrr']:>9.3f}"
    )
    for t, s in by_type.items():
        print(
            f"{t:<12}{s['n']:>6}{pct(s['r1']):>11}{pct(s['r3']):>11}"
            f"{pct(s['rk']):>11}{s['mrr']:>9.3f}"
        )

    print(f"\n耗时 {elapsed:.1f}s（{elapsed / max(1, len(records)) * 1000:.0f} ms/条）")

    # ---------- 失败样例 ----------
    fails = [r for r in records if r["rank"] == 0]
    real_fails = [r for r in fails if not r.get("same_source")]

    if fails:
        print(
            f"\n未命中 {len(fails)} 条：其中 {near_miss} 条与返回内容高度重合（重复内容导致的假失败），"
            f"真实失败 {len(real_fails)} 条"
        )

        if near_miss:
            print("  ── 同源误判（不算检索错误）──")
            for r in [x for x in fails if x.get("same_source")]:
                print(f"  [{r.get('type', '?'):<8}] {r['question']}")
                print(f"      期望 pid {r['gold_pid']}｜实际 {r['top_pids']}｜内容重合 {r['same_source_score']:.0%}")

        if real_fails and args.show_fail:
            print(f"  ── 真实失败（展示前 {min(args.show_fail, len(real_fails))} 条）──")
            for r in real_fails[: args.show_fail]:
                print(f"  [{r.get('type', '?'):<8}] {r['question']}")
                print(f"      期望 pid {r['gold_pid']}｜实际 {r['top_pids']} 分数 {r['top_scores']}")

    # ---------- 落盘报告 ----------
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 检索评测报告",
        "",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 集合：`{args.collection}`（{store.size:,} 条）",
        f"- 模型：{store.info()['model']}（{store.info()['dim']} 维）",
        f"- 问题集：`{Path(args.file).name}`（{len(records)} 条）",
        f"- K = {args.k}",
        "",
        "| 分组 | 条数 | Recall@1 | Recall@3 | Recall@K | MRR |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 总体 | {overall['n']} | {pct(overall['r1'])} | {pct(overall['r3'])} | "
        f"{pct(overall['rk'])} | {overall['mrr']:.3f} |",
    ]
    for t, s in by_type.items():
        lines.append(
            f"| {t} | {s['n']} | {pct(s['r1'])} | {pct(s['r3'])} | {pct(s['rk'])} | {s['mrr']:.3f} |"
        )

    if fails:
        lines += [
            "",
            f"## 未命中（{len(fails)} 条）",
            "",
            f"其中 **{near_miss} 条**与返回内容高度重合（重复内容导致的假失败，"
            f"本次检测未发现）；**真实失败 {len(real_fails)} 条**。",
            "",
        ]

        if near_miss:
            lines += ["### 同源误判（不算检索错误）", ""]
            for r in [x for x in fails if x.get("same_source")]:
                lines.append(
                    f"- `[{r.get('type', '?')}]` {r['question']}　→ 期望 `pid {r['gold_pid']}`，"
                    f"实际 `{r['top_pids']}`，内容重合 {r['same_source_score']:.0%}"
                )
            lines.append("")

        if real_fails:
            lines += ["### 真实失败", ""]
            for r in real_fails[:20]:
                lines.append(
                    f"- `[{r.get('type', '?')}]` {r['question']}　→ 期望 `pid {r['gold_pid']}`，"
                    f"实际 `{r['top_pids']}`"
                )

    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入 {save_path}")


if __name__ == "__main__":
    main()
