"""阶段 1 入口：终端里跑的一次完整 RAG 问答。

用法：
    python main.py                          # 进入交互模式
    python main.py "商品支持七天无理由退货吗"   # 只问一个问题
    python main.py "运费多少" --topk 3 --debug # 看检索得分
"""

import argparse
import sys
from pathlib import Path

from rag.chroma_store import ChromaStore, list_collections
from rag.config import DEFAULT_TOP_K
from rag.llm import is_llm_configured
from rag.pipeline import run_rag


def answer(
    question: str,
    retriever: ChromaStore,
    top_k: int,
    debug: bool = False,
) -> None:
    """执行一次完整流程并打印。

    真正的检索 + 生成逻辑在 rag/pipeline.py 里，CLI 和 Web 共用那一份。
    这里只负责"把结果按终端的样子摆出来"。
    """
    result = run_rag(question, retriever, top_k)

    # ---------- 1. Retrieval ----------
    print("\n[检索] 命中资料：")
    for item in result["retrieval"]["candidates"]:
        if not item["selected"]:
            continue
        score = item["score"]
        # 用方块直观显示分数高低；分数为 0 时不画，避免误导
        bar = "" if score <= 0 else "█" * max(1, round(score * 30))
        score_text = f"{score:.4f}" if debug else f"{score:.2f}"
        print(f"  {item['rank']}. {item['title']}   相似度 {score_text}  {bar}")

    if debug:
        # 看看问题被切成了什么 token，这是排查"检索不到"的第一手线索
        print(f"\n[调试] 问题分词：{result['retrieval']['query_tokens']}")
        print("[调试] Prompt 全文如下（前 600 字）：")
        print(result["prompt"]["user"][:600] + " ...\n")

    # ---------- 2. Generation ----------
    print("\n[回答]")
    if result["answer"]["error"]:
        print(f"  调用失败：{result['answer']['error']}")
        print("  已降级为模拟模式输出，避免中断学习流程：\n")
    print(result["answer"]["text"])
    print()


def build_retriever(collection: str | None = None):
    """打开一个向量库集合。不指定就优先用切片集合。

    TF-IDF 后端已移除（原始 txt 知识库不再维护），所以这里只认 Chroma。
    """
    persist = Path(__file__).resolve().parent / "vector_store" / "chroma"

    cols = [c for c in list_collections(persist) if c["count"] > 0]
    if not cols:
        print("[错误] 向量库为空，请先灌库：")
        print("       python scripts/ingest_chroma.py --limit 2000 --chunked")
        raise SystemExit(1)

    names = [c["name"] for c in cols]
    target = collection or next((n for n in names if "chunk" in n), names[0])

    if target not in names:
        print(f"[错误] 集合 {target!r} 不存在，可用：{', '.join(names)}")
        raise SystemExit(1)

    store = ChromaStore(persist, target)
    tag = "切片" if store.is_chunked else "整篇"
    print(f"[启动] 集合 {target}：{store.size:,} 条（{tag}）")
    return store


def interactive(retriever, top_k: int, debug: bool) -> None:
    """终端交互循环。"""
    print("\n输入问题开始提问，输入 :q 退出，:help 查看命令。\n")

    while True:
        try:
            question = input("你问 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not question:
            continue

        if question in {":q", ":quit", "exit", "quit"}:
            print("再见。")
            break

        if question == ":help":
            print("  :q        退出\n  :debug    查看分词与 Prompt 全文\n  :topk N   设置召回数量")
            continue

        if question.startswith(":topk"):
            parts = question.split()
            if len(parts) == 2 and parts[1].isdigit():
                top_k = int(parts[1])
                print(f"  召回数量已设为 {top_k}")
            else:
                print("  用法：:topk 3")
            continue

        if question == ":debug":
            debug = not debug
            print(f"  调试模式：{'开' if debug else '关'}")
            continue

        answer(question, retriever, top_k, debug)


def main() -> None:
    # Windows 控制台默认 GBK，遇到特殊字符可能报 UnicodeEncodeError，这里强制 utf-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="阶段 1：最小 RAG 问答 Demo")
    parser.add_argument("question", nargs="?", help="直接提问（不填则进入交互模式）")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K, help="召回文档数量")
    parser.add_argument("--debug", action="store_true", help="打印分词、Prompt 与完整得分")
    parser.add_argument("--collection", help="指定集合名，默认优先用切片集合")
    args = parser.parse_args()

    retriever = build_retriever(args.collection)

    if not is_llm_configured():
        print(
            "\n[提示] 未检测到 LLM_API_KEY，将以【模拟模式】运行。\n"
            "       模拟模式只摘抄检索结果，不会真正生成回答。\n"
            "       要体验真实效果：复制 .env.example 为 .env，填入你的 API Key。"
        )

    if args.question:
        answer(args.question, retriever, args.topk, args.debug)
    else:
        interactive(retriever, args.topk, args.debug)


if __name__ == "__main__":
    main()
