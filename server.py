"""极简 HTTP 服务：把 RAG 的中间态暴露给前端。

为什么不用 FastAPI？
    阶段 8 我们才正式学 FastAPI（异步、Pydantic 校验、依赖注入）。
    现在只需要两个接口、单机自用，标准库就够，还少装几个包。
    注意这里只是"外壳"，真正干活的是 rag/pipeline.py —— 换成 FastAPI 时，
    pipeline 那 100 行一个字都不用改。

接口：
    GET  /            前端页面
    GET  /api/kb      知识库概况（文档列表、词表大小）
    POST /api/ask     提问，返回本次 RAG 的全部中间态

启动：
    python server.py                 # 默认 8000 端口
    python server.py --port 8080
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rag.chroma_store import ChromaStore, list_collections
from rag.llm import is_llm_configured
from rag.pipeline import run_rag

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
CHROMA_DIR = ROOT_DIR / "vector_store" / "chroma"

# 可用的检索后端。key 是给前端的标识，value 是实现了 search/size/info 的对象。
# 现在只有 Chroma 各集合；TF-IDF 后端已移除（原始 txt 知识库不再维护）。
RETRIEVERS: dict[str, object] = {}
DEFAULT_BACKEND = ""


class RAGHandler(BaseHTTPRequestHandler):
    server_version = "RAGStage1/1.0"

    # ---------------- 路由 ----------------
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path == "/api/kb":
            self._send_json(
                {
                    "backends": [r.info() for r in RETRIEVERS.values()],
                    "default": DEFAULT_BACKEND,
                    "llm_ready": is_llm_configured(),
                }
            )
        else:
            self._send_json({"error": "接口不存在"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/api/ask":
            self._send_json({"error": "接口不存在"}, status=404)
            return

        payload = self._read_json()
        if payload is None:
            return

        question = str(payload.get("question", "")).strip()
        if not question:
            self._send_json({"error": "question 不能为空"}, status=400)
            return

        if not RETRIEVERS:
            self._send_json(
                {
                    "error": "向量库为空，先把语料灌进去：\n"
                    "python scripts/ingest_chroma.py --limit 2000 --chunked"
                },
                status=503,
            )
            return

        backend = str(payload.get("backend") or DEFAULT_BACKEND)
        retriever = RETRIEVERS.get(backend)
        if retriever is None:
            self._send_json(
                {"error": f"检索后端不可用：{backend}（可用：{', '.join(RETRIEVERS)}）"},
                status=400,
            )
            return

        # top_k 夹在 [1, 库内条数] 之间，防止前端传个 999 把逻辑搞乱
        try:
            top_k = int(payload.get("top_k", 2))
        except (TypeError, ValueError):
            top_k = 2
        top_k = max(1, min(top_k, retriever.size))

        try:
            result = run_rag(question, retriever, top_k)
        except Exception as exc:  # 兜底：任何异常都变成 JSON，前端不会白屏
            self._send_json({"error": f"服务端异常：{type(exc).__name__}: {exc}"}, status=500)
            return

        self._send_json(result)

    # ---------------- 工具方法 ----------------
    def _read_json(self) -> dict | None:
        """读请求体并解析 JSON，失败时直接回 400 并返回 None。"""
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0

        if length <= 0:
            self._send_json({"error": "请求体为空"}, status=400)
            return None

        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json({"error": f"请求体不是合法 JSON：{exc}"}, status=400)
            return None

        if not isinstance(data, dict):
            self._send_json({"error": "请求体必须是 JSON 对象"}, status=400)
            return None

        return data

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._send_json({"error": f"文件不存在：{path.name}"}, status=404)
            return

        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        # 默认日志太吵，压成一行
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="RAG 阶段 1 Web 学习台")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    args = parser.parse_args()

    global DEFAULT_BACKEND

    # 只注册 Chroma 集合。TF-IDF 后端已随 txt 知识库一起移除。
    try:
        registered = []

        for item in list_collections(CHROMA_DIR):
            if item["count"] == 0:
                continue

            store = ChromaStore(CHROMA_DIR, item["name"])
            RETRIEVERS[item["name"]] = store
            registered.append(store)

            tag = "切分" if store.is_chunked else "整篇"
            print(f"[启动] Chroma 后端（{tag}）：{item['name']} {item['count']:,} 条")

        if registered:
            # 默认优先用切分集合（召回更好），没有才退到整篇
            chunked = [s for s in registered if s.is_chunked]
            DEFAULT_BACKEND = (chunked or registered)[0].collection.name
        else:
            print("[启动] 向量库里没有任何集合，服务将以空状态启动")
            print("       灌库：python scripts/ingest_chroma.py --limit 20000 --chunked")
    except Exception as exc:  # noqa: BLE001
        print(f"[启动] Chroma 不可用：{type(exc).__name__}: {exc}")

    print(f"[启动] 默认检索后端：{DEFAULT_BACKEND}")

    if is_llm_configured():
        print("[启动] 已检测到 LLM 配置，将调用真实大模型")
    else:
        print("[启动] 未配置 LLM_API_KEY → 走模拟模式（摘抄式回答，非真实生成）")

    server = ThreadingHTTPServer((args.host, args.port), RAGHandler)
    server.daemon_threads = True
    print(f"\n[就绪] 打开浏览器访问 http://{args.host}:{args.port}\n       按 Ctrl+C 停止\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[停止] 服务已关闭")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
