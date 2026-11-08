# ragxuexi

一个从零手写的 RAG（Retrieval-Augmented Generation，检索增强生成）知识库问答系统。

目标不是"跑通一个 demo"，而是**把向量检索的每一个环节摊开来看**：问题怎么变成向量、切片怎么切、召回怎么排序、Prompt 里到底塞了什么、模型为什么这样回答。

## 它现在是什么

- 语料：中文通用文本，`collection.tsv`，格式 `pid \t text`，约 230 万条 / 3.66 GB
- 已入库：**前 2000 篇 → 5,792 个切片**（可扩展到 2 万条，见下方"扩量"）
- 切分：384 token / 64 overlap，按 token 精确计量，句末对齐
- Embedding：`BAAI/bge-small-zh-v1.5`，512 维，ONNX CPU 推理（不需要 GPU、不需要 torch）
- 索引：Chroma 持久化库 + HNSW（cosine）
- 生成：任何兼容 OpenAI `/chat/completions` 的服务
- 前端：单页原生 JS，展示候选分数、Top-K 截断、完整 Prompt、各阶段耗时

## 快速开始

### 1. 环境

Python ≥ 3.10（开发环境 3.13）。

### 2. 装依赖

```bash
python -m pip install -r requirements.txt
```

国内网络建议走镜像（本项目实测阿里云源可用，清华源在某些代理环境下会解析失败）：

```bash
python -m pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt
```

### 3. 配置模型

```bash
cp .env.example .env
```

编辑 `.env`，至少填这三项：

```ini
LLM_API_KEY=sk-xxxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

不填 `LLM_API_KEY` 也能启动，会走**模拟模式**（从检索结果里摘抄，不调用模型），用来验证检索链路是否正常。界面上会明确标注，不会伪装成真实生成。

### 4. 灌库

仓库里**不含** `collection.tsv`（3.66 GB，被 `.gitignore` 排除），需要自备并放在项目根目录。

先小规模验证（约 4 分钟）：

```bash
python scripts/ingest_chroma.py --limit 2000 --chunked --collection corpus_chunk_384_64
```

确认无误后扩到 2 万（约 40 分钟，可中断，重跑会续传）：

```bash
python scripts/ingest_chroma.py --limit 20000 --chunked --collection corpus_chunk_384_64
```

首次运行会自动下载 embedding 模型（约 100 MB）到 `.model_cache/`，通过 `hf-mirror.com` 镜像拉取。

### 5. 启动

```bash
# 网页
python server.py              # 打开 http://127.0.0.1:8000

# 终端
python main.py "产后肚子大怎么恢复"
python main.py                # 交互模式，:q 退出，:debug 看分词与 Prompt，:topk N 改召回数
```

## 项目结构

```
rag/
├── chunker.py       按 token 切分，句末对齐，带 overlap
├── embedder.py      bge-small-zh-v1.5 封装（fastembed / ONNX）
├── chroma_store.py  Chroma 封装：余弦检索 + 按 pid 去重
├── pipeline.py      一次问答的完整流程，返回结构化中间态
├── prompt.py        system prompt + 资料拼接
├── llm.py           模型调用，含重试与降级
├── config.py        .env 解析与常量
├── schema.py        Document / SearchResult
└── tokenizer.py     中文分词（字符 bigram，仅供展示与调试）

scripts/ingest_chroma.py   灌库，支持整篇/切分、断点续传
server.py                  标准库 HTTP 服务
web/index.html             单页前端
```

分层的目的：`pipeline.py` 是业务层，`server.py` / `main.py` 只是适配器。换 FastAPI、加微信机器人，业务逻辑一行都不用改。

## 检索链路

```
离线：collection.tsv → 清洗 → chunker 切分 → embedder 向量化 → Chroma 落盘
在线：问题 → 向量 → HNSW 近邻（取 top_k×20，至少 40 条）→ 按 pid 去重 → 阈值过滤 → Top-K → Prompt → 模型
```

一次问答会返回这些中间态，前端逐项展示：

| 字段 | 含义 |
|---|---|
| `retrieval.query_tokens` | 问题被切成了哪些 token |
| `retrieval.candidates[].flag` | `selected`（入选）/ `over_k`（分数够但名额不够）/ `low_score`（没查到） |
| `contexts` | 真正拼进 Prompt 的原文 |
| `prompt.system` / `prompt.user` | 完整 Prompt，一字不删 |
| `timings` | 检索 / 生成 / 合计耗时 |

## 几个刻意的设计选择

| 决定 | 原因 |
|---|---|
| 切分用 **384 / 64**，不是 512 / 0 | 模型上限 512，留 128 余量防边界截断；64 token ≈ 两句，接住跨句语义；overlap 再大只会让相邻块高度相似，挤占 Top-K |
| 按 **token** 而非字符计量 | 模型上限是 token。中英混排时按字符切长度会失控 |
| 检索后**按 pid 去重** | 一篇文档切 5 段，5 段向量高度相似。不去重，Top-3 会全是同一篇的第 0/1/2 段 |
| 召回为空时**直接拒答，不调模型** | 没资料还去问模型，它一定用通用常识编一段，而用户会以为这是知识库的答案 |
| `ChromaStore(create=False)` 默认只读 | `get_or_create_collection` 在只读场景会静默建空集合，表现为"检索 0 条"而不是"集合不存在"，极难排查 |
| 只用标准库写 HTTP 服务 | 只有两个接口、单机自用。换成框架的收益不足以抵消学习成本 |
| system prompt 里写死"资料不足就承认" | 压制幻觉最便宜有效的手段。实测召回质量差时，模型会逐条说明资料讲的是什么 |

## 已知限制

- **只有 2000 篇文档**。问库里没有的内容会得到"根据现有资料无法确定"——这是设计行为，不是 bug
- **纯向量检索，没有关键词一路**。型号、编号、专有名词容易漏（`灵越7590` 这类罕见词表征弱）
- **没有重排**。Top-K 顺序完全由向量相似度决定，没有 Cross-Encoder 二次打分
- **阈值 `MIN_SCORE=0`** 形同虚设，只拦"字面完全不沾边"。真正可用的阈值要靠评测集标定
- **没有评测集**。任何"效果提升"的说法目前都无数据支撑
- 单机、单进程、无鉴权。别把它直接暴露到公网

## 踩过的坑

| 现象 | 原因 |
|---|---|
| `pip` 装不上，报 `from versions: none` | 代理环境下清华源解析异常，换阿里云源 |
| 模型下载卡住 | `huggingface.co` 不可达，需 `HF_ENDPOINT=https://hf-mirror.com`（代码已内置，必须在 import 前设置） |
| `An instance of Chroma already exists ... with different settings` | 同一持久化路径只允许一种 `Settings`，全项目统一走 `_client()` 工厂 |
| `delete_collection` 后磁盘没变小 | SQLite 页面不释放，HNSW 段目录也残留。需 `VACUUM`，彻底清理则删目录重建 |
| `NameError: TfidfRetriever` | 删了 import 忘了改类型注解。注解在 `def` 时求值，`ast.parse` 查不出来 |
| 启动日志一行都没有 | 后台跑服务时 `\| grep` 会块缓冲，改 `python -u server.py > server.log 2>&1` |

## 路线图

已完成的与其说是"阶段"，不如说是被真实问题推着走的顺序：

- [x] 手写 TF-IDF + 余弦相似度（已随 txt 知识库移除）
- [x] 文档切分与 overlap
- [x] Embedding + 语义检索
- [x] Chroma 向量库 + HNSW + 持久化
- [ ] BM25 + 向量双路召回，RRF 融合
- [ ] Cross-Encoder 重排
- [ ] 评测集：Recall@K / MRR / NDCG
- [ ] FastAPI + 异步任务 + 多租户
- [ ] Agent 化：工具调用、多步检索

## 许可

学习用途，未指定许可证。
