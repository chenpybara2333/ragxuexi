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

## 评测

没有基线数字，任何"我觉得更准了"都是感觉。`eval/` 下是一套最小评测：

```bash
python eval/generate_questions.py --n 60   # 生成问题集（调模型，约 5 分钟）
python eval/run_eval.py                    # 跑检索、算指标、写 report.md
```

**问题集怎么来的**：从已入库的 chunk 反向生成。抽一条真实 chunk，让模型基于它写一个"用户会这么问"的问题，并要求**不许照抄原文措辞**。这样 gold 标签天然准确——答案一定在这条 chunk 里，pid 已知。同时标注类型：`keyword`（含专名/型号/数字）与 `semantic`（纯口语改写），两类难度不同，混在一起统计会掩盖问题。

**为什么决定加 BM25**：纯向量基线跑完后，7 条失败里有 5 条含精确字符串（`95017`、`百度文库`、`找教育平台`、`微信`、`林旭`），其余 2 条是问题本身过于空泛。两类原因解法不同——前者补字面召回，后者要 Query Rewrite。先做前者，于是有了下面这张对比。

**基线对比**（57 条问题 / 5,792 个切片 / bge-small-zh-v1.5）：

| 模式 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| 纯向量 | 77% | 88% | 88% | 0.822 |
| **混合（BM25 + 向量 + RRF）** | **86%** | **93%** | **95%** | **0.892** |

分类型对比：

| 类型 | 条数 | 纯向量 R@3 | 混合 R@3 |
|---|---:|---:|---:|
| keyword（含专名/数字） | 43 | 88% | **95%** |
| semantic（纯口语改写） | 14 | 86% | 86% |

**semantic 一类完全没有提升**，这和预期一致：BM25 靠字面重合吃饭，对"问题本身空泛"无能为力。它的收益全部来自 keyword 那一路（+7pp），以及失败数从 7 条降到 3 条。

具体捞回了哪些（`95017` / `百度文库` / `微信` 三条），看 `eval/report.hybrid.md`。剩下 2 条 semantic 失败需要 Query Rewrite 或多查询，不是换检索算法能解决的。

> 顺带记一次判断失误：我原本怀疑"期望 pid 1358、实际返回 1351/1352/1357"是标注粒度问题（同一篇文章被切成连续 pid）。实测 5-gram 重合率只有 0.028 / 0 / 0——它们只是都提到"林旭"这个人物，内容并不相同。**先假设、再测量，测量否掉了假设，就照实写。**

## 混合检索怎么工作

```
问题 ──┬── BM25（字符 bigram，倒排索引）──> 有序 chunk 列表 ─┐
       └── bge 向量（HNSW，cosine）   ──> 有序 chunk 列表 ─┤
                                                          ├─ RRF 融合 ─> 按 pid 去重 ─> Top-K
```

三个实现细节值得说：

1. **融合在 chunk 级做，去重放最后**。两路唯一共享的标识是 `chunk_id`（`pid::序号`）。如果先按 pid 去重再融合，一篇文档的多个片段会被压成一条，等于提前扔掉"哪一段更相关"的信息。
2. **RRF 只看名次**。公式 `Σ 1/(k + rank)`，`k=60`。余弦固定在 `[0,1]`，BM25 分数无上界，直接相加会被 BM25 主导；归一化又会被单条异常高分扭曲。用名次就绕开了尺度问题。
3. **BM25 用字符 bigram 分词**，`"95017"` 会作为整体 token（ASCII 段整段保留），所以号码、型号能精确命中——这正是稠密向量缺失的能力。`b=0.5` 而非默认 0.75，因为切片后各块长度本来就接近，过度长度归一化没有意义。

代价：启动时要把全部 chunk 重新分词建倒排索引，5,792 条约 2 秒。数据量上到十万级时这会成为启动瓶颈，届时需要把索引持久化。

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
- **评测集规模小（57 条）**，且问题由模型生成，可能与真实用户提问分布有偏差。它能测出"改动是不是让指标变差"，但不足以代表线上效果
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
- [x] 评测集：Recall@K / MRR（NDCG 未做）
- [x] BM25 + 向量双路召回，RRF 融合（R@3 88% → 93%）
- [ ] Cross-Encoder 重排
- [ ] Query Rewrite / 多查询（解决空泛提问）
- [ ] FastAPI + 异步任务 + 多租户
- [ ] Agent 化：工具调用、多步检索

## 许可

学习用途，未指定许可证。
