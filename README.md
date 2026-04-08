# Local RAG - 个人碎片知识落库系统

> 本地化优先、大模型驱动的个人知识资产管理与 RAG 问答系统。
> 把散落在公众号、小红书、知乎等平台的碎片信息，一键转化为结构化、持续生长的专属知识网络。

**v0.6.3** | 2026-04-08

---

## 核心能力

| 能力 | 说明 | 版本 |
|------|------|------|
| **微信转发即入库** | 企业微信应用，转发链接/图片/语音/文件自动入库 | v0.2+ |
| **跨平台抓取** | 公众号、小红书(MCP)、知乎、GitHub、通用网页自动解析 | v0.1+ |
| **多模态转文字** | PDF 解析 + 图片 OCR + 音频转录，统一转为 Markdown | v0.4+ |
| **图文穿插 OCR** | 图片内容原位替换，保持图文混排语义连贯 | **v0.6** |
| **长文分段清洗** | 万字长文分段 LLM 清洗+合并，零知识丢失 | **v0.6** |
| **LLM 智能清洗** | 自动降噪、提炼标题/摘要/标签/干货正文 | v0.1+ |
| **Wiki 编译引擎** | LLM 持续将文章编译为知识网络（主题页+实体页+交叉引用） | v0.3+ |
| **并行多路检索** | Wiki(宏观结构) + Data(微观细节) 双路召回 + RRF 融合 + 查询改写 | v0.3+ |
| **知识图谱** | D3 力导向图展示 Wiki 网络关系，融合页面列表+标签云+时间轴 | v0.3+ |
| **Web 仪表盘** | 知识图谱、分类浏览、入库（URL+文件上传）、搜索一站式界面 | v0.2+ |
| **AI 智能助手** | 右下角悬浮窗，多轮对话 + 意图识别 + 对话中入库 + 流式回复 | v0.5+ |
| **主动推送系统** | 每周摘要 / 知识回顾 / 关联推荐 / 编译通知，企微+Web 双渠道 | v0.5+ |
| **小红书 MCP** | 通过 MCP 协议完整抓取小红书笔记（文字+图片+标签），绕过反爬 | **v0.6** |
| **BM25 持久化** | pickle 序列化，重启毫秒级加载 | **v0.6** |

---

## 系统架构

```
输入源                路由分发            专用解析器           标准化池          编译引擎
┌──────────────┐    ┌────────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────────┐
│ 网页链接      │─┐  │            │──→ │ SCRAPER      │─┐  │            │    │              │
│ 小红书(MCP)  │─┤  │            │──→ │ VISION OCR   │─┤  │  Markdown  │    │  Wiki 编译    │
│ PDF 报告     │─┼→ │ DISPATCHER │──→ │ PDF PARSER   │─┼→ │  SSOT 落库  │──→ │  (Append-only │
│ 播客音频     │─┤  │            │──→ │ WHISPER      │─┤  │            │    │   非破坏性)    │
│ 图片截图     │─┘  │            │──→ │ (Kimi Vision)│─┘  │            │    │              │
└──────────────┘    └────────────┘    └──────────────┘    └────────────┘    └──────────────┘
                                            ↕                    ↕
                                      图文穿插 OCR        长文分段清洗
                                      (占位符原地替换)    (滑动窗口+合并)

智能助手层 (v0.5+)
┌──────────────────────────────────────────────────────────────────────────────────
│   用户输入 → 意图识别 (search/ingest/wiki/stats/chat)
│       ├── search → 查询改写 → Wiki+Data 并行召回 → LLM 综合答案
│       ├── ingest → URL 提取 → 自动入库流程
│       ├── wiki   → Wiki 页面查询/编译/健康检查
│       ├── stats  → 系统状态汇总
│       └── chat   → 多轮对话 (含历史上下文)
│                         ↓
│              SSE 流式回复 → Web 悬浮窗 / 企微推送
│
│   定时调度器 → 每周摘要 / 知识回顾 / 关联推荐 / 月度健康
└──────────────────────────────────────────────────────────────────────────────────
```

### 双层数据架构

```
data/     📦 原始来源层 (不可变, 入库后不再修改)
wiki/     📝 Wiki 层 (LLM 编译维护, Append-only 持续更新)
vectordb/ 🔍 向量索引 (可随时从 data/ + wiki/ 重建)
```

| 层 | 内容 | 谁写 | 可变性 |
|----|------|------|--------|
| `data/` | 原始文章（抓取+清洗后） | 入库流程 | 不可变 |
| `wiki/` | 主题页、实体页、洞察页 | LLM 编译引擎 | Append-only |
| `vectordb/` | ChromaDB 向量索引 | 索引器 | 可重建 |

---

## 使用方式

### 方式一：微信转发（推荐）

- 转发链接 → 自动抓取 → LLM 清洗 → 入库 → Wiki 编译 → 推送结果
- 发送图片 → 自动 OCR → 入库
- 发送语音 → 自动转录 → 入库
- 发送文件 → PDF 解析 / 自动分发 → 入库
- 发送文字 → AI 助手多轮对话（自动搜索知识库）
- 发送"订阅推送" → 开启每周摘要/知识回顾等定时推送

### 方式二：Web 界面

浏览器访问 `http://124.222.99.141:8900`

- 知识图谱：Wiki 网络可视化 + 标签云 + 时间轴 + 编译日志 + 健康检查
- 知识索引：分类浏览所有文章
- 入库：粘贴 URL 或拖拽上传 PDF/图片/音频
- 搜索：自然语言提问，Wiki+Data 双路 RAG 答案
- **AI 助手**：右下角悬浮窗，多轮对话 + 意图识别 + 对话中粘贴链接自动入库

### 方式三：命令行

```bash
python cli.py ingest <url>              # URL 入库
python cli.py ingest-file <文件路径>     # 文件入库 (PDF/图片/音频)
python cli.py search "你的问题"          # 搜索
python cli.py list                      # 列出所有知识
python cli.py stats                     # 系统状态
python cli.py reindex                   # 重建向量索引
python cli.py wiki-compile              # 全量 Wiki 编译
python cli.py wiki-inspect              # Wiki 健康检查
python cli.py wiki-list                 # 列出 Wiki 页面
```

### 方式四：API

```bash
# URL 入库
curl -X POST http://localhost:8900/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://mp.weixin.qq.com/s/xxx"}'

# 文件上传入库
curl -X POST http://localhost:8900/ingest/upload \
  -F "file=@report.pdf"

# 搜索
curl -X POST http://localhost:8900/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Vibe Coding 的核心流程是什么？"}'

# AI 助手对话 (SSE 流式)
curl -N -X POST http://localhost:8900/api/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "最近入了什么？"}'

# 手动触发推送
curl -X POST http://localhost:8900/api/assistant/push/weekly
```

---

## 项目结构

```
local-rag/
├── main.py                        # FastAPI 主入口 (v0.6.3)
├── config.py                      # 全局配置 (LLM Provider + 路径 + 企微 + MCP)
├── cli.py                         # 命令行入口
├── eval_rag.py                    # RAG 评测对比工具
├── requirements.txt               # Python 依赖
├── Dockerfile                     # Docker 容器化
├── docker-compose.yml             # 编排 (Local RAG + 小红书 MCP)
├── .env.example                   # 环境变量模板
│
├── assistant/                     # 🤖 智能助手 (v0.5)
│   ├── intent.py                 #   意图识别 (search/ingest/wiki/stats/chat)
│   ├── chat_engine.py            #   多轮对话引擎 (会话管理 + SSE 流式)
│   ├── router.py                 #   API 路由 (/api/assistant/*)
│   └── scheduler.py              #   定时任务调度 + 主动推送系统
│
├── ingestion/                     # 📥 数据摄入层 (多模态)
│   ├── base.py                    #   抓取器基类 + RawContent (含 images 字段)
│   ├── dispatcher.py              #   统一路由分发 (URL/PDF/图片/音频)
│   ├── router.py                  #   URL 路由 (自动读取 MCP 配置)
│   ├── wechat.py                  #   微信公众号 (占位符图文穿插)
│   ├── xiaohongshu.py             #   小红书 (MCP 完整抓取 + HTTP 降级)
│   ├── general.py                 #   通用网页 + 知乎 (图片提取 + 异常日志)
│   ├── pdf_parser.py              #   PDF 解析器 (pymupdf, 扫描件降级 OCR)
│   ├── vision_ocr.py              #   图片 OCR (Kimi Vision API)
│   └── audio_transcriber.py       #   音频转录 (Whisper API)
│
├── transform/                     # 🧹 AI 清洗层
│   └── llm_cleaner.py             #   LLM 脱水 (短文单次/长文分段清洗)
│
├── storage/                       # 💾 落库引擎
│   └── markdown_engine.py         #   Markdown + YAML 存储 (SSOT)
│
├── retrieval/                     # 🔍 检索层
│   ├── chunker.py                 #   语义切片器
│   ├── indexer.py                 #   ChromaDB 向量索引
│   ├── bm25.py                    #   BM25 关键词索引 (pickle 持久化)
│   ├── hybrid_searcher.py         #   并行多路召回 (Wiki+Data) + RAG
│   └── searcher.py                #   基础 RAG (CLI 用)
│
├── wiki/                          # 📝 Wiki 编译系统
│   ├── compiler.py                #   编译引擎 (LLM 分析→计划→执行)
│   ├── compile_queue.py           #   单线程编译队列 (防并发)
│   ├── page_store.py              #   Wiki 页面读写 (Append-only)
│   ├── index_builder.py           #   索引/日志自动生成 (零 LLM)
│   ├── inspector.py               #   健康检查 (孤立/缺失/过时)
│   └── _schema.md                 #   编译规则模式文件
│
├── wecom/                         # 💬 企业微信
│   ├── crypto.py                  #   消息加解密 (AES + 签名)
│   ├── sender.py                  #   主动推送 (文本 + 卡片)
│   └── callback.py                #   回调 (文本/链接/图片/语音/文件 + AI 对话)
│
├── utils/
│   └── url_utils.py               #   URL 归一化、平台识别、去重
│
├── web/
│   └── index.html                 #   SPA (Tailwind + Alpine.js + D3.js + AI 助手)
│
├── data/                          #   原始知识文件 (Markdown, git 忽略)
└── vectordb/                      #   ChromaDB 持久化 (git 忽略)
```

---

## 部署

### 方式一：Docker Compose（推荐）

```bash
# 1. 克隆项目
git clone <repo> && cd local-rag

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key 等

# 3. 启动（Local RAG + 小红书 MCP）
docker compose up -d

# 4. 首次小红书登录（扫码）
docker compose exec xhs-mcp /app/xiaohongshu-login
# 用小红书 App 扫码，之后 Cookie 持久化

# 5. 验证
curl http://localhost:8900/stats
```

### 方式二：直接运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 .env
cp .env.example .env

# 3. (可选) 启动小红书 MCP
# 下载二进制: https://github.com/xpzouying/xiaohongshu-mcp/releases
./xiaohongshu-login    # 首次扫码
./xiaohongshu-mcp &    # 后台运行
# .env 中设置: XHS_MCP_ENDPOINT=http://localhost:18060/mcp

# 4. 启动服务
uvicorn main:app --host 0.0.0.0 --port 8900
```

### 小红书 MCP 服务部署

小红书 MCP 服务提供完整的笔记数据抓取能力（文字+图片+标签+互动数据），绕过平台反爬。

| 项目 | 说明 |
|------|------|
| 服务端点 | `http://localhost:18060/mcp` |
| 协议 | MCP (JSON-RPC over HTTP + Session) |
| 登录方式 | 首次小红书 App 扫码，Cookie 持久化 |
| 降级策略 | MCP 不可用时自动降级为 HTTP 模式 |

**服务器部署步骤**：

```bash
# 1. 下载对应架构的预编译二进制
# darwin-arm64 / linux-amd64 / linux-arm64
wget https://github.com/xpzouying/xiaohongshu-mcp/releases/latest/download/xiaohongshu-mcp-linux-amd64.tar.gz
tar xzf xiaohongshu-mcp-linux-amd64.tar.gz

# 2. 首次登录（需要图形化环境或 VNC）
./xiaohongshu-login
# 扫码后 cookies.json 持久化

# 3. 后台启动 MCP 服务
nohup ./xiaohongshu-mcp > /var/log/xhs-mcp.log 2>&1 &

# 4. 设置环境变量
echo "XHS_MCP_ENDPOINT=http://localhost:18060/mcp" >> .env

# 5. 重启 Local RAG
docker compose restart local-rag
```

> **注意**：Cookie 有效期约 30 天，过期后需重新扫码登录。

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/ingest` | URL 入库 |
| POST | `/ingest/upload` | 文件上传入库 (PDF/图片/音频) |
| POST | `/search` | 并行多路检索 + RAG 答案生成 |
| POST | `/reindex` | 重建向量索引 |
| GET | `/stats` | 系统状态 (含 Wiki 页面数 + 编译队列) |
| GET | `/` | Web UI |
| POST | `/api/assistant/chat` | AI 助手流式对话 (SSE) |
| POST | `/api/assistant/chat-sync` | AI 助手非流式对话 |
| DELETE | `/api/assistant/sessions/{id}` | 清空对话会话 |
| GET | `/api/assistant/notifications` | 获取推送通知 |
| POST | `/api/assistant/push/{type}` | 手动触发推送 |
| GET | `/api/knowledge` | 知识列表 |
| GET | `/api/knowledge/{filename}` | 知识详情 |
| DELETE | `/api/knowledge/{filename}` | 删除知识 |
| GET | `/api/categories` | 分类聚合 |
| GET | `/api/tags` | 标签统计 |
| GET | `/api/timeline` | 时间轴 |
| GET | `/api/wiki/pages` | Wiki 页面列表 |
| GET | `/api/wiki/graph` | Wiki 知识图谱数据 |
| GET | `/api/wiki/log` | Wiki 编译日志 |
| GET | `/api/wiki/page/{subdir}/{filename}` | Wiki 页面详情 |
| POST | `/api/wiki/compile-all` | 全量 Wiki 编译 |
| POST | `/api/wiki/inspect` | Wiki 健康检查 |
| GET/POST | `/wecom/callback` | 企微回调 |

---

## 环境变量

```bash
# ===== 必填：LLM =====
LLM_PROVIDER=kimi
KIMI_API_KEY=sk-xxxxxxxx

# ===== 可选：备选 LLM =====
DEEPSEEK_API_KEY=sk-xxxxxxxx
OLLAMA_BASE_URL=http://localhost:11434/v1

# ===== Embedding =====
EMBEDDING_MODEL=all-MiniLM-L6-v2
HF_ENDPOINT=https://hf-mirror.com          # 国内加速

# ===== 服务 =====
HOST=0.0.0.0
PORT=8900

# ===== 小红书 MCP =====
XHS_MCP_ENDPOINT=http://localhost:18060/mcp  # 可选，不配则降级 HTTP

# ===== 企业微信 =====
WECOM_CORP_ID=wwxxxxxxxxx
WECOM_SECRET=xxxxxxxx
WECOM_AGENT_ID=1000002
WECOM_TOKEN=xxxxxxxx
WECOM_ENCODING_AES_KEY=xxxxxxxx
```

---

## 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| Web 框架 | FastAPI 0.115 | 异步高性能，自带 OpenAPI 文档 |
| 向量数据库 | ChromaDB 0.5 | 本地持久化，Cosine 距离 |
| Embedding | all-MiniLM-L6-v2 | SentenceTransformers，本地运行 |
| LLM 接口 | OpenAI SDK | 兼容 Kimi / DeepSeek / Ollama |
| PDF 解析 | PyMuPDF | 15MB，纯 CPU，文本提取+页面渲染 |
| 图片 OCR | Kimi Vision API | 零新依赖，~¥0.01/张 |
| 音频转录 | OpenAI Whisper API | 零新依赖，~¥2.5/小时 |
| 关键词检索 | BM25 (自实现) | 纯 Python，中英文 bigram，pickle 持久化 |
| 数据格式 | Markdown + YAML | SSOT，兼容 Obsidian/Logseq |
| 前端 | HTML + Tailwind + Alpine.js + D3 | 零构建 SPA |
| 企微接入 | 自建应用 + 回调 | AES 加解密，支持文本/链接/图片/语音/文件 |
| 流式输出 | SSE (Server-Sent Events) | 零 WebSocket 依赖 |
| 定时调度 | asyncio 后台任务 | 零额外框架，随应用启动 |
| 小红书抓取 | xiaohongshu-mcp | MCP 协议完整抓取，HTTP 降级 |
| 容器化 | Docker Compose | Local RAG + 小红书 MCP 编排 |

---

## 运行成本

| 项目 | 费用 |
|------|------|
| 腾讯云轻量服务器 (2C4G) | ¥80/年 (~¥6.7/月) |
| Kimi LLM (清洗+编译+检索+助手) | ~¥5/月 |
| Kimi Vision (图片 OCR) | ~¥0.2/月 |
| Whisper API (音频转录) | ~¥5/月 |
| **月度总计** | **~¥17/月** |
| **年度总计** | **~¥204/年** |

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-04-07 | 底层基座闭环 (CLI + API + 跨平台抓取 + 基础 RAG) |
| v0.2 | 2026-04-07 | Web 仪表盘 + 企微"转发即入库" + 混合检索 |
| v0.3 | 2026-04-08 | Wiki 编译模式 (LLM 持续编译知识网络 + 知识图谱融合) |
| v0.3.1 | 2026-04-08 | 稳定加固 (Prompt 调优 + 编译去重 + 队列状态 + 代码清理) |
| v0.4 | 2026-04-08 | 多模态转文字 (PDF + 图片 OCR + 音频转录 + 文件上传) |
| v0.5 | 2026-04-08 | 智能助手 (AI 悬浮窗 + 意图识别 + 多轮对话 + 主动推送 + 企微对话升级) |
| v0.6 | 2026-04-08 | 入库场景优化 (图片 OCR 串联 + Bug 修复 + 企微消息扩展) |
| v0.6.1 | 2026-04-08 | 图文穿插 OCR (占位符原地替换，解决语义脱节) |
| v0.6.2 | 2026-04-08 | 长文分段清洗 + BM25 持久化 + Wiki 截断优化 |
| **v0.6.3** | **2026-04-08** | **小红书 MCP 集成 (完整数据抓取 + Session 管理)** |
| v0.6.4 | 2026-04-08 | 正文保留原文 (LLM 只提取元数据，知识零丢失) |
| **v0.6.5** | **2026-04-08** | **前端重构 (图谱标签聚合 + Markdown 美化 + 编辑功能 + AI 助手升级)** ← 当前 |

---

## Roadmap

- [x] **v0.1** — 底层基座闭环
- [x] **v0.2** — Web 仪表盘 + 企微接入
- [x] **v0.3** — Wiki 编译模式 + 知识图谱融合
- [x] **v0.3.1** — 稳定加固
- [x] **v0.4** — 多模态转文字 (PDF/图片/音频)
- [x] **v0.5** — 智能助手
- [x] **v0.6** — 入库场景优化 + 前端重构 ← 当前版本
  - [x] 图片内容提取（小红书/微信/知乎图片 OCR 串联）
  - [x] 图文穿插 OCR（占位符原地替换，保持语义连贯）
  - [x] 长文分段清洗（滑动窗口清洗+合并，万字长文零丢失）
  - [x] 正文保留原文（LLM 只提取元数据，知识零丢失）
  - [x] BM25 持久化（pickle 序列化，重启毫秒级加载）
  - [x] Bug 修复（企微语音 file:// 链路、后缀硬编码、短文误判）
  - [x] 企微消息扩展（file/video/event 类型）
  - [x] 小红书 MCP 集成（完整数据抓取绕过反爬）
  - [x] 知识图谱重构（标签聚合视角 + 点击展开文章列表）
  - [x] Markdown 渲染美化 + 暗黑模式适配
  - [x] 文章编辑功能（在线编辑 + 自动重建索引）
  - [x] AI 助手升级（机器人图标 + 自动推荐问题）
  - [x] RAG 评测工具（eval_rag.py，AB 对比）
- [ ] **v0.7** — RAG 策略优化
  - 切片 overlap（150 字滑动窗口）
  - jieba 分词替换 bigram
  - Summary Chunk 扩充
  - 视频转录（yt-dlp + ffmpeg + Whisper）
- [ ] **v1.0** — 成熟版
  - 稳定运行 3 个月+
  - 多端访问 (PWA)
  - 定时备份到对象存储
  - 监控告警
  - 开源发布

---

## License

MIT
