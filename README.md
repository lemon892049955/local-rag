# Local RAG - 个人碎片知识落库系统

> 本地化优先、大模型驱动的个人知识资产管理与 RAG 问答系统。
> 把散落在公众号、小红书、知乎等平台的碎片信息，一键转化为结构化、可即时调用的专属外脑。

**v0.2.0** | 2026-04-07

---

## 一、项目定位

在信息爆炸的时代，我们每天在微信公众号、小红书、知乎等平台刷到大量有价值的内容，但这些碎片知识散落各处、难以检索。Local RAG 就是为了解决这个问题而生的个人知识管理系统。

**核心理念**：**转发即入库，提问即检索**。通过企业微信接入，在微信中转发一个链接就能自动完成抓取、AI 清洗、结构化存储、向量索引的全流程，之后随时可以用自然语言提问检索。

---

## 二、核心能力

| 能力 | 说明 | 状态 |
|------|------|------|
| **微信转发即入库** | 企业微信应用接入，手机/PC 转发链接自动入库 | ✅ 已实现 |
| **跨平台抓取** | 公众号、小红书、知乎、通用网页自动识别和解析 | ✅ 已实现 |
| **LLM 智能清洗** | 自动降噪、提炼标题/摘要/标签/干货正文 | ✅ 已实现 |
| **Markdown SSOT** | 所有知识以 `.md` 文件存储，兼容 Obsidian/Logseq | ✅ 已实现 |
| **混合检索 + RAG** | 向量语义搜索 + BM25 关键词搜索 + RRF 融合 + LLM 答案生成 | ✅ 已实现 |
| **Web 仪表盘** | 知识图谱、分类浏览、入库、搜索一站式界面 | ✅ 已实现 |
| **Docker 部署** | 支持一键容器化部署到云服务器 | ✅ 已实现 |

---

## 三、系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         使用入口                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ 企业微信  │  │ Web UI   │  │ CLI 命令行│  │ REST API (/docs) │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘    │
└───────┼──────────────┼──────────────┼─────────────────┼─────────────┘
        │              │              │                 │
        ▼              ▼              ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI 服务层 (main.py)                          │
│  POST /ingest  ·  POST /search  ·  GET /api/*  ·  GET/POST /wecom  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
        ┌───────────┬───────┼───────┬───────────┐
        ▼           ▼       ▼       ▼           ▼
  ┌──────────┐ ┌─────────┐ ┌─────┐ ┌─────────┐ ┌──────────┐
  │ 数据摄入  │ │ AI 清洗 │ │ 落库 │ │ 向量检索 │ │ 企微接入  │
  │ingestion/│ │transform│ │store│ │retrieval│ │  wecom/  │
  └──────────┘ └─────────┘ └─────┘ └─────────┘ └──────────┘
```

### 数据流转全流程

```
URL → 平台识别 → 专用抓取器 → LLM 脱水清洗 → Markdown 落库 → 向量索引 + BM25 索引
                                                                        ↓
用户提问 → 查询改写 → 向量召回 + BM25 召回 → RRF 融合排序 → LLM 答案生成 → 返回结果
```

---

## 四、项目结构

```
local-rag/
├── main.py                    # FastAPI 主入口，所有 HTTP API 端点
├── config.py                  # 全局配置（环境变量 + LLM Provider 管理）
├── cli.py                     # 命令行入口
├── requirements.txt           # Python 依赖（25 个包）
├── Dockerfile                 # Docker 容器化配置
├── .env.example               # 环境变量模板
├── .dockerignore              # Docker 构建忽略
├── .gitignore                 # Git 忽略
├── WECOM_SETUP.md             # 企微接入指南
│
├── web/                       # 📱 前端
│   └── index.html             #   SPA 单页应用 (Tailwind + Alpine.js + D3.js)
│
├── ingestion/                 # 📥 数据摄入层
│   ├── base.py                #   抓取器基类 (BaseFetcher) + 数据模型 (RawContent)
│   ├── router.py              #   URL 路由分发器 (自动识别平台)
│   ├── wechat.py              #   微信公众号抓取器 (BeautifulSoup)
│   ├── xiaohongshu.py         #   小红书抓取器 (SSR JSON + MCP 双模式)
│   └── general.py             #   通用网页 + 知乎抓取器 (Readability + API)
│
├── transform/                 # 🧹 AI 清洗层
│   └── llm_cleaner.py         #   LLM 脱水器 (可插拔 Provider)
│
├── storage/                   # 💾 落库引擎
│   └── markdown_engine.py     #   Markdown + YAML Front-matter 存储
│
├── retrieval/                 # 🔍 向量检索层
│   ├── chunker.py             #   语义切片器 (标题层级 + 兜底固定长度)
│   ├── indexer.py             #   ChromaDB 向量索引
│   ├── searcher.py            #   基础 RAG 搜索器 (CLI 使用)
│   ├── bm25.py                #   BM25 关键词索引 (纯 Python, 中英混合分词)
│   └── hybrid_searcher.py     #   混合检索 (查询改写 + 双路召回 + RRF 融合)
│
├── wecom/                     # 💬 企业微信接入
│   ├── crypto.py              #   消息加解密 (AES-CBC + SHA1 签名)
│   ├── sender.py              #   主动推送 (文本消息 + 卡片消息)
│   └── callback.py            #   回调路由 (GET 验证 + POST 消息接收 + 异步处理)
│
├── utils/                     # 🔧 工具
│   └── url_utils.py           #   URL 归一化、平台识别、去重检测
│
├── data/                      # 📂 知识文件 (Markdown SSOT, git 忽略)
└── vectordb/                  # 📂 ChromaDB 持久化 (可从 data/ 重建, git 忽略)
```

---

## 五、核心模块详解

### 5.1 数据摄入层 (`ingestion/`)

**职责**：根据 URL 自动识别平台，调用对应抓取器获取原始内容。

| 抓取器 | 适用平台 | 抓取策略 |
|--------|---------|---------|
| `WechatFetcher` | 微信公众号 (`mp.weixin.qq.com`) | requests + BeautifulSoup，提取 `#js_content` 正文区域，清理二维码/广告干扰 |
| `XiaohongshuFetcher` | 小红书 (`xiaohongshu.com`, `xhslink.com`) | 优先 MCP 协议抓取，降级为 SSR `window.__INITIAL_STATE__` JSON 解析，兜底 OG meta |
| `GeneralFetcher` | 知乎 + 通用网页 | 知乎：API 优先 (`/api/v4/answers/`, `/api/v4/articles/`)，SSR `initialData` 解析兜底；通用：Readability 算法提取主要内容 |

**数据模型**：
```python
@dataclass
class RawContent:
    url: str              # 原始 URL
    title: str            # 页面标题
    content: str          # 纯文本正文
    author: str           # 作者
    source_platform: str  # "wechat" | "xiaohongshu" | "general"
    original_tags: list   # 平台原始标签（小红书）
```

**路由规则** (`url_utils.detect_source`)：
- `mp.weixin.qq.com` → `wechat`
- `xiaohongshu.com` / `xhslink.com` / `xhs.cn` → `xiaohongshu`
- 其他 → `general`

---

### 5.2 AI 清洗层 (`transform/`)

**职责**：通过 LLM 将原始抓取内容"脱水"为标准化知识资产。

**输入**：原始标题 + 正文 + 来源信息
**输出**：
```python
@dataclass
class CleanedKnowledge:
    title: str            # 精炼标题
    summary: str          # 20-50 字核心总结
    tags: list[str]       # 3-5 个业务标签
    cleaned_content: str  # 纯干货正文 (Markdown 格式)
```

**清洗规则**：
1. **降噪**：剥离营销话术、过多 Emoji、引导语
2. **提标签**：提取高维业务标签（如"产品方法论"、"技术架构"）
3. **结构化**：使用 Markdown 标题层级组织正文
4. **容错**：LLM 调用失败时返回基础版本（原文不清洗）

**LLM Provider 支持**（通过 `LLM_PROVIDER` 环境变量切换）：

| Provider | API 地址 | 默认模型 | 适合场景 |
|----------|---------|---------|---------|
| **Kimi** (默认) | `api.moonshot.cn/v1` | `moonshot-v1-8k` | 云端，中文效果好 |
| **DeepSeek** | `api.deepseek.com/v1` | `deepseek-chat` | 云端，性价比高 |
| **Ollama** | `localhost:11434/v1` | `qwen2.5:7b` | 纯本地，零费用 |

---

### 5.3 落库引擎 (`storage/`)

**职责**：将结构化知识持久化为 Markdown 文件，作为系统唯一数据源 (SSOT)。

**文件命名**：`[YYMMDD]_[8位短UUID]_[精简标题].md`
- 示例：`260407_SiSeA42t_Vibe_Coding工作流程详解.md`

**文件格式**：
```markdown
---
title: Vibe Coding 工作流程详解
summary: 通过重组产品构建流程，实现从想法到产品落地的高效转化
tags:
  - Vibe Coding
  - 产品构建
  - 工作流程
source_url: https://mp.weixin.qq.com/s/xxxxx
source_platform: wechat
author: 麥柯Michael
created_at: '2026-04-07 12:01:34'
updated_at: '2026-04-07 12:01:34'
---

# Vibe Coding 工作流程详解

> **摘要**: 通过重组产品构建流程...

## 一、定义问题
...
```

**设计哲学**：Markdown 文件即数据源，ChromaDB 索引可随时从 `.md` 文件重建。迁移时只需拷贝 `data/` 目录。

---

### 5.4 向量检索层 (`retrieval/`)

#### 语义切片器 (`chunker.py`)

**切片策略**：
1. **摘要切片 (Summary Chunk)**：每篇文档必生成，包含标题 + 摘要 + 标签，用于泛化检索
2. **章节切片 (Section Chunk)**：按 `##` / `###` 标题层级拆分，每片 ≤ 1500 字符
3. **段落切片 (Segment Chunk)**：无标题结构的文档按固定长度拆分

#### ChromaDB 向量索引 (`indexer.py`)

- **Embedding 模型**：`all-MiniLM-L6-v2`（本地运行，约 80MB，无 API 费用）
- **距离度量**：Cosine
- **持久化**：本地 SQLite 存储 (`vectordb/` 目录)
- 支持单文件增量索引和全量重建

#### BM25 关键词索引 (`bm25.py`)

- 纯 Python 实现，零外部依赖
- **中英文混合分词**：英文按空格分词 + 中文 unigram + bigram 滑窗
- 从 Markdown 文件目录构建，与向量索引互补

#### 混合检索器 (`hybrid_searcher.py`)

**完整检索流程**：
1. **查询改写**：LLM 将模糊问题分解为 2-3 个精确检索 query（一个偏语义，一个偏关键词）
2. **双路召回**：每个 query 同时走向量搜索和 BM25 搜索
3. **RRF 融合排序**：Reciprocal Rank Fusion (k=60) 合并两路结果
   - 摘要切片获得 1.3x 加权
   - 双路命中获得 1.2x 加权
4. **LLM 答案生成**：将 Top-K 切片组装为 Context，调用 LLM 生成带引用来源的答案

---

### 5.5 企业微信接入 (`wecom/`)

#### 消息加解密 (`crypto.py`)

- 实现企微官方 AES-CBC 加解密方案
- SHA1 签名验证，防篡改
- Corp ID 校验，防跨企业攻击

#### 回调路由 (`callback.py`)

| 端点 | 方法 | 功能 |
|------|------|------|
| `/wecom/callback` | GET | URL 验证（企微后台配置时触发） |
| `/wecom/callback` | POST | 消息接收 + 异步处理 |

**消息处理逻辑**：
- **文本消息含 URL** → 提取链接 → 异步入库 → 立即回复"已收到"
- **文本消息无 URL** → 当作搜索查询 → 异步混合检索 → 返回 RAG 答案
- **链接消息** (微信转发) → 直接异步入库
- 内置 MsgId 去重缓存（500 条，5 分钟 TTL），防企微重试导致重复处理

#### 消息推送 (`sender.py`)

- `send_text_msg()`：文本消息
- `send_text_card()`：卡片消息（入库成功/失败通知）
- access_token 自动缓存 + 过期刷新

---

### 5.6 Web 前端 (`web/index.html`)

**技术栈**：零构建 SPA，所有依赖通过 CDN 加载
- **Tailwind CSS**：样式
- **Alpine.js**：响应式数据绑定
- **D3.js**：知识图谱力导向图
- **Marked.js**：Markdown 渲染

**页面功能**：

| 页面 | 功能 |
|------|------|
| 知识图谱 | D3 力导向图可视化，节点 = 文章 + 标签，边 = 关联关系，右侧标签云 + 时间轴 |
| 知识索引 | 分类浏览所有知识，支持按分类/标签筛选 |
| 入库 | 粘贴 URL 一键入库 |
| 搜索 | 自然语言提问，返回 AI 答案 + 来源引用 |
| 详情页 | 查看单篇知识的元数据 + Markdown 正文 |

**特性**：暗黑模式、侧边栏折叠、快速搜索、平台图标区分、相对时间显示。

---

## 六、API 端点一览

| 方法 | 路径 | 说明 | 请求体 |
|------|------|------|--------|
| POST | `/ingest` | URL 入库（抓取→清洗→落库→索引） | `{"url": "https://..."}` |
| POST | `/search` | 混合检索 + RAG 答案生成 | `{"query": "...", "top_k": 3}` |
| POST | `/reindex` | 重建全部向量索引 | - |
| GET | `/stats` | 系统状态（文件数 + 切片数） | - |
| GET | `/` | Web UI 首页 | - |
| GET | `/api/knowledge` | 知识列表（所有文件的元数据） | - |
| GET | `/api/knowledge/{filename}` | 知识详情（元数据 + 正文） | - |
| DELETE | `/api/knowledge/{filename}` | 删除知识文件 | - |
| GET | `/api/categories` | 分类聚合（6 大类 + 其他） | - |
| GET | `/api/graph` | 知识图谱数据（节点 + 边） | - |
| GET | `/api/tags` | 标签统计 | - |
| GET | `/api/timeline` | 时间轴数据 | - |
| GET | `/wecom/callback` | 企微回调 URL 验证 | - |
| POST | `/wecom/callback` | 企微消息接收 | XML body |

**自动 API 文档**：启动服务后访问 `http://localhost:8900/docs` (Swagger UI)

---

## 七、技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| Web 框架 | FastAPI 0.115 | 异步高性能，自带 OpenAPI 文档 |
| 向量数据库 | ChromaDB 0.5 | 本地持久化，零运维，Cosine 距离 |
| Embedding | all-MiniLM-L6-v2 | SentenceTransformers，本地运行 |
| LLM 接口 | OpenAI SDK | 兼容 Kimi / DeepSeek / Ollama |
| 数据格式 | Markdown + YAML | SSOT，兼容 Obsidian/Logseq |
| 网页抓取 | requests + BS4 + Readability | 轻量高效，覆盖公众号/小红书/知乎 |
| 关键词检索 | BM25 (自实现) | 纯 Python，中英文 bigram 分词 |
| 前端 | HTML + Tailwind + Alpine.js + D3 | 零构建，单文件 SPA |
| 企微接入 | 自建应用 + 回调 | 官方 API，AES 加解密 |
| 加密库 | PyCryptodome | 企微消息 AES-CBC 加解密 |
| 容器化 | Docker (Python 3.11-slim) | 一键部署 |

---

## 八、使用方式

### 方式一：微信转发（推荐）

1. 刷到好文章 → 转发给「知识库助手」→ 自动入库 → 推送结果卡片
2. 直接发文字 → 自动搜索知识库 → 返回 RAG 答案

### 方式二：Web 界面

浏览器访问 `http://your-server:8900`，支持入库、搜索、浏览知识库、查看知识图谱。

### 方式三：命令行

```bash
python cli.py ingest <url>         # 入库
python cli.py search "你的问题"     # 搜索
python cli.py list                 # 列出所有知识
python cli.py stats                # 系统状态
python cli.py reindex              # 重建向量索引
python cli.py serve                # 启动 Web 服务
```

### 方式四：API

```bash
# 入库
curl -X POST http://localhost:8900/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://mp.weixin.qq.com/s/xxx"}'

# 搜索
curl -X POST http://localhost:8900/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Vibe Coding 的核心流程是什么？"}'
```

---

## 九、部署指南

### 方式一：本地开发

```bash
# 1. 克隆项目
git clone <your-repo-url> local-rag && cd local-rag

# 2. 创建虚拟环境
python3 -m venv .venv && source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LLM API Key 等配置

# 5. 启动服务
python cli.py serve
# 访问 http://localhost:8900
```

### 方式二：Docker 部署到云服务器

```bash
# 1. 构建镜像
docker build -t local-rag .

# 2. 运行容器
docker run -d --name local-rag --restart unless-stopped \
  -p 8900:8900 \
  -v /root/local-rag-data:/app/data \
  -v /root/local-rag-vectordb:/app/vectordb \
  -e LLM_PROVIDER=kimi \
  -e KIMI_API_KEY=sk-your-key \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e WECOM_CORP_ID=your-corp-id \
  -e WECOM_TOKEN=your-token \
  -e WECOM_ENCODING_AES_KEY=your-aes-key \
  local-rag
```

### 当前生产部署信息

| 项目 | 值 |
|------|-----|
| 云服务器 | 腾讯云轻量应用服务器 (上海) |
| 实例 ID | `lhins-3mnho28y` |
| 公网 IP | `124.222.99.141` |
| 服务端口 | `8900` |
| 访问地址 | `http://124.222.99.141:8900` |
| LLM Provider | Kimi (moonshot-v1-8k) |
| 企微回调 URL | `http://124.222.99.141:8900/wecom/callback` |

---

## 十、环境变量说明

```bash
# ===== 必填：LLM =====
LLM_PROVIDER=kimi                          # kimi / deepseek / ollama
KIMI_API_KEY=sk-xxxxxxxx                   # Kimi API Key

# ===== 可选：备选 LLM =====
DEEPSEEK_API_KEY=sk-xxxxxxxx               # DeepSeek API Key
OLLAMA_BASE_URL=http://localhost:11434/v1   # Ollama 地址
OLLAMA_MODEL=qwen2.5:7b                    # Ollama 模型

# ===== Embedding =====
EMBEDDING_MODEL=all-MiniLM-L6-v2           # 本地 Embedding 模型
HF_ENDPOINT=https://hf-mirror.com          # HuggingFace 镜像 (国内加速)

# ===== 服务 =====
HOST=0.0.0.0
PORT=8900

# ===== 企业微信 =====
WECOM_CORP_ID=wwxxxxxxxxx                  # 企业 ID
WECOM_SECRET=xxxxxxxx                      # 应用 Secret (发消息用)
WECOM_AGENT_ID=1000002                     # 应用 AgentID
WECOM_TOKEN=xxxxxxxx                       # 回调 Token
WECOM_ENCODING_AES_KEY=xxxxxxxx            # 回调 EncodingAESKey (43位)
```

---

## 十一、知识库现状

截至 2026-04-07，已入库 **7 篇知识**：

| 标题 | 来源 | 标签 |
|------|------|------|
| 科技爱好者周刊（第 330 期） | 公众号 | AI, 技术, 科技动态 |
| Vibe Coding 工作流程详解 | 公众号 | Vibe Coding, 产品构建, 工作流程 |
| AI 产品经理的职场焦虑与挑战 | 小红书 | AI, 产品经理, 职场 |
| Karpathy 的 AI 个人知识库搭建教程 | 公众号 | AI, 知识库, 教程 |
| AI 时代文科生的就业前景分析 | 通用网页 | AI, 就业, 文科 |
| AI 仿真人短剧角色设计的核心技巧 | 公众号 | AI, 短剧, 角色设计 |
| 腾讯产品经理晋升的共同点：影响力半径 | 公众号 | 产品经理, 晋升, 影响力 |

---

## 十二、分类体系

系统基于标签关键词自动将知识归类为 6 大类：

| 分类 | 关键词匹配规则 |
|------|---------------|
| AI & 技术 | AI, 人工智能, 技术, 深度学习, 机器学习, 模型, 算法, 编程, 开发, 架构, 工程 |
| 产品 & 方法论 | 产品, 方法论, 设计, 工作流, SOP, 流程, 策略, 框架, Coding |
| 职场 & 行业 | 职场, 行业, 就业, 求职, 薪资, 焦虑, 转型, 职业 |
| 运营 & 增长 | 运营, 增长, 营销, 推广, 电商, 转化, 投放 |
| 内容 & 创作 | 内容, 创作, 写作, 文案, 短剧, 视频, 自媒体, 角色 |
| 工具 & 资源 | 工具, 资源, 分享, 推荐, 教程, 指南, 科技动态 |

---

## 十三、Roadmap

- [x] **v0.1** — 底层基座闭环（CLI + API + 跨平台抓取 + 基础 RAG）
- [x] **v0.2** — Web 仪表盘 + 企业微信"转发即入库" + 混合检索 ← **当前版本**
- [ ] **v0.3** — 云服务器稳定运行 + 企微消息收发完善
- [ ] **v0.4** — 知识联结（双向链接 + 标签云 + 时间轴深度整合）
- [ ] **v0.5** — 多模态（图片 OCR、PDF 解析、音频转文字）

---

## 十四、依赖列表

```
# Web Framework
fastapi==0.115.6          # Web 框架
uvicorn==0.34.0           # ASGI 服务器

# HTTP & Scraping
requests==2.32.3          # HTTP 客户端
beautifulsoup4==4.12.3    # HTML 解析
readability-lxml==0.8.1   # 正文提取算法
lxml==5.3.0               # XML/HTML 解析器
lxml_html_clean           # lxml HTML 清洗补丁

# LLM
openai==1.58.1            # OpenAI 兼容 SDK (Kimi/DeepSeek/Ollama)

# Vector DB & Embedding
chromadb==0.5.23          # 向量数据库
sentence-transformers==3.3.1  # 本地 Embedding

# Storage
pyyaml==6.0.2             # YAML 解析

# Utilities
python-dotenv==1.0.1      # 环境变量加载
shortuuid==1.0.13         # 短 UUID 生成
pycryptodome==3.21.0      # 企微消息 AES 加解密
```

---

## License

MIT
