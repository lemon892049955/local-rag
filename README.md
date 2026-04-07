# Local RAG - 个人碎片知识落库系统

> 本地化优先、大模型驱动的个人知识资产管理与 RAG 问答系统。
> 把散落在公众号、小红书、知乎等平台的碎片信息，一键转化为结构化、可即时调用的专属外脑。

**v0.2** | 2026-04-07

---

## 核心能力

| 能力 | 说明 |
|------|------|
| **微信转发即入库** | 企业微信应用接入，手机/PC 转发链接自动入库 |
| **跨平台抓取** | 公众号、小红书、通用网页自动解析 |
| **LLM 智能清洗** | 自动降噪、提炼标题/摘要/标签/干货正文 |
| **Markdown SSOT** | 所有知识以 .md 文件存储，最强可移植性 |
| **语义搜索 + RAG** | 本地向量化 + LLM 带引用答案生成 |
| **Web 仪表盘** | 知识浏览、入库、搜索一站式界面 |

## 使用方式

### 方式一：微信转发（推荐）

刷到好文章 → 转发给「知识库助手」→ 自动入库 → 推送结果卡片

直接发文字 → 自动搜索知识库 → 返回 RAG 答案

### 方式二：Web 界面

浏览器访问 `http://localhost:8900`，支持入库、搜索、浏览知识库。

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

## 新电脑部署指南

### 前置要求

- Python 3.9+
- Git

### 第一步：拉取代码

```bash
git clone <your-repo-url> local-rag
cd local-rag
```

### 第二步：创建虚拟环境 & 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> 首次运行会自动下载 Embedding 模型 `all-MiniLM-L6-v2`（约 80MB）。

### 第三步：配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入必要配置：

```bash
# ===== 必填：LLM =====
LLM_PROVIDER=kimi
KIMI_API_KEY=sk-xxxxxxxxxxxxxxxx     # 从 https://platform.moonshot.cn 获取

# ===== 可选：企业微信（微信转发入库功能） =====
WECOM_CORP_ID=wwxxxxxxxxx            # 企业 ID
WECOM_SECRET=xxxxxxxxxxxxxxxx        # 应用 Secret
WECOM_AGENT_ID=1000002               # 应用 AgentID
WECOM_TOKEN=xxxxxxx                  # 回调 Token
WECOM_ENCODING_AES_KEY=xxxxxxx       # 回调 EncodingAESKey (43位)
```

#### LLM Provider 选择

| Provider | 适合场景 | 获取方式 |
|----------|----------|----------|
| **Kimi** (默认) | 云端，效果好 | [platform.moonshot.cn](https://platform.moonshot.cn) |
| **DeepSeek** | 云端，性价比高 | [platform.deepseek.com](https://platform.deepseek.com) |
| **Ollama** | 纯本地，零费用 | [ollama.ai](https://ollama.ai) 安装后 `ollama pull qwen2.5:7b` |

### 第四步：启动服务

```bash
python cli.py serve
```

访问 `http://localhost:8900` 即可使用 Web 界面。

### 第五步（可选）：配置企微 - 微信转发入库

详见 [WECOM_SETUP.md](./WECOM_SETUP.md)，主要步骤：

1. 注册企业微信 → 创建自建应用「知识库助手」
2. 配置应用回调 URL → 指向你的服务地址
3. 开启微信插件 → 个人微信扫码关注
4. 配置 IP 白名单

> **公网访问**：本地开发可用 [ngrok](https://ngrok.com) 或 [localtunnel](https://github.com/localtunnel/localtunnel) 做内网穿透。
> 生产环境建议部署到云服务器（腾讯云 Lighthouse ~30元/月）。

### 第六步（可选）：恢复知识库数据

如果从旧电脑迁移，只需要拷贝 `data/` 目录（Markdown 文件），然后重建索引：

```bash
# 拷贝 data/ 目录到新电脑的项目下
cp -r /path/to/old/data ./data

# 重建向量索引
python cli.py reindex
```

Markdown 文件就是 SSOT，ChromaDB 索引随时可以从 `.md` 文件重建。

---

## 项目结构

```
local-rag/
├── config.py              # 全局配置
├── main.py                # FastAPI 入口（API + Web）
├── cli.py                 # 命令行入口
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量模板
├── WECOM_SETUP.md         # 企微接入指南
│
├── data/                  # 📂 知识文件 (Markdown SSOT)
├── vectordb/              # ChromaDB 持久化 (可从 data/ 重建)
├── web/                   # Web 前端
│   └── index.html         # SPA 单页应用
│
├── ingestion/             # 数据摄入层
│   ├── base.py            #   抓取器基类
│   ├── router.py          #   URL 路由 (自动识别平台)
│   ├── wechat.py          #   微信公众号 (BeautifulSoup)
│   ├── xiaohongshu.py     #   小红书 (SSR JSON 解析)
│   └── general.py         #   通用网页 + 知乎 (Readability)
│
├── transform/             # AI 清洗层
│   └── llm_cleaner.py     #   LLM 脱水 (可插拔 Provider)
│
├── storage/               # 落库引擎
│   └── markdown_engine.py #   Markdown + YAML Front-matter
│
├── retrieval/             # 向量检索层
│   ├── chunker.py         #   语义切片 (标题层级 + 兜底)
│   ├── indexer.py         #   ChromaDB 索引
│   └── searcher.py        #   RAG 答案生成 (带 Citation)
│
├── wecom/                 # 企业微信接入
│   ├── crypto.py          #   消息加解密 (AES + 签名)
│   ├── sender.py          #   主动推送 (入库通知卡片)
│   └── callback.py        #   回调路由 (消息接收 + 去重)
│
└── utils/
    └── url_utils.py       # URL 归一化、去重检测
```

## Markdown 知识文件格式

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
---

# Vibe Coding 工作流程详解

> **摘要**: 通过重组产品构建流程...

## 一、定义问题
...
```

## 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | 异步高性能，自带 OpenAPI 文档 |
| 向量库 | ChromaDB | 本地持久化，零运维 |
| Embedding | all-MiniLM-L6-v2 | 本地运行，无 API 费用 |
| LLM | Kimi / DeepSeek / Ollama | 可插拔切换 |
| 数据格式 | Markdown + YAML | SSOT，兼容 Obsidian/Logseq |
| 网页抓取 | requests + BS4 + Readability | 轻量高效 |
| 前端 | HTML + Tailwind + Alpine.js | 零构建，单文件 SPA |
| 企微接入 | 自建应用 + 回调 | 官方 API，零封号风险 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/ingest` | URL 入库 |
| POST | `/ingest/text` | 手动粘贴入库 |
| POST | `/search` | 语义搜索 + RAG |
| GET | `/api/knowledge` | 知识列表 |
| GET | `/api/knowledge/{filename}` | 知识详情 |
| DELETE | `/api/knowledge/{filename}` | 删除知识 |
| POST | `/reindex` | 重建向量索引 |
| GET | `/stats` | 系统状态 |
| GET/POST | `/wecom/callback` | 企微回调 |

## Roadmap

- [x] **v0.1** - 底层基座闭环（CLI + API + 跨平台抓取 + RAG）
- [x] **v0.2** - Web 仪表盘 + 企业微信"转发即入库" ← 当前版本
- [ ] **v0.3** - 云服务器部署 + 稳定长期运行
- [ ] **v0.4** - 知识联结（双向链接 + 标签云 + 时间轴）
- [ ] **v0.5** - 多模态（图片 OCR、PDF 解析）

## License

MIT
