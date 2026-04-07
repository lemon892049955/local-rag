# 🧠 Local RAG - 个人碎片知识落库系统

> 本地化优先、大模型驱动的个人知识资产管理与 RAG 问答系统。
> 把散落在公众号、小红书、知乎等平台的碎片信息，一键转化为结构化、可即时调用的专属外脑。

## 核心特性

- **一键入库**：发送链接，自动完成抓取 → LLM 清洗 → Markdown 落库 → 向量索引
- **Markdown SSOT**：所有知识以 `.md` 文件存储，带 YAML Front-matter 元数据，最强可移植性
- **语义搜索**：基于本地 Embedding 模型 + ChromaDB，余弦相似度召回 + LLM 答案生成
- **LLM 可插拔**：支持 Kimi / DeepSeek / Ollama 无缝切换
- **零外部依赖**：Embedding 使用本地 `all-MiniLM-L6-v2`，不产生 API 费用
- **跨平台抓取**：支持微信公众号、小红书（MCP 协议）、通用网页

## 快速开始

### 1. 安装依赖

```bash
cd local-rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 LLM API Key
```

支持三种 LLM Provider：

| Provider | 适合场景 | 配置项 |
|----------|----------|--------|
| **Kimi** (默认) | 云端，效果好 | `KIMI_API_KEY` |
| **DeepSeek** | 云端，性价比高 | `DEEPSEEK_API_KEY` |
| **Ollama** | 纯本地，零费用 | 安装 [Ollama](https://ollama.ai) + 拉取模型 |

### 3. 命令行使用

```bash
# 入库 - 发送一个公众号/网页链接
python cli.py ingest "https://mp.weixin.qq.com/s/xxxx"

# 搜索 - 自然语言查询
python cli.py search "之前那篇关于 AI 优化版权投诉工作流的文章提到了哪些工具？"

# 列出所有已入库的知识
python cli.py list

# 查看系统状态
python cli.py stats

# 重建向量索引（从 .md 文件恢复）
python cli.py reindex
```

### 4. API 服务

```bash
# 启动 FastAPI 服务
python cli.py serve
# 或
uvicorn main:app --host 0.0.0.0 --port 8900 --reload
```

#### 入库接口

```bash
curl -X POST http://localhost:8900/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://mp.weixin.qq.com/s/xxxx"}'
```

响应：
```json
{
  "success": true,
  "file_path": "data/260407_a1b2c3d4_AI版权投诉工作流优化.md",
  "title": "AI 版权投诉工作流优化",
  "tags": ["AI工具", "版权保护", "自动化SOP"],
  "message": "入库成功，生成 5 个索引切片"
}
```

#### 检索接口

```bash
curl -X POST http://localhost:8900/search \
  -H "Content-Type: application/json" \
  -d '{"query": "如何用 AI 优化版权投诉流程？", "top_k": 3}'
```

响应：
```json
{
  "answer": "根据知识库中的记录，优化版权投诉工作流主要用到三个工具...[来源: AI版权投诉工作流优化]",
  "sources": [
    {"title": "AI版权投诉工作流优化", "distance": 0.12, "source_url": "..."}
  ]
}
```

## 项目结构

```
├── config.py              # 全局配置
├── main.py                # FastAPI 入口
├── cli.py                 # 命令行入口
├── data/                  # 📂 知识文件 (Markdown SSOT)
├── vectordb/              # ChromaDB 持久化 (可从 data/ 重建)
├── ingestion/             # 数据摄入层
│   ├── base.py            #   抓取器基类
│   ├── router.py          #   URL 路由器
│   ├── wechat.py          #   微信公众号
│   ├── xiaohongshu.py     #   小红书 (MCP)
│   └── general.py         #   通用网页 (Readability)
├── transform/             # AI 清洗层
│   └── llm_cleaner.py     #   LLM 结构化提取 (可插拔)
├── storage/               # 落库引擎
│   └── markdown_engine.py #   Markdown + Front-matter
├── retrieval/             # 向量检索层
│   ├── chunker.py         #   语义切片
│   ├── indexer.py         #   ChromaDB 索引
│   └── searcher.py        #   RAG 答案生成
└── utils/
    └── url_utils.py       # URL 归一化、去重
```

## Markdown 文件格式

每篇入库的知识会生成如下格式的 `.md` 文件：

```markdown
---
title: AI 版权投诉工作流优化指南
summary: 介绍如何利用 AI 工具将版权投诉效率提升 10 倍的三步法
tags:
  - AI工具
  - 版权保护
  - 自动化SOP
source_url: https://mp.weixin.qq.com/s/xxxx
source_platform: wechat
author: 某某公众号
created_at: '2026-04-07 11:30:00'
---

# AI 版权投诉工作流优化指南

> **摘要**: 介绍如何利用 AI 工具将版权投诉效率提升 10 倍的三步法

## 核心步骤
...
```

## 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 框架 | FastAPI | 异步高性能 |
| 向量库 | ChromaDB | 本地持久化，零运维 |
| Embedding | all-MiniLM-L6-v2 | 本地运行，无 API 费用 |
| LLM | Kimi / DeepSeek / Ollama | 可插拔切换 |
| 数据格式 | Markdown + YAML | SSOT，最强可移植 |
| 网页抓取 | requests + BS4 + Readability | 轻量高效 |

## Roadmap

- [x] Phase 1: 底层基座闭环 (ingest + search CLI/API)
- [ ] Phase 2: 对话框接入 (飞书/企微机器人)
- [ ] Phase 3: Web 仪表盘 + 知识联结
