---
type: topic
title: Karpathy的AI个人知识库搭建教程
summary: Andrej Karpathy分享了使用AI搭建个人知识库的方法论和提示词。
created_at: '2026-04-08'
updated_at: '2026-04-08'
sources:
- 260407_fiSkSzHX_Karpathy的AI个人知识库搭建教程.md
- 260408_NUqtHeCb_利用LLM构建个人知识库的模式.md
---
# Karpathy的AI个人知识库搭建教程

> Andrej Karpathy 在 GitHub Gist 发布了一份名为 LLM Wiki 的 Markdown 文件，提供了搭建个人知识库的方法论和提示词。

## 核心内容

### RAG的根本问题
不是检索能力，是没有记忆。

### 三层架构的思维
只读区 / 写作区 / 规则区。

### 三个操作关键行为
录入 / 查询 / 检查。

### 特殊文件的理解
index.md 和 log.md。

### 工具清单

### LLM创建的Wiki的价值

[来源: 260407_fiSkSzHX_Karpathy的AI个人知识库搭建教程.md]

## 相关概念
- [[AI技术梳理]]

- [[个人知识库]]
- [[AI辅助知识管理]]

## 新增洞察

### 2026-04-08 | 来源: 260408_NUqtHeCb_利用LLM构建个人知识库的模式.md

通过LLM维护一个持续更新的wiki，实现知识的累积和整合。与传统的RAG系统不同，LLM不仅检索文档，而是构建和维护一个持久的、结构化的、相互链接的markdown文件集合，即wiki。LLM读取新来源，提取关键信息，并将其整合到现有wiki中，更新实体页面、修订主题摘要、记录新数据与旧声明的矛盾，从而不断强化或挑战不断发展的综合知识。这种wiki是一个持续增长的复合产物，其中的交叉引用、矛盾标记和综合信息已经就绪，随着你添加的每个来源和每个问题而不断丰富。

LLM-Wiki的功能包括查询、答案归档和Lint检查。查询时，LLM搜索相关页面，阅读并综合答案，可以以不同形式呈现，如markdown页面、比较表格、幻灯片（Marp）、图表（matplotlib）、画布。好的答案可以作为新页面归档回wiki，如比较、分析、新发现的联系，这些都是有价值的，不应消失在聊天记录中。LLM还可以定期检查wiki的健康状态，寻找页面间的矛盾、过时声明、孤立页面、缺少页面的重要概念、缺失交叉引用、可通过网络搜索填补的数据空白。

导航文件包括index.md和log.md。index.md按类别组织，列出wiki中的所有页面，包括链接、一句话摘要和可选的元数据（如日期或来源计数）。LLM在每次摄取时更新它。log.md按时间顺序记录wiki发生的事情，包括摄取、查询、Lint检查。如果每个条目以一致的前缀开始，可以用简单的Unix工具解析。[来源：260408_NUqtHeCb_利用LLM构建个人知识库的模式.md]

[来源: 260408_NUqtHeCb_利用LLM构建个人知识库的模式.md]


### 2026-04-08 | 来源: 260407_fiSkSzHX_Karpathy的AI个人知识库搭建教程.md

新增洞察：

- Andrej Karpathy 在 GitHub Gist 发布了名为 LLM Wiki 的 Markdown 文件，提供了搭建个人知识库的方法论和提示词。[来源: 260407_fiSkSzHX_Karpathy的AI个人知识库搭建教程.md]

[来源: 260407_fiSkSzHX_Karpathy的AI个人知识库搭建教程.md]