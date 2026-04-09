---
type: topic
title: Karpathy的AI个人知识库构建指南
summary: Andrej Karpathy分享使用LLMs构建个人知识库的方法和提示词。
created_at: '2026-04-09'
updated_at: '2026-04-09'
sources:
  - 260409_8WK4z8Y5_Karpathy的AI个人知识库构建指南.md
---

# Karpathy的AI个人知识库构建指南

> Andrej Karpathy提供了一份使用大型语言模型（LLMs）构建个人知识库的保姆级教程。

## 核心内容

### 核心理念
Karpathy提出了一种不同于传统RAG（Retrieval-Augmented Generation）的方法，即通过LLMs持续构建并维护一个永久性的Wiki，而不是在每次提问时检索原始文档。这种方法允许知识积累和复用，避免了重复劳动。

### 三层架构
- **第一层：原始资料（Raw Sources）** - 只读，不可修改，存放所有文章、论文、图片、数据文件等。
- **第二层：Wiki知识库** - 由LLM负责写作和维护，包括摘要页、实体页、概念页等。
- **第三层：Schema规范文档** - 配置文件，定义Wiki的目录结构、页面格式和新资料处理流程。

### 三个操作关键行为
- **录入（Ingest）** - 处理新资料，更新Wiki页面。
- **查询（Query）** - 从知识库中提取信息，综合作答。
- **检查（Check）** - 审核和维护Wiki内容的准确性和完整性。

### 特殊文件的理解
- **index.md** - 作为Wiki的索引，帮助快速定位相关内容。
- **log.md** - 记录所有更新和变更的历史。

## 相关概念
- [[Karpathy的AI个人知识库搭建教程]]
- [[利用LLM构建个人知识库的模式]]

- [[LLMs]]
- [[知识管理]]
- [[个人知识库]]

## 新增洞察

_(后续更新在此追加)_

[来源: 260409_8WK4z8Y5_Karpathy的AI个人知识库构建指南.md]