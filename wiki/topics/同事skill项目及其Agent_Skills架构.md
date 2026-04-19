---
type: topic
title: 深度解析GitHub上的'同事.skill'项目及其Agent Skills架构
summary: 从开发者视角深度解析GitHub上'同事.skill'项目的技术架构和Agent Skills发展现状。
created_at: '2026-04-10'
updated_at: '2026-04-10'
sources:
  - 260409_DZzwKmPn_深度解析GitHub上的'同事.skill'项目及其Agent_Skills架构.md
---

# 深度解析GitHub上的'同事.skill'项目及其Agent Skills架构

> 本文从开发者视角深度解析GitHub上'同事.skill'项目的技术架构和Agent Skills发展现状。

## 核心内容

### 事件回顾：从恶作剧到全网刷屏
- GitHub上的“同事.skill”项目由开发者titanwings发起，通过提供同事的邮件聊天记录、钉钉文件和工作文档，AI能生成一个完美复刻其工作方式的数字分身。
- 核心亮点包括复刻知识和工作流，模仿说话语气、甩锅话术、对规则的反向执行，以及区分“字节范儿”和“阿里味儿”等不同企业文化风格。
- 该项目的思路值得技术人关注：自动识别交互内容中的核心知识，替换为“正确的废话”。

### 技术深度解析：SKILL.md多层架构
- Agent Skills是2025年Anthropic率先发布的AI能力扩展标准。核心思想是一个SKILL.md文件就能给AI装上新能力。
- 到2026年，该标准被Cursor、Codex CLI、OpenClaw全面采用，Skill Marketplace已收录70万+技能包。
- 这些火爆的“人格.skill”都采用了相同的架构，包括知识层（Memory）和人格层（Persona）。

### Agent Skills生态：70万+技能包的世界
- 展示了一个“code-reviewer”技能包的基本框架，包括角色定义、检查规则和输出格式。
- 安装只需一行命令，支持多个Skill时需要一个路由层，根据任务类型自动分发到合适的Skill。

### 发展记录与最佳实践
- 提供了敏感数据保护的示例，展示了如何发布Skill前清洗敏感信息。

[来源: 260409_DZzwKmPn_深度解析GitHub上的'同事.skill'项目及其Agent_Skills架构.md]

## 相关概念
- [[OpenClaw]]
- [[Cursor]]
- [[Codex]]
- [[Anthropic]]
- [[Github开源项目动态]]

- [[AI能力扩展标准]]
- [[数字分身]]

## 新增洞察

_(后续更新在此追加)_