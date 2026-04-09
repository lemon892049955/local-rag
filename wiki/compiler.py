"""Wiki 编译引擎 — v0.3 核心模块

每次入库新文章后，读取新文章 + 现有 Wiki 轻量摘要，
让 LLM 决定创建/更新哪些 Wiki 页面，执行 Append-only 编译。
"""

import json
import re
import logging
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from config import get_llm_config, WIKI_DIR
from wiki import index_builder, page_store

logger = logging.getLogger(__name__)


# ===== Prompts =====

COMPILE_PLAN_PROMPT = """你是一个知识 Wiki 编译器。用户刚刚添加了一篇新文章到知识库。
你的任务是分析这篇文章，并决定如何将其知识编译到 Wiki 中。

{wiki_summary}

---以下是新文章的内容---

文件名: {filename}
标题: {title}
标签: {tags}
摘要: {summary}

正文:
{content}

---

请输出一个 JSON 编译计划。规则：
1. 一篇文章最多创建 1-2 个新页面，聚焦文章最核心的主题，不要过度拆分
2. 不要为过于宽泛的概念建页（如"技术""互联网""AI""工作"）
3. 不要为文章中一笔带过的概念建页，只为有实质内容（>200字描述）的概念建页
4. update_pages 的 action 只能是 "append"（追加新内容）或 "flag_conflict"（标记矛盾）
5. 页面路径格式: topics/概念名.md 或 entities/实体名.md，中文概念直接用中文
6. 如果 Wiki 中已有高度相关的页面，优先更新而非新建

JSON 格式（只输出 JSON，不要任何其他文字）:
{{
  "new_pages": [
    {{"path": "topics/xxx.md", "type": "topic", "title": "页面标题", "reason": "创建原因"}}
  ],
  "update_pages": [
    {{"path": "topics/xxx.md", "action": "append", "reason": "更新原因"}}
  ],
  "cross_references": [
    {{"from": "topics/a.md", "to": "topics/b.md"}}
  ]
}}

如果这篇文章没有足够的新知识需要编译（比如内容太短或太浅），可以返回空计划:
{{"new_pages": [], "update_pages": [], "cross_references": []}}"""


CREATE_PAGE_PROMPT = """你是一个知识 Wiki 编译器。请根据以下文章内容，创建一个 Wiki {page_type}页。

文章文件名: {filename}
文章内容:
{content}

请严格按照以下模板格式输出完整的 Wiki 页面内容（包含 YAML front-matter）。
来源引用格式: [来源: {filename}]
交叉引用格式: [[相关概念名]]
今天日期: {today}
页面标题: {page_title}

{template}

只输出页面内容，不要任何前缀或后缀文字。"""


APPEND_INSIGHT_PROMPT = """你是一个知识 Wiki 编译器。以下是一篇新文章的内容，请提取与 Wiki 页面「{page_title}」相关的新增洞察。

新文章文件名: {filename}
新文章内容:
{article_content}

已有 Wiki 页面「{page_title}」的当前内容:
{page_content}

规则:
1. 只提取与该页面主题相关的新信息
2. 不要重复已有内容
3. 输出纯文本段落（可用 Markdown 格式），不要输出标题和 front-matter
4. 如果新文章与已有内容有矛盾，用 "> ⚠️ 矛盾:" 标记
5. 如果没有相关的新信息，输出空字符串 ""
6. 来源引用: [来源: {filename}]

只输出要追加的内容文本，不要任何前缀后缀。"""


TOPIC_TEMPLATE = """模板:
```
---
type: topic
title: {页面标题}
summary: {一句话摘要，20-50字}
created_at: '{今天日期}'
updated_at: '{今天日期}'
sources:
  - {文章文件名}
---

# {页面标题}

> {一句话定义}

## 核心内容

{结构化干货，使用 ### 组织小节}

[来源: {文章文件名}]

## 相关概念

- [[相关概念1]]
- [[相关概念2]]

## 新增洞察

_(后续更新在此追加)_
```"""

ENTITY_TEMPLATE = """模板:
```
---
type: entity
title: {实体名}
summary: {一句话描述}
entity_type: {person|company|product|other}
created_at: '{今天日期}'
updated_at: '{今天日期}'
sources:
  - {文章文件名}
---

# {实体名}

> {一句话介绍}

## 基本信息

{关键事实}

[来源: {文章文件名}]

## 相关概念

- [[相关概念]]

## 新增洞察

_(后续更新在此追加)_
```"""


class WikiCompiler:
    """Wiki 编译引擎"""

    def __init__(self):
        config = get_llm_config()
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
        self.model = config["model"]

    async def compile(self, article_path: Path) -> dict:
        """编译单篇文章到 Wiki

        Args:
            article_path: data/ 目录下的文章路径

        Returns:
            {"new_pages": [...], "updated_pages": [...], "log_details": [...]}
        """
        # Step 1: 读取文章
        content = article_path.read_text(encoding="utf-8")
        from utils.frontmatter import parse_frontmatter
        meta, body = parse_frontmatter(content)
        if not body.strip():
            return {"new_pages": [], "updated_pages": [], "log_details": []}

        filename = article_path.name
        title = meta.get("title", article_path.stem)
        tags = meta.get("tags", [])
        summary = meta.get("summary", "")

        # 去重检查：如果该文章已被编译过（出现在某个 Wiki 页面的 sources 中），跳过
        existing_pages = page_store.list_wiki_pages()
        for ep in existing_pages:
            if filename in ep.get("sources", []):
                logger.info(f"跳过已编译文章: {filename}")
                return {"new_pages": [], "updated_pages": [], "log_details": [f"跳过: {filename} 已编译过"]}

        # Step 2: 获取 Wiki 轻量摘要
        wiki_summary = index_builder.build_lightweight_summary()

        # Step 3: LLM 制定编译计划
        plan = await self._make_plan(
            wiki_summary=wiki_summary,
            filename=filename,
            title=title,
            tags=", ".join(tags) if tags else "无",
            summary=summary,
            content=body[:8000],  # 输入已是 cleaned_content，8000 字足够
        )

        if not plan:
            return {"new_pages": [], "updated_pages": [], "log_details": ["无需编译"]}

        # Step 4: 执行编译
        new_pages = []
        updated_pages = []
        log_details = []
        today = datetime.now().strftime("%Y-%m-%d")

        # 4a: 创建新页面
        for page_info in plan.get("new_pages", []):
            page_path = page_info.get("path", "")
            page_title = page_info.get("title", "")
            page_type = page_info.get("type", "topic")
            if not page_path or not page_title:
                continue

            try:
                page_content = await self._create_page(
                    page_type=page_type,
                    page_title=page_title,
                    filename=filename,
                    article_content=body[:8000],
                    today=today,
                )
                if page_content:
                    page_store.create_page(page_path, page_content)
                    new_pages.append(page_path)
                    log_details.append(f"创建: {page_path} ({page_title})")
            except Exception as e:
                logger.error(f"创建页面失败 {page_path}: {e}")

        # 4b: 更新已有页面 (Append-only)
        for page_info in plan.get("update_pages", []):
            page_path = page_info.get("path", "")
            if not page_path:
                continue

            existing = page_store.read_page(page_path)
            if not existing:
                continue

            try:
                page_title = existing["meta"].get("title", "")
                insight = await self._generate_insight(
                    page_title=page_title,
                    filename=filename,
                    article_content=body[:6000],
                    page_content=existing["body"][:4000],
                )
                if insight and insight.strip():
                    page_store.append_insight(page_path, today, filename, insight)
                    page_store.add_source_to_frontmatter(page_path, filename)
                    updated_pages.append(page_path)
                    log_details.append(f"更新: {page_path} (追加洞察)")
            except Exception as e:
                logger.error(f"更新页面失败 {page_path}: {e}")

        # 4c: 交叉引用
        for ref in plan.get("cross_references", []):
            from_path = ref.get("from", "")
            to_path = ref.get("to", "")
            if from_path and to_path:
                # 获取 to 页面的标题
                to_page = page_store.read_page(to_path)
                if to_page:
                    to_title = to_page["meta"].get("title", to_path)
                    page_store.append_cross_reference(from_path, to_title)

        # Step 5: 代码更新索引 + 日志
        index_builder.rebuild_index()
        if log_details:
            index_builder.append_log("INGEST", filename, log_details)

        # Step 6: 增量向量索引 (Wiki 页面)
        self._index_affected_pages(new_pages + updated_pages)

        return {
            "new_pages": new_pages,
            "updated_pages": updated_pages,
            "log_details": log_details,
        }

    async def _make_plan(self, wiki_summary: str, filename: str, title: str,
                         tags: str, summary: str, content: str) -> dict:
        """LLM 制定编译计划"""
        prompt = COMPILE_PLAN_PROMPT.format(
            wiki_summary=wiki_summary,
            filename=filename,
            title=title,
            tags=tags,
            summary=summary,
            content=content,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content.strip()
            return self._parse_json(raw)
        except Exception as e:
            logger.error(f"编译计划生成失败: {e}")
            return None

    async def _create_page(self, page_type: str, page_title: str,
                           filename: str, article_content: str, today: str) -> str:
        """LLM 生成新 Wiki 页面内容"""
        template = TOPIC_TEMPLATE if page_type == "topic" else ENTITY_TEMPLATE

        prompt = CREATE_PAGE_PROMPT.format(
            page_type="主题" if page_type == "topic" else "实体",
            filename=filename,
            content=article_content,
            today=today,
            page_title=page_title,
            template=template,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=3000,
            )
            result = response.choices[0].message.content.strip()
            # 清理可能的 markdown 代码块包裹
            if result.startswith("```"):
                result = re.sub(r"^```(?:markdown)?\s*\n?", "", result)
                result = re.sub(r"\n?```\s*$", "", result)
            return result
        except Exception as e:
            logger.error(f"创建页面内容生成失败: {e}")
            return ""

    async def _generate_insight(self, page_title: str, filename: str,
                                article_content: str, page_content: str) -> str:
        """LLM 生成要追加到已有页面的洞察内容"""
        prompt = APPEND_INSIGHT_PROMPT.format(
            page_title=page_title,
            filename=filename,
            article_content=article_content,
            page_content=page_content,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            result = response.choices[0].message.content.strip()
            # 如果 LLM 返回空或明确表示无新信息
            if not result or result in ('""', "无", "无新信息", "没有相关新信息"):
                return ""
            return result
        except Exception as e:
            logger.error(f"生成洞察失败: {e}")
            return ""

    def _index_affected_pages(self, page_paths: list[str]):
        """对编译影响的 Wiki 页面增量索引"""
        try:
            from retrieval.indexer import VectorIndexer
            indexer = VectorIndexer()
            for page_path in page_paths:
                filepath = WIKI_DIR / page_path
                if filepath.exists():
                    indexer.index_file(filepath)
        except Exception as e:
            logger.error(f"Wiki 页面索引失败: {e}")

    def _parse_json(self, raw: str) -> dict:
        """解析 LLM 的 JSON 输出"""
        # 清理 markdown 代码块
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 尝试修复
            cleaned = re.sub(r",\s*}", "}", raw)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                logger.error(f"JSON 解析失败: {raw[:200]}")
                return None

