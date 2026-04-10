"""Wiki 编译引擎 — v0.4 核心模块

v0.4 改造：
- [P0] 编译前语义去重门控：用 Embedding 找相似页面，注入 Prompt
- [P0] 洞察追加去重：强化 Prompt + Embedding 相似度校验
- [P1] 丰富摘要上下文：聚类分组 + Taxonomy 分类树
- [P1] 自演化 Taxonomy：新页面自动分类归档

每次入库新文章后，读取新文章 + 现有 Wiki 轻量摘要 + 相似页面信息，
让 LLM 决定创建/更新哪些 Wiki 页面，执行 Append-only 编译。
"""

import json
import re
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from openai import OpenAI

from config import get_llm_config, WIKI_DIR
from wiki import index_builder, page_store, taxonomy

logger = logging.getLogger(__name__)


# ===== Prompts =====

COMPILE_PLAN_PROMPT = """你是一个知识 Wiki 编译器。用户刚刚添加了一篇新文章到知识库。
你的任务是分析这篇文章，并决定如何将其知识编译到 Wiki 中。

{wiki_summary}

{similar_pages_section}

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
2. ⚠️ 如果上方列出了相似度 ≥ 85% 的已有页面，你**必须**优先使用 update_pages 追加到该页面，**禁止**创建内容重复的新页面
3. 如果相似度在 70%-85% 之间，需在 reason 中明确说明为什么选择新建而非更新
4. 不要为过于宽泛的概念建页（如"技术""互联网""AI""工作"）
5. 不要为文章中一笔带过的概念建页，只为有实质内容（>200字描述）的概念建页
6. update_pages 的 action 只能是 "append"（追加新内容）或 "flag_conflict"（标记矛盾）
7. 页面类型决策指引（type 字段）：
   - **topic**: 围绕一个概念/方法论/技术/趋势的知识主题（如"Vibe Coding工作流""AI时代产品经理实践"）
   - **entity**: 围绕一个具体的人物/公司/产品/品牌的事实（如"霸王茶姬""Karpathy""Claude Code"）
   - 如果文章主要在讲某个公司、产品或人物的具体事实，应建 entity 页
8. 页面路径格式: topics/概念名.md 或 entities/实体名.md，中文概念直接用中文

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
1. 只提取与该页面主题**直接相关**的新信息
2. **严格去重** — 在输出前，逐条检查你要追加的每个观点：
   a) 如果已有"核心内容"或历史"新增洞察"中已有相同/近似表述 → 必须跳过
   b) 如果新信息只是已有观点的换一种说法、换一个角度复述 → 必须跳过
   c) 只有**真正的增量信息**（新事实、新案例、新数据、新观点）才值得追加
3. **关联度要求** — 只追加与页面核心主题直接相关的信息：
   - 如果新文章的主题与本页面的核心主题关联度低，应该输出空字符串
   - 不要强行建立关联（例如"梁宁商业思考"与"AI产品经理焦虑"关联度低，应跳过）
4. 如果经过筛选后没有任何增量信息，直接输出空字符串 ""
5. 输出纯文本段落（可用 Markdown 格式），不要输出标题和 front-matter
6. 如果新文章与已有内容有矛盾，用 "> ⚠️ 矛盾:" 标记
7. 来源引用: [来源: {filename}]
8. 每条洞察尽量精简，只写增量部分，不要重复铺垫已有上下文

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

        # Step 2: 获取 Wiki 轻量摘要 + Taxonomy 分类上下文
        wiki_summary = index_builder.build_lightweight_summary()
        taxonomy_summary = taxonomy.get_taxonomy_summary()
        if taxonomy_summary:
            wiki_summary = wiki_summary + "\n\n" + taxonomy_summary

        # Step 2.5: 语义去重 — 用 Embedding 找到与新文章高度相似的 Wiki 页面
        similar_pages = await self._find_similar_pages(body)
        similar_pages_section = ""
        if similar_pages:
            parts = []
            for sp in similar_pages:
                parts.append(
                    f"  - [{sp['path']}] {sp['title']} (相似度: {sp['similarity']:.0%})\n"
                    f"    预览: {sp['preview'][:100]}..."
                )
            similar_pages_section = (
                "⚠️ 以下 Wiki 页面与本文高度相似（基于语义向量检测），请优先考虑 update 而非 new：\n"
                + "\n".join(parts)
            )
            logger.info(f"发现 {len(similar_pages)} 个相似页面: {[sp['title'] for sp in similar_pages]}")

        # Step 3: LLM 制定编译计划
        plan = await self._make_plan(
            wiki_summary=wiki_summary,
            similar_pages_section=similar_pages_section,
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

                    # P1: 自动分类归档到 Taxonomy
                    try:
                        classify_result = await taxonomy.classify_page(
                            page_path=page_path,
                            page_title=page_title,
                            page_summary=page_info.get("reason", ""),
                            page_type=page_type,
                        )
                        if classify_result:
                            taxonomy.add_page_to_taxonomy(
                                page_path,
                                classify_result.get("category", "未分类"),
                                classify_result.get("subcategory", ""),
                            )
                    except Exception as te:
                        logger.warning(f"Taxonomy 分类失败: {te}")

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
                    # P0: 代码层去重校验 — Embedding 相似度检查
                    is_novel = self._check_insight_novelty(insight, existing["body"])
                    if not is_novel:
                        logger.info(f"洞察与已有内容过于相似，跳过追加: {page_path}")
                        log_details.append(f"跳过: {page_path} (洞察无增量)")
                        continue
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

    async def _make_plan(self, wiki_summary: str, similar_pages_section: str,
                         filename: str, title: str,
                         tags: str, summary: str, content: str) -> dict:
        """LLM 制定编译计划"""
        prompt = COMPILE_PLAN_PROMPT.format(
            wiki_summary=wiki_summary,
            similar_pages_section=similar_pages_section,
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

    async def _find_similar_pages(self, article_text: str, threshold: float = 0.82) -> list[dict]:
        """用 Embedding 检索与新文章高度相似的 Wiki 页面

        Args:
            article_text: 文章正文
            threshold: 相似度阈值（cosine similarity），默认 0.82

        Returns:
            [{"path": "topics/xxx.md", "title": "...", "similarity": 0.89, "preview": "..."}]
        """
        try:
            from retrieval.indexer import VectorIndexer
            indexer = VectorIndexer()

            # 用文章前 2000 字做语义检索
            hits = indexer.search(article_text[:2000], top_k=8)

            # 只保留 Wiki 来源的命中
            wiki_hits = [h for h in hits if "wiki" in h.get("source_file", "")]

            similar = []
            seen_pages = set()
            for hit in wiki_hits:
                # ChromaDB cosine distance = 1 - cosine_similarity
                cosine_sim = 1 - hit.get("distance", 1.0)
                source_file = hit.get("source_file", "")

                # 从 source_file 提取 Wiki 页面路径
                page_path = self._extract_wiki_page_path(source_file)
                if not page_path or page_path in seen_pages:
                    continue
                seen_pages.add(page_path)

                if cosine_sim >= threshold:
                    similar.append({
                        "path": page_path,
                        "title": hit.get("title", ""),
                        "similarity": round(cosine_sim, 3),
                        "preview": hit.get("text", "")[:200],
                    })

            similar.sort(key=lambda x: -x["similarity"])
            return similar[:5]
        except Exception as e:
            logger.warning(f"语义去重检查失败: {e}")
            return []

    def _extract_wiki_page_path(self, source_file: str) -> str:
        """从文件路径中提取 Wiki 页面相对路径（如 topics/xxx.md）"""
        for prefix in ["topics/", "entities/", "insights/"]:
            idx = source_file.find(prefix)
            if idx != -1:
                return source_file[idx:]
        return ""

    def _check_insight_novelty(self, insight: str, page_content: str) -> bool:
        """检查追加洞察是否与已有内容有足够差异

        Returns:
            True = 有增量价值，可以追加
            False = 与已有内容过于相似，应跳过
        """
        try:
            from retrieval.indexer import VectorIndexer
            indexer = VectorIndexer()

            # 文档模式生成 embedding
            indexer.embedding_fn.set_query_mode(False)

            insight_embedding = np.array(indexer.embedding_fn([insight])[0])

            # 将已有内容分段，逐段比对
            paragraphs = [p.strip() for p in page_content.split("\n\n") if len(p.strip()) > 50]
            if not paragraphs:
                return True

            page_embeddings = indexer.embedding_fn(paragraphs[:20])

            # 计算最大相似度（已归一化，dot product = cosine similarity）
            max_sim = 0.0
            for pe in page_embeddings:
                sim = float(np.dot(insight_embedding, np.array(pe)))
                max_sim = max(max_sim, sim)

            logger.info(f"洞察与已有内容最大相似度: {max_sim:.3f}")
            return max_sim < 0.80
        except Exception as e:
            logger.warning(f"洞察去重检查失败: {e}")
            return True  # 失败时放行

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

