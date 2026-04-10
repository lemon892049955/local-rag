"""Wiki 分类树维护器 (Taxonomy)

维护 wiki/_taxonomy.yaml，为编译决策提供分类上下文。
分类树由 LLM 自动演化 — 每次新建页面时更新，定期重整理。

核心价值：
1. 为编译计划 Prompt 提供分类体系上下文
2. 解决分类扁平化问题（所有页面平铺在 topics/ 下）
3. 辅助 LLM 决定 topic / entity / insight 类型
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI

from config import get_llm_config, WIKI_DIR

logger = logging.getLogger(__name__)

TAXONOMY_FILE = WIKI_DIR / "_taxonomy.yaml"


# ===== Prompts =====

CLASSIFY_PROMPT = """你是一个知识分类专家。请根据页面的**语义内容**将其归入最合适的分类。

## 参考分类体系（语义引导，非精确匹配）

以下是推荐的分类体系。请根据页面内容的**核心主题语义**选择最接近的分类：

| 分类 | 语义范围（什么样的内容应归入） |
|------|-------------------------------|
| 产品经理面试与求职 | 面试经验、求职攻略、面经分享、简历建议、转行指导、各公司面试流程 |
| AI产品实践与方法论 | AI产品设计、产品经理能力模型、需求交付、PRD写作、产品工作流变革 |
| 产品设计通用方法 | 交互设计原则、用户研究、数据分析方法、A/B测试、统计检验、行为分析 |
| AI应用与行业趋势 | AI行业报告、技术趋势、市场分析、出海机会、投资方向、行业预测 |
| AI产品案例分析 | 具体AI产品/公司/社区的分析（如Loopit、陪伴产品）、商业模式分析 |
| AIGC内容创作 | AI短剧、AI漫剧、AI角色设计、AI视频制作、AIGC创作方法论 |
| 开发工具与技术实践 | Vibe Coding、编程助手、RAG系统、知识库搭建、技术架构、代码实践 |
| 产品人物与思想 | 产品大牛的思想/哲学、人物访谈、商业洞察（如张小龙、梁宁） |
| 项目文档 | 项目自身的架构文档、版本记录、迭代计划、代码评审 |

你也可以新建分类，但**必须在 reason 中说明为什么上述分类都不合适**。

## 当前分类树（已有页面分布）
{taxonomy_tree}

## 新页面信息
- 路径: {page_path}
- 标题: {page_title}
- 摘要: {page_summary}
- 类型: {page_type}

## 分类判断规则

1. **按语义而非关键词分类**：即使标题含"AI"，如果内容主要讲面试经验，应归入「产品经理面试与求职」
2. **同主题聚合**：面试经验类内容（无论是字节/腾讯/阿里/智谱）都应归入同一分类
3. **优先使用已有分类**：如果已有分类能容纳，不要新建
4. **页面类型判断**：
   - topic: 围绕概念/方法论/技术/趋势的知识主题
   - entity: 围绕具体人物/公司/产品/品牌的事实集合
   - insight: 跨主题的综合对比分析

只输出 JSON（不要任何其他文字），格式:
{{"category": "分类名", "subcategory": "子分类名(可为空字符串)", "suggested_type": "topic|entity|insight", "reason": "一句话原因"}}"""


# ===== 核心函数 =====

def load_taxonomy() -> dict:
    """加载分类树"""
    if TAXONOMY_FILE.exists():
        try:
            data = yaml.safe_load(TAXONOMY_FILE.read_text(encoding="utf-8"))
            return data if data else {"version": 1, "categories": {}}
        except Exception:
            return {"version": 1, "categories": {}}
    return {"version": 1, "categories": {}}


def save_taxonomy(taxonomy: dict):
    """保存分类树"""
    yaml_str = yaml.dump(
        taxonomy, default_flow_style=False,
        allow_unicode=True, sort_keys=False,
    )
    header = (
        f"# 由编译引擎自动维护，请勿手动编辑\n"
        f"# 上次更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    )
    TAXONOMY_FILE.write_text(header + yaml_str, encoding="utf-8")


def get_taxonomy_summary() -> str:
    """生成分类树的文本摘要（供编译 Prompt 使用）"""
    taxonomy = load_taxonomy()
    categories = taxonomy.get("categories", {})
    if not categories:
        return ""

    lines = ["当前分类体系："]
    for cat_name, cat_data in categories.items():
        if isinstance(cat_data, dict):
            pages = cat_data.get("pages", [])
            desc = cat_data.get("description", "")
            children = cat_data.get("children", {})
            desc_str = f" — {desc}" if desc else ""
            lines.append(f"├─ {cat_name}{desc_str} ({len(pages)} 个页面)")
            for child_name, child_data in (children or {}).items():
                child_pages = child_data.get("pages", []) if isinstance(child_data, dict) else []
                lines.append(f"│  ├─ {child_name} ({len(child_pages)} 个页面)")

    return "\n".join(lines)


def add_page_to_taxonomy(page_path: str, category: str, subcategory: str = "",
                         description: str = ""):
    """将页面添加到分类树

    Args:
        page_path: Wiki 页面路径，如 "topics/xxx.md"
        category: 一级分类名
        subcategory: 二级分类名（可选）
        description: 分类描述（仅新建分类时使用）
    """
    taxonomy = load_taxonomy()
    categories = taxonomy.setdefault("categories", {})

    if category not in categories:
        categories[category] = {
            "description": description,
            "pages": [],
            "children": {},
        }

    cat = categories[category]
    if not isinstance(cat, dict):
        cat = {"description": "", "pages": [], "children": {}}
        categories[category] = cat

    if subcategory:
        children = cat.setdefault("children", {})
        if subcategory not in children:
            children[subcategory] = {"pages": []}
        sub = children[subcategory]
        if not isinstance(sub, dict):
            sub = {"pages": []}
            children[subcategory] = sub
        if page_path not in sub.get("pages", []):
            sub.setdefault("pages", []).append(page_path)
    else:
        if page_path not in cat.get("pages", []):
            cat.setdefault("pages", []).append(page_path)

    save_taxonomy(taxonomy)
    logger.info(f"Taxonomy: {page_path} → {category}" + (f"/{subcategory}" if subcategory else ""))


async def classify_page(page_path: str, page_title: str,
                        page_summary: str, page_type: str) -> Optional[dict]:
    """用 LLM 对新页面进行分类

    Returns:
        {"category": "...", "subcategory": "...", "suggested_type": "...", "reason": "..."}
        或 None（分类失败时）
    """
    import json

    taxonomy_tree = get_taxonomy_summary()
    if not taxonomy_tree:
        taxonomy_tree = "当前暂无分类体系（这是第一个页面）。"

    config = get_llm_config()
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

    prompt = CLASSIFY_PROMPT.format(
        taxonomy_tree=taxonomy_tree,
        page_path=page_path,
        page_title=page_title,
        page_summary=page_summary,
        page_type=page_type,
    )

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()

        # 清理 markdown 代码块
        import re
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1)

        result = json.loads(raw)
        return result
    except Exception as e:
        logger.warning(f"LLM 页面分类失败: {e}")
        return None


async def init_taxonomy_from_existing():
    """从现有 Wiki 页面初始化分类树（一次性操作）

    v0.5: 改为 LLM 语义分类，不再用关键词规则。
    """
    from wiki.page_store import list_wiki_pages

    pages = list_wiki_pages()
    if not pages:
        return

    # 清空后逐个用 LLM 分类
    taxonomy = {"version": 1, "categories": {}}
    save_taxonomy(taxonomy)

    success = 0
    for page in pages:
        path = page.get("path", "")
        title = page.get("title", "")
        summary = page.get("summary", "")
        page_type = page.get("type", "topic")

        try:
            result = await classify_page(
                page_path=path,
                page_title=title,
                page_summary=summary,
                page_type=page_type,
            )
            if result:
                add_page_to_taxonomy(
                    path,
                    result.get("category", "未分类"),
                    result.get("subcategory", ""),
                )
                success += 1
        except Exception as e:
            logger.warning(f"Taxonomy 初始化分类失败 {path}: {e}")
            add_page_to_taxonomy(path, "未分类")

    logger.info(f"Taxonomy 初始化完成: {success}/{len(pages)} 个页面已分类")
