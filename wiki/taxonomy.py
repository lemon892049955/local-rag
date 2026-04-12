"""Wiki 分类树维护器 (Taxonomy) — 自演化版

维护 wiki/_taxonomy.yaml，为编译决策提供分类上下文。
分类树由 LLM **完全自动演化** — 根据入库内容自动新建、归并、拆分分类。

核心价值：
1. 为编译计划 Prompt 提供分类体系上下文
2. 解决分类扁平化问题（所有页面平铺在 topics/ 下）
3. 辅助 LLM 决定 topic / entity / insight 类型
4. 分类体系随知识库内容动态生长
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI

from config import get_llm_config, WIKI_DIR

logger = logging.getLogger(__name__)

TAXONOMY_FILE = WIKI_DIR / "_taxonomy.yaml"

# 分类体系约束
MAX_CATEGORIES = 12     # 一级分类上限，超过触发合并
MIN_PAGES_PER_CAT = 2   # 低于此数的分类在重整理时会被合并

# 并发保护：防止 reclassify 与编译同时写 _taxonomy.yaml
_taxonomy_lock = asyncio.Lock()
_reclassify_running = False

# LLM 客户端单例（复用连接池）
_llm_client = None
_llm_model = None


def _get_llm_client():
    """获取 LLM 客户端单例"""
    global _llm_client, _llm_model
    if _llm_client is None:
        config = get_llm_config()
        _llm_client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        _llm_model = config["model"]
    return _llm_client, _llm_model


# ===== Prompts =====

CLASSIFY_PROMPT = """你是一个知识分类专家。请根据页面的**语义内容**将其归入最合适的分类。

## 当前分类体系
{taxonomy_tree}

## 一级分类规则

1. **优先归入已有分类/子分类**：语义匹配时直接归入
2. **允许新建一级分类**：如果现有分类都不合适，大胆新建。目标是 **6-10 个一级分类**
3. **一级分类 = 知识领域**：每个一级分类代表一个独立的知识领域，不要把不相关的领域强行塞到同一个分类
4. **分类命名**：4-10 个字中文短语

### 领域划分指引（不同领域应该是不同的一级分类）

- 产品设计方法（设计原则、交互设计、用户研究） ≠ AI产品经理（面试、求职、职业发展）
- 数据分析与统计（统计检验、数据方法） ≠ 开发工具（编程工具、RAG系统）
- 产品人物与思想（人物故事、商业哲学） ≠ AI产品案例（具体产品分析）
- 项目文档（本项目的架构、代码） ≠ 通用开发工具
- AIGC内容创作（短剧、视频、角色设计） ≠ AI产品实践

### ⚠️ 常见错误（必须避免）

- ❌ 把统计检验方法（t检验、卡方检验）放到"开发工具实践" → ✅ 应放到"数据分析与统计"
- ❌ 把设计原则（尼尔森原则、一致性原则）放到"AI产品经理" → ✅ 应放到"产品设计方法"
- ❌ 把项目自身文档（Local RAG架构）放到"开发工具实践" → ✅ 应放到"项目文档"
- ❌ 把社交平台（小红书、抖音、饭否）放到"AI产品" → ✅ 应放到"互联网产品"或单独分类
- ❌ 把人物（张小龙、Karpathy）放到"AI产品经理" → ✅ 应放到"产品人物与思想"

## 二级分类（subcategory）规则

5. **积极使用子分类**：一级分类下内容可进一步区分时，用 subcategory
6. 命名 2-6 个字，简洁准确
7. entity/concept 按其**所属知识领域**分，不要按类型堆

## 新页面信息
- 路径: {page_path}
- 标题: {page_title}
- 摘要: {page_summary}
- 类型: {page_type}

只输出 JSON（不要任何其他文字），格式:
{{"category": "已有分类名 或 新分类名", "subcategory": "子分类名（建议填写）", "description": "新分类时必填", "suggested_type": "topic|entity|concept", "reason": "一句话原因"}}"""

MERGE_PROMPT = """你是一个知识分类专家。当前分类体系需要优化。

## 当前分类体系
{taxonomy_tree}

## 优化规则

1. 将语义相近的**一级分类**合并，目标是 **5-10 个一级分类**
2. 合并时选一个更概括的分类名
3. 只合并语义确实重叠的分类，不强行合并
4. 页面数少于 {min_pages} 的小分类优先被合并到最相近的大分类
5. 分类名要简洁（4-10 个字）
4. 页面数少于 {min_pages} 的小分类优先被合并到最相近的大分类
5. 分类名要简洁（4-10 个字），不要太泛也不要太窄

输出 JSON 数组，每个元素表示一次合并操作:
[{{"from": "被合并的分类名", "to": "合并到的目标分类名", "reason": "合并原因"}}]

如果不需要合并，输出空数组: []"""


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
    """保存分类树（原子写入：先写临时文件再重命名）"""
    yaml_str = yaml.dump(
        taxonomy, default_flow_style=False,
        allow_unicode=True, sort_keys=False,
    )
    header = (
        f"# 由编译引擎自动维护，请勿手动编辑\n"
        f"# 上次更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    )
    tmp_file = TAXONOMY_FILE.with_suffix(".yaml.tmp")
    tmp_file.write_text(header + yaml_str, encoding="utf-8")
    tmp_file.rename(TAXONOMY_FILE)


def get_taxonomy_summary() -> str:
    """生成分类树的文本摘要（供编译/分类 Prompt 使用）

    包含每个分类和子分类的页面标题列表，帮助 LLM 做出精确的分类决策。
    """
    taxonomy = load_taxonomy()
    categories = taxonomy.get("categories", {})
    if not categories:
        return ""

    def _page_label(path: str) -> str:
        """从路径提取简短标签"""
        name = path.rsplit("/", 1)[-1].replace(".md", "")
        return name

    lines = ["当前分类体系："]
    for cat_name, cat_data in categories.items():
        if not isinstance(cat_data, dict):
            continue
        pages = cat_data.get("pages", [])
        desc = cat_data.get("description", "")
        children = cat_data.get("children", {})
        desc_str = f" — {desc}" if desc else ""
        lines.append(f"├─ {cat_name}{desc_str} ({len(pages)} 个页面)")
        # 显示该分类下的页面标题（最多 5 个 + 省略）
        for p in pages[:5]:
            lines.append(f"│  · {_page_label(p)}")
        if len(pages) > 5:
            lines.append(f"│  · ...等 {len(pages)} 个")
        # 子分类
        for child_name, child_data in (children or {}).items():
            child_pages = child_data.get("pages", []) if isinstance(child_data, dict) else []
            lines.append(f"│  ├─ {child_name} ({len(child_pages)} 个页面)")
            for p in child_pages[:3]:
                lines.append(f"│  │  · {_page_label(p)}")
            if len(child_pages) > 3:
                lines.append(f"│  │  · ...等 {len(child_pages)} 个")

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


# ===== 人工分类调整（pinned 机制）=====

def get_pinned_pages() -> set:
    """获取所有被用户手动钉住的页面路径"""
    taxonomy = load_taxonomy()
    return set(taxonomy.get("pinned_pages", []))


def _remove_page_from_all(taxonomy: dict, page_path: str):
    """从分类树中移除某页面（所有出现位置）"""
    for cat_data in taxonomy.get("categories", {}).values():
        if not isinstance(cat_data, dict):
            continue
        pages = cat_data.get("pages", [])
        if page_path in pages:
            pages.remove(page_path)
        for child_data in cat_data.get("children", {}).values():
            if isinstance(child_data, dict):
                child_pages = child_data.get("pages", [])
                if page_path in child_pages:
                    child_pages.remove(page_path)


def move_page_category(page_path: str, category: str, subcategory: str = "",
                       description: str = "") -> dict:
    """用户手动将页面移到指定分类（自动标记 pinned）

    被 pinned 的页面在 AI 重分类时会被跳过，保留用户的判断。

    Returns:
        {"success": True, "pinned": True, "category": "...", "subcategory": "..."}
    """
    taxonomy = load_taxonomy()

    # 1. 从旧位置移除
    _remove_page_from_all(taxonomy, page_path)

    # 2. 添加到新位置
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
            sub["pages"].append(page_path)
    else:
        if page_path not in cat.get("pages", []):
            cat.setdefault("pages", []).append(page_path)

    # 3. 标记 pinned
    pinned = set(taxonomy.get("pinned_pages", []))
    pinned.add(page_path)
    taxonomy["pinned_pages"] = sorted(pinned)

    save_taxonomy(taxonomy)
    logger.info(f"Taxonomy 手动调整: {page_path} → {category}/{subcategory} (pinned)")

    return {
        "success": True,
        "pinned": True,
        "category": category,
        "subcategory": subcategory,
    }


async def classify_page(page_path: str, page_title: str,
                        page_summary: str, page_type: str) -> Optional[dict]:
    """用 LLM 对新页面进行分类（动态分类，允许新建）

    Returns:
        {"category": "...", "subcategory": "...", "description": "...",
         "suggested_type": "...", "reason": "..."}
        或 None（分类失败时）
    """
    taxonomy_tree = get_taxonomy_summary()
    if not taxonomy_tree:
        taxonomy_tree = "当前暂无分类体系（这是第一个页面，请根据内容创建合适的分类）。"

    client, model = _get_llm_client()

    prompt = CLASSIFY_PROMPT.format(
        taxonomy_tree=taxonomy_tree,
        page_path=page_path,
        page_title=page_title,
        page_summary=page_summary,
        page_type=page_type,
    )

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        if not response.choices:
            logger.warning("LLM 分类返回空 choices")
            return None
        content = response.choices[0].message.content
        if not content:
            return None
        raw = content.strip()

        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1)

        return json.loads(raw)
    except Exception as e:
        logger.warning(f"LLM 页面分类失败: {e}")
        return None


async def maybe_merge_categories():
    """当分类数超过上限时，用 LLM 自动合并语义相近的分类"""
    taxonomy = load_taxonomy()
    categories = taxonomy.get("categories", {})

    if len(categories) <= MAX_CATEGORIES:
        return

    logger.info(f"Taxonomy 分类数 {len(categories)} 超过上限 {MAX_CATEGORIES}，触发自动合并")

    taxonomy_tree = get_taxonomy_summary()
    client, model = _get_llm_client()

    prompt = MERGE_PROMPT.format(
        taxonomy_tree=taxonomy_tree,
        min_pages=MIN_PAGES_PER_CAT,
    )

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        if not response.choices:
            logger.warning("LLM 合并返回空 choices")
            return
        content = response.choices[0].message.content
        if not content:
            return
        raw = content.strip()

        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1)

        merges = json.loads(raw)
        if not isinstance(merges, list) or not merges:
            return

        # 执行合并
        for merge in merges:
            from_cat = merge.get("from", "")
            to_cat = merge.get("to", "")
            if not from_cat or not to_cat or from_cat == to_cat:
                continue
            if from_cat not in categories:
                continue

            # 确保目标分类存在
            if to_cat not in categories:
                categories[to_cat] = {
                    "description": "",
                    "pages": [],
                    "children": {},
                }

            # 移动页面
            from_data = categories[from_cat]
            if isinstance(from_data, dict):
                to_data = categories[to_cat]
                for page in from_data.get("pages", []):
                    if page not in to_data.get("pages", []):
                        to_data.setdefault("pages", []).append(page)
                # 移动子分类
                for child_name, child_data in from_data.get("children", {}).items():
                    to_children = to_data.setdefault("children", {})
                    if child_name not in to_children:
                        to_children[child_name] = child_data
                    else:
                        for page in child_data.get("pages", []):
                            if page not in to_children[child_name].get("pages", []):
                                to_children[child_name].setdefault("pages", []).append(page)

            del categories[from_cat]
            logger.info(f"Taxonomy 合并: 「{from_cat}」→「{to_cat}」({merge.get('reason', '')})")

        save_taxonomy(taxonomy)
        logger.info(f"Taxonomy 合并完成，剩余 {len(categories)} 个分类")

    except Exception as e:
        logger.warning(f"Taxonomy 自动合并失败: {e}")


async def init_taxonomy_from_existing():
    """从现有 Wiki 页面初始化分类树（一次性操作）

    分类完全由 LLM 根据内容动态生成，不预设任何固定分类。
    完成后自动触发合并检查，确保分类数量合理。
    使用锁防止与编译任务并发写入。
    """
    global _reclassify_running
    if _reclassify_running:
        logger.warning("重分类已在运行中，跳过")
        return
    _reclassify_running = True

    try:
        from wiki.page_store import list_wiki_pages, read_page

        pages = list_wiki_pages()
        if not pages:
            return

        # 备份旧分类，万一中断可恢复
        old_taxonomy = load_taxonomy()
        backup_file = TAXONOMY_FILE.with_suffix(".yaml.bak")
        if TAXONOMY_FILE.exists():
            backup_file.write_text(TAXONOMY_FILE.read_text(encoding="utf-8"), encoding="utf-8")

        # 保留 pinned 页面的分类信息
        pinned = set(old_taxonomy.get("pinned_pages", []))
        pinned_entries = []  # [(page_path, category, subcategory)]
        if pinned:
            for cat_name, cat_data in old_taxonomy.get("categories", {}).items():
                if not isinstance(cat_data, dict):
                    continue
                for p in cat_data.get("pages", []):
                    if p in pinned:
                        pinned_entries.append((p, cat_name, ""))
                for child_name, child_data in cat_data.get("children", {}).items():
                    if isinstance(child_data, dict):
                        for p in child_data.get("pages", []):
                            if p in pinned:
                                pinned_entries.append((p, cat_name, child_name))
            logger.info(f"保留 {len(pinned_entries)} 个用户钉住的分类")

        # 清空分类树（保留 pinned_pages 列表）
        new_taxonomy = {"version": 1, "categories": {}, "pinned_pages": sorted(pinned)}
        save_taxonomy(new_taxonomy)

        # 先恢复 pinned 页面的分类
        for page_path, cat, subcat in pinned_entries:
            add_page_to_taxonomy(page_path, cat, subcat)

        success = 0
        for page in pages:
            path = page.get("path", "")
            title = page.get("title", "")
            summary = page.get("summary", "")
            page_type = page.get("type", "topic")

            # 跳过用户钉住的页面（已在上面恢复）
            if path in pinned:
                continue

            if not summary or len(summary) < 20:
                page_data = read_page(path)
                if page_data:
                    body = page_data.get("body", "")
                    summary = body[:300].strip() if body else title

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
                        result.get("category", "其他"),
                        result.get("subcategory", ""),
                        description=result.get("description", ""),
                    )
                    success += 1
                else:
                    # LLM 返回 None（调用失败），兜底归入"其他"
                    add_page_to_taxonomy(path, "其他")
            except Exception as e:
                logger.warning(f"Taxonomy 初始化分类失败 {path}: {e}")
                add_page_to_taxonomy(path, "其他")

        logger.info(f"Taxonomy 初始化完成: {success}/{len(pages)} 个页面已分类")

        # 成功后删除备份
        if backup_file.exists():
            backup_file.unlink()

        await maybe_merge_categories()
    finally:
        _reclassify_running = False
