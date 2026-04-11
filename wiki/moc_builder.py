"""MOC (Map of Content) 导航页自动生成器

按 Taxonomy 分类自动生成主题导航页，串联 topics + concepts + entities。
零 LLM 调用，纯代码遍历。
"""

import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import yaml

from config import WIKI_DIR
from utils.frontmatter import read_frontmatter as _read_frontmatter

logger = logging.getLogger(__name__)

MOC_DIR = WIKI_DIR / "moc"


def rebuild_moc_pages():
    """读取 taxonomy + 遍历所有页面，为每个分类生成 MOC 导航页

    Returns:
        生成的 MOC 页面数
    """
    MOC_DIR.mkdir(parents=True, exist_ok=True)

    # 读取 taxonomy
    taxonomy_path = WIKI_DIR / "_taxonomy.yaml"
    if not taxonomy_path.exists():
        logger.warning("_taxonomy.yaml 不存在，跳过 MOC 生成")
        return 0

    try:
        tax = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"读取 taxonomy 失败: {e}")
        return 0

    categories = tax.get("categories", {}) if tax else {}
    if not categories:
        return 0

    # 收集所有页面元信息
    page_meta = {}  # path -> meta dict
    for subdir in ["topics", "entities", "concepts"]:
        dir_path = WIKI_DIR / subdir
        if not dir_path.exists():
            continue
        for md_file in sorted(dir_path.glob("*.md")):
            meta = _read_frontmatter(md_file)
            if meta:
                path = f"{subdir}/{md_file.name}"
                page_meta[path] = {
                    "title": meta.get("title", md_file.stem),
                    "summary": (meta.get("summary", "") or "")[:60],
                    "type": meta.get("type", subdir.rstrip("s")),
                    "sources_count": len(meta.get("sources", [])),
                    "related_concepts": meta.get("related_concepts", []),
                }

    # 建立 concept name -> path 映射（用于关联）
    concept_name_to_path = {}
    concepts_dir = WIKI_DIR / "concepts"
    if concepts_dir.exists():
        for md_file in concepts_dir.glob("*.md"):
            meta = _read_frontmatter(md_file)
            if meta:
                name = meta.get("title", md_file.stem)
                concept_name_to_path[name] = f"concepts/{md_file.name}"

    # 为每个分类生成 MOC
    count = 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    for cat_name, cat_data in categories.items():
        if not isinstance(cat_data, dict):
            continue

        # 收集该分类下的所有页面路径
        cat_pages = list(cat_data.get("pages", []))
        for child_data in (cat_data.get("children") or {}).values():
            if isinstance(child_data, dict):
                cat_pages.extend(child_data.get("pages", []))

        if not cat_pages:
            continue

        # 按类型分组
        topics = []
        entities = []
        concepts_in_cat = []

        for p in cat_pages:
            meta = page_meta.get(p)
            if not meta:
                continue
            ptype = meta["type"]
            entry = f"- [[{meta['title']}]] — {meta['summary']}"
            if meta["sources_count"] > 1:
                entry += f" ({meta['sources_count']} 篇来源)"

            if ptype in ("topic", "topics"):
                topics.append(entry)
            elif ptype in ("entity", "entities"):
                entities.append(entry)
            elif ptype in ("concept", "concepts"):
                concepts_in_cat.append(entry)

        # 从 topics 的 related_concepts 收集关联概念
        for p in cat_pages:
            meta = page_meta.get(p)
            if not meta:
                continue
            for rc in meta.get("related_concepts", []):
                if rc in concept_name_to_path:
                    rc_meta = page_meta.get(concept_name_to_path[rc])
                    if rc_meta:
                        entry = f"- [[{rc}]] — {rc_meta['summary']}"
                        if entry not in concepts_in_cat:
                            concepts_in_cat.append(entry)

        # 找关联分类
        related_cats = []
        for other_cat in categories:
            if other_cat != cat_name:
                # 简单关联：共享实体或概念
                related_cats.append(f"- → [[{other_cat}]]")

        # 生成 MOC 文件
        lines = [
            f"---",
            f"type: moc",
            f"title: '{cat_name}'",
            f"generated_at: '{now}'",
            f"page_count: {len(cat_pages)}",
            f"---",
            f"",
            f"# {cat_name}",
            f"",
            f"> 本页面自动生成，汇总「{cat_name}」分类下的所有知识页面。",
            f"",
        ]

        if topics:
            lines.append(f"## 主题页 ({len(topics)})")
            lines.append("")
            lines.extend(topics)
            lines.append("")

        if concepts_in_cat:
            lines.append(f"## 相关概念 ({len(concepts_in_cat)})")
            lines.append("")
            lines.extend(concepts_in_cat)
            lines.append("")

        if entities:
            lines.append(f"## 相关实体 ({len(entities)})")
            lines.append("")
            lines.extend(entities)
            lines.append("")

        if related_cats:
            lines.append("## 关联分类")
            lines.append("")
            lines.extend(related_cats[:5])  # 最多显示5个
            lines.append("")

        content = "\n".join(lines) + "\n"
        moc_path = MOC_DIR / f"{cat_name}.md"
        moc_path.write_text(content, encoding="utf-8")
        count += 1
        logger.info(f"MOC 生成: {cat_name} ({len(cat_pages)} 页面)")

    logger.info(f"MOC 导航页生成完成: {count} 个分类")
    return count
