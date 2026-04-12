"""Wiki 健康检查器 (增强版)

纯 Python 实现，不依赖 LLM。
检查项: 孤立页面、缺失页面、过时页面、来源统计、
         内容质量(空/超短)、标签覆盖率、概念卡质量、交叉引用密度。
"""

import re
from datetime import datetime, timedelta
from pathlib import Path

from config import WIKI_DIR
from wiki.page_store import list_wiki_pages, read_page


def inspect() -> dict:
    """执行 Wiki 健康检查，返回检查报告

    Returns:
        {
            "orphan_pages": [...],       # 零入链的页面
            "missing_pages": [...],      # 被引用但不存在的页面
            "stale_pages": [...],        # 超过 30 天未更新且来源 ≤ 1
            "low_quality_pages": [...],  # 内容过短的页面
            "unlabeled_pages": [...],    # 无标签页面
            "thin_concepts": [...],      # 超短概念卡
            "source_stats": [...],       # 每个页面的来源数统计
            "cross_ref_density": float,  # 交叉引用密度
            "tag_coverage": float,       # 标签覆盖率
            "total_pages": int,
            "summary": str,
        }
    """
    pages = list_wiki_pages()
    total = len(pages)

    # 收集所有 [[]] 引用关系
    all_refs = set()      # 被引用的标题集合
    page_titles = {}      # path -> title
    page_incoming = {}    # title -> 入链数

    for page_info in pages:
        path = page_info.get("path", "")
        title = page_info.get("title", "")
        page_titles[path] = title
        page_incoming[title] = 0

    # 扫描所有页面中的 [[]] 引用
    total_refs = 0
    for page_info in pages:
        path = page_info.get("path", "")
        page_data = read_page(path)
        if not page_data:
            continue

        content = page_data.get("full_content", "")
        refs = re.findall(r"\[\[(.+?)\]\]", content)
        total_refs += len(refs)
        for ref in refs:
            all_refs.add(ref)
            if ref in page_incoming:
                page_incoming[ref] += 1

    # 1. 孤立页面（零入链）
    orphan_pages = []
    for title, count in page_incoming.items():
        if count == 0:
            orphan_pages.append(title)

    # 2. 缺失页面（被引用但不存在）
    existing_titles = set(page_titles.values())
    missing_pages = [ref for ref in all_refs if ref not in existing_titles]

    # 3. 过时页面（30天未更新 + 来源 ≤ 1）
    stale_pages = []
    cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    for page_info in pages:
        updated = page_info.get("updated_at", "")
        sources = page_info.get("sources", [])
        if updated and updated < cutoff and len(sources) <= 1:
            stale_pages.append({
                "path": page_info.get("path", ""),
                "title": page_info.get("title", ""),
                "updated_at": updated,
                "sources_count": len(sources),
            })

    # 4. 内容质量：过短页面（正文 < 200 字）
    low_quality_pages = []
    for page_info in pages:
        path = page_info.get("path", "")
        page_data = read_page(path)
        if page_data:
            body = page_data.get("body", "")
            if len(body.strip()) < 200:
                low_quality_pages.append({
                    "path": path,
                    "title": page_info.get("title", ""),
                    "body_length": len(body.strip()),
                })

    # 5. 标签覆盖率
    tagged_count = sum(1 for p in pages if p.get("tags"))
    tag_coverage = round(tagged_count / max(total, 1), 2)
    unlabeled_pages = [p.get("title", "") for p in pages if not p.get("tags")]

    # 6. 超短概念卡
    thin_concepts = []
    for page_info in pages:
        path = page_info.get("path", "")
        if "concepts/" in path:
            page_data = read_page(path)
            if page_data:
                body = page_data.get("body", "")
                if len(body.strip()) < 300:
                    thin_concepts.append({
                        "path": path,
                        "title": page_info.get("title", ""),
                        "body_length": len(body.strip()),
                    })

    # 7. 来源统计
    source_stats = []
    for page_info in pages:
        source_stats.append({
            "path": page_info.get("path", ""),
            "title": page_info.get("title", ""),
            "sources_count": len(page_info.get("sources", [])),
        })
    source_stats.sort(key=lambda x: -x["sources_count"])

    # 8. 交叉引用密度（平均每页的 [[]] 引用数）
    cross_ref_density = round(total_refs / max(total, 1), 2)

    # 生成摘要
    issues = []
    if orphan_pages:
        issues.append(f"{len(orphan_pages)} 个孤立页面")
    if missing_pages:
        issues.append(f"{len(missing_pages)} 个缺失引用")
    if stale_pages:
        issues.append(f"{len(stale_pages)} 个过时页面")
    if low_quality_pages:
        issues.append(f"{len(low_quality_pages)} 个超短页面")
    if thin_concepts:
        issues.append(f"{len(thin_concepts)} 个超短概念卡")
    if unlabeled_pages:
        issues.append(f"{len(unlabeled_pages)} 个无标签页面")

    summary = f"Wiki 共 {total} 个页面。引用密度 {cross_ref_density}，标签覆盖率 {tag_coverage*100:.0f}%。" + (
        "发现问题: " + "、".join(issues) if issues else "状态良好，无问题。"
    )

    return {
        "orphan_pages": orphan_pages,
        "missing_pages": missing_pages,
        "stale_pages": stale_pages,
        "low_quality_pages": low_quality_pages,
        "unlabeled_pages": unlabeled_pages,
        "thin_concepts": thin_concepts,
        "source_stats": source_stats,
        "cross_ref_density": cross_ref_density,
        "tag_coverage": tag_coverage,
        "total_pages": total,
        "summary": summary,
    }
