"""Wiki 索引和日志自动生成器

_index.md 由 Python 脚本自动生成（非 LLM），100% 准确，零 Token 成本。
_log.md 由 Python 代码追加写入（非 LLM）。
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from config import WIKI_DIR
from utils.frontmatter import read_frontmatter as _read_frontmatter


def rebuild_index() -> str:
    """遍历 wiki/ 目录，自动生成 _index.md — 零 LLM 调用"""
    sections = {"topics": [], "entities": [], "insights": []}

    for subdir in ["topics", "entities", "insights"]:
        dir_path = WIKI_DIR / subdir
        if not dir_path.exists():
            continue
        for md_file in sorted(dir_path.glob("*.md")):
            meta = _read_frontmatter(md_file)
            if meta:
                sections[subdir].append({
                    "file": f"{subdir}/{md_file.name}",
                    "title": meta.get("title", md_file.stem),
                    "summary": (meta.get("summary", "") or "")[:50],
                    "sources_count": len(meta.get("sources", [])),
                    "updated_at": meta.get("updated_at", ""),
                })

    lines = [
        "# Wiki 索引\n",
        f"> 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}，请勿手动编辑\n",
    ]

    for section_name, label in [("topics", "主题"), ("entities", "实体"), ("insights", "洞察")]:
        items = sections[section_name]
        lines.append(f"\n## {label} ({len(items)})\n")
        if items:
            lines.append("| 页面 | 摘要 | 来源数 | 更新时间 |")
            lines.append("|------|------|--------|---------|")
            for item in items:
                lines.append(
                    f"| [{item['title']}]({item['file']}) "
                    f"| {item['summary']} "
                    f"| {item['sources_count']} "
                    f"| {item['updated_at']} |"
                )
        else:
            lines.append("_(暂无)_")

    content = "\n".join(lines) + "\n"
    (WIKI_DIR / "_index.md").write_text(content, encoding="utf-8")
    return content


def append_log(action: str, source: str, details: list[str]):
    """追加编译日志 — 纯代码写入，不经过 LLM

    Args:
        action: "INGEST" | "QUERY_BACKFILL" | "INSPECT"
        source: 来源描述（文件名或查询内容）
        details: 操作详情列表，如 ["创建: topics/xxx.md", "更新: topics/yyy.md"]
    """
    log_path = WIKI_DIR / "_log.md"
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    entry = f"- **{time_str} {action}** | 来源: `{source}`\n"
    for detail in details:
        entry += f"  - {detail}\n"
    entry += "\n"

    if log_path.exists():
        content = log_path.read_text(encoding="utf-8")
        date_header = f"## {date_str}"
        if date_header in content:
            content = content.replace(date_header + "\n", date_header + "\n\n" + entry, 1)
        else:
            # 新日期，插入到标题下方
            insert_after = "# Wiki 操作日志\n"
            if insert_after in content:
                content = content.replace(insert_after, insert_after + f"\n{date_header}\n\n{entry}", 1)
            else:
                content += f"\n{date_header}\n\n{entry}"
    else:
        content = f"# Wiki 操作日志\n\n## {date_str}\n\n{entry}"

    log_path.write_text(content, encoding="utf-8")


def build_lightweight_summary() -> str:
    """为 LLM 编译决策生成轻量摘要（替代读取完整 _index.md）

    返回页面名列表 + 一句话描述，供 LLM 编译时了解现有 Wiki 状态。
    """
    lines = []
    for subdir in ["topics", "entities", "insights"]:
        dir_path = WIKI_DIR / subdir
        if not dir_path.exists():
            continue
        for md_file in sorted(dir_path.glob("*.md")):
            meta = _read_frontmatter(md_file)
            title = meta.get("title", md_file.stem) if meta else md_file.stem
            summary = (meta.get("summary", "") or "")[:60] if meta else ""
            lines.append(f"- [{subdir}] {title}: {summary}")

    if not lines:
        return "当前 Wiki 为空，没有任何页面。"

    return "当前 Wiki 已有页面:\n" + "\n".join(lines)

