"""Wiki 页面读写工具

提供 Wiki 页面的创建、追加更新、列举等操作。
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from config import WIKI_DIR
from utils.frontmatter import parse_frontmatter as _parse_frontmatter


def read_page(page_path: str) -> Optional[dict]:
    """读取 Wiki 页面，返回 {meta, body, full_content}"""
    filepath = WIKI_DIR / page_path
    if not filepath.exists():
        return None

    content = filepath.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(content)
    return {"meta": meta, "body": body, "full_content": content, "path": page_path}


def create_page(page_path: str, content: str):
    """创建新 Wiki 页面"""
    filepath = WIKI_DIR / page_path
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")


def append_insight(page_path: str, date: str, source_name: str, insight_content: str):
    """向已有页面的 '## 新增洞察' 区域追加内容 (Append-only)

    Args:
        page_path: Wiki 页面相对路径，如 "topics/vibe-coding.md"
        date: 日期字符串 "YYYY-MM-DD"
        source_name: 来源文件名
        insight_content: 要追加的内容文本
    """
    filepath = WIKI_DIR / page_path
    if not filepath.exists():
        return

    content = filepath.read_text(encoding="utf-8")

    # 构建追加块
    append_block = f"\n### {date} | 来源: {source_name}\n\n{insight_content}\n\n[来源: {source_name}]\n"

    # 在 "## 新增洞察" 标记后追加
    marker = "## 新增洞察"
    if marker in content:
        # 替换占位文本
        content = content.replace("_(后续更新在此追加)_", "")
        # 找到 marker 位置，在其后追加
        idx = content.index(marker) + len(marker)
        # 跳过 marker 同行剩余内容
        newline_idx = content.find("\n", idx)
        if newline_idx == -1:
            newline_idx = len(content)
        content = content[:newline_idx] + "\n" + append_block + content[newline_idx:]
    else:
        # 没有 marker，追加到文件末尾
        content += f"\n{marker}\n{append_block}"

    # 更新 front-matter 中的 updated_at
    content = _update_frontmatter_field(content, "updated_at", f"'{date}'")

    filepath.write_text(content, encoding="utf-8")


def append_cross_reference(page_path: str, ref_title: str):
    """向页面的 '## 相关概念' 区域追加交叉引用链接"""
    filepath = WIKI_DIR / page_path
    if not filepath.exists():
        return

    content = filepath.read_text(encoding="utf-8")
    ref_line = f"- [[{ref_title}]]"

    # 检查是否已存在
    if ref_line in content:
        return

    marker = "## 相关概念"
    if marker in content:
        idx = content.index(marker) + len(marker)
        newline_idx = content.find("\n", idx)
        if newline_idx == -1:
            newline_idx = len(content)
        content = content[:newline_idx] + f"\n{ref_line}" + content[newline_idx:]
    else:
        content += f"\n{marker}\n\n{ref_line}\n"

    filepath.write_text(content, encoding="utf-8")


def add_source_to_frontmatter(page_path: str, source_filename: str):
    """向页面 front-matter 的 sources 列表追加新来源"""
    filepath = WIKI_DIR / page_path
    if not filepath.exists():
        return

    content = filepath.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(content)
    if not meta:
        return

    sources = meta.get("sources", [])
    if source_filename not in sources:
        sources.append(source_filename)
        meta["sources"] = sources
        meta["updated_at"] = datetime.now().strftime("%Y-%m-%d")

        # 重建文件
        yaml_str = yaml.dump(meta, default_flow_style=False, allow_unicode=True, sort_keys=False)
        new_content = f"---\n{yaml_str.strip()}\n---\n{body}"
        filepath.write_text(new_content, encoding="utf-8")


def list_wiki_pages() -> list[dict]:
    """列举所有 Wiki 页面的元数据"""
    results = []
    for subdir in ["topics", "entities", "insights", "concepts"]:
        dir_path = WIKI_DIR / subdir
        if not dir_path.exists():
            continue
        for md_file in sorted(dir_path.glob("*.md")):
            meta, _ = _parse_frontmatter(md_file.read_text(encoding="utf-8"))
            if meta:
                results.append({
                    "path": f"{subdir}/{md_file.name}",
                    "type": subdir.rstrip("s"),  # topic, entity, insight
                    **meta,
                })
    return results


def _update_frontmatter_field(content: str, field: str, value: str) -> str:
    """更新 front-matter 中的单个字段"""
    pattern = rf"^({field}:\s*).*$"
    replacement = rf"\g<1>{value}"
    return re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)
