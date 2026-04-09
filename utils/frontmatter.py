"""统一的 YAML Front-matter 解析工具

消除 chunker.py / compiler.py / page_store.py / index_builder.py / main.py 中的重复实现。
"""

import yaml
from pathlib import Path
from typing import Optional


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """解析 YAML Front-matter，返回 (meta, body)"""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, parts[2].strip()


def read_frontmatter(filepath: Path) -> Optional[dict]:
    """读取文件的 YAML Front-matter，返回 meta dict 或 None"""
    try:
        content = filepath.read_text(encoding="utf-8")
        meta, _ = parse_frontmatter(content)
        return meta if meta else None
    except Exception:
        return None
