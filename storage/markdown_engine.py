"""Markdown 落库引擎

将 LLM 清洗后的结构化知识写入本地 .md 文件。
- 命名规范: [YYMMDD]_[短UUID]_[精简标题].md
- 格式: YAML Front-matter + Markdown 正文
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import shortuuid
import yaml

from config import DATA_DIR
from transform.llm_cleaner import CleanedKnowledge


class MarkdownEngine:
    """Markdown 落库引擎"""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        knowledge: CleanedKnowledge,
        source_url: str,
        source_platform: str,
        author: str = "",
    ) -> Path:
        """将结构化知识保存为 Markdown 文件

        Args:
            knowledge: LLM 清洗后的知识数据
            source_url: 原始 URL
            source_platform: 来源平台
            author: 作者

        Returns:
            保存的文件路径
        """
        # 生成文件名
        filename = self._generate_filename(knowledge.title)
        filepath = self.data_dir / filename

        # 构建 YAML Front-matter
        frontmatter = {
            "title": knowledge.title,
            "summary": knowledge.summary,
            "tags": knowledge.tags,
            "source_url": source_url,
            "source_platform": source_platform,
            "author": author,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 组装 Markdown 内容
        content = self._build_markdown(frontmatter, knowledge)

        # 写入文件
        filepath.write_text(content, encoding="utf-8")

        return filepath

    def _generate_filename(self, title: str) -> str:
        """生成文件名: [YYMMDD]_[短UUID]_[精简标题].md"""
        date_str = datetime.now().strftime("%y%m%d")
        short_id = shortuuid.uuid()[:8]
        safe_title = self._sanitize_title(title)
        return f"{date_str}_{short_id}_{safe_title}.md"

    def _sanitize_title(self, title: str) -> str:
        """清理标题为安全的文件名部分"""
        # 移除不安全字符
        safe = re.sub(r'[\\/:*?"<>|\n\r\t]', "", title)
        # 替换空格为下划线
        safe = re.sub(r"\s+", "_", safe)
        # 截断
        if len(safe) > 50:
            safe = safe[:50]
        # 去除首尾下划线
        safe = safe.strip("_")
        return safe or "untitled"

    def _build_markdown(self, frontmatter: dict, knowledge: CleanedKnowledge) -> str:
        """构建完整的 Markdown 文件内容"""
        # YAML Front-matter
        yaml_str = yaml.dump(
            frontmatter,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

        # 组装
        parts = [
            "---",
            yaml_str.strip(),
            "---",
            "",
            f"# {knowledge.title}",
            "",
            f"> **摘要**: {knowledge.summary}",
            "",
            knowledge.cleaned_content,
            "",
        ]

        return "\n".join(parts)

    def list_all(self) -> list[dict]:
        """列出所有已落库的知识文件的元数据"""
        results = []
        for md_file in sorted(self.data_dir.glob("*.md"), reverse=True):
            meta = self._read_frontmatter(md_file)
            if meta:
                meta["file_path"] = str(md_file)
                results.append(meta)
        return results

    def _read_frontmatter(self, filepath: Path):
        """读取文件的 YAML Front-matter"""
        try:
            content = filepath.read_text(encoding="utf-8")
            if not content.startswith("---"):
                return None
            parts = content.split("---", 2)
            if len(parts) < 3:
                return None
            return yaml.safe_load(parts[1])
        except Exception:
            return None
