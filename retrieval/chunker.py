"""语义切片器

将 Markdown 文件切成语义相关的文本块，用于向量化索引。

切片策略：
1. 按 ## / ### 标题层级进行逻辑切片
2. 如果文档没有标题层级，按固定 token 数（约 512）切片
3. 每篇文档额外生成一个"摘要切片(Summary Chunk)"用于泛化检索
"""

import re
from dataclasses import dataclass
from pathlib import Path

from utils.frontmatter import parse_frontmatter


@dataclass
class Chunk:
    """文本切片"""
    chunk_id: str         # 唯一标识: filename#section_index
    text: str             # 切片文本
    source_file: str      # 来源文件路径
    title: str            # 文档标题
    section_title: str    # 切片所属章节标题
    chunk_type: str       # "summary" | "section" | "segment"
    metadata: dict        # 额外元数据 (tags, source_url 等)


class SemanticChunker:
    """语义切片器"""

    MAX_CHUNK_CHARS = 1500   # 单切片最大字符数 (约 512 token)
    MIN_CHUNK_CHARS = 50     # 最小有意义切片

    def chunk_file(self, filepath: Path) -> list[Chunk]:
        """将单个 Markdown 文件切片"""
        content = filepath.read_text(encoding="utf-8")

        # 解析 Front-matter
        meta, body = parse_frontmatter(content)
        if not body.strip():
            return []

        doc_title = meta.get("title", filepath.stem)
        filename = filepath.name

        chunks = []

        # 1. 摘要切片 (Summary Chunk) - 永远生成
        summary = meta.get("summary", "")
        tags = meta.get("tags", [])
        summary_text = f"标题: {doc_title}\n摘要: {summary}\n标签: {', '.join(tags)}"
        chunks.append(Chunk(
            chunk_id=f"{filename}#summary",
            text=summary_text,
            source_file=str(filepath),
            title=doc_title,
            section_title="摘要",
            chunk_type="summary",
            metadata=meta,
        ))

        # 2. 按标题层级切片
        sections = self._split_by_headers(body)

        if len(sections) > 1:
            # 文档有标题结构，按章节切片
            for i, (section_title, section_text) in enumerate(sections):
                if len(section_text.strip()) < self.MIN_CHUNK_CHARS:
                    continue

                # 如果章节过长，进一步拆分
                sub_chunks = self._split_long_text(section_text)
                for j, sub_text in enumerate(sub_chunks):
                    chunk_idx = f"{i}" if len(sub_chunks) == 1 else f"{i}_{j}"
                    chunks.append(Chunk(
                        chunk_id=f"{filename}#section_{chunk_idx}",
                        text=f"## {section_title}\n\n{sub_text}" if section_title else sub_text,
                        source_file=str(filepath),
                        title=doc_title,
                        section_title=section_title or f"段落 {i+1}",
                        chunk_type="section",
                        metadata=meta,
                    ))
        else:
            # 文档没有标题结构，按固定长度切片
            plain_text = body.strip()
            segments = self._split_long_text(plain_text)
            for i, seg in enumerate(segments):
                if len(seg.strip()) < self.MIN_CHUNK_CHARS:
                    continue
                chunks.append(Chunk(
                    chunk_id=f"{filename}#segment_{i}",
                    text=seg,
                    source_file=str(filepath),
                    title=doc_title,
                    section_title=f"段落 {i+1}",
                    chunk_type="segment",
                    metadata=meta,
                ))

        return chunks

    def chunk_directory(self, data_dir: Path) -> list[Chunk]:
        """切片整个目录下的所有 Markdown 文件"""
        all_chunks = []
        for md_file in sorted(data_dir.glob("*.md")):
            chunks = self.chunk_file(md_file)
            all_chunks.extend(chunks)
        return all_chunks

    def _split_by_headers(self, text: str) -> list[tuple[str, str]]:
        """按 # / ## / ### 标题拆分文本

        Returns:
            [(section_title, section_text), ...]
        """
        # 匹配 Markdown 标题
        pattern = r"^(#{1,3})\s+(.+)$"
        lines = text.split("\n")

        sections = []
        current_title = ""
        current_lines = []

        for line in lines:
            match = re.match(pattern, line)
            if match:
                # 保存前一个 section
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                current_title = match.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)

        # 保存最后一个 section
        if current_lines:
            sections.append((current_title, "\n".join(current_lines).strip()))

        return sections

    def _split_long_text(self, text: str) -> list[str]:
        """将过长的文本按段落边界拆分"""
        if len(text) <= self.MAX_CHUNK_CHARS:
            return [text]

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > self.MAX_CHUNK_CHARS:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text[:self.MAX_CHUNK_CHARS]]
