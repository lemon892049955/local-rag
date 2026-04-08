"""PDF 解析器 — pymupdf 文本提取

轻量方案：pymupdf 直接提取文本型 PDF（~15MB 包，纯 CPU）。
扫描型 PDF 降级到 Vision OCR。
转录完成后删除原始文件节省磁盘。
"""

import logging
from pathlib import Path

from .base import BaseFetcher, RawContent, FetchError

logger = logging.getLogger(__name__)


class PDFParser(BaseFetcher):
    """PDF 文本提取器"""

    MIN_TEXT_LENGTH = 100  # 低于此长度视为扫描件，降级 OCR

    async def fetch(self, url: str) -> RawContent:
        """url 参数在此为本地文件路径"""
        return await self.parse_file(Path(url))

    async def parse_file(self, filepath: Path) -> RawContent:
        """解析 PDF 文件

        Args:
            filepath: PDF 文件路径

        Returns:
            RawContent
        """
        if not filepath.exists():
            raise FetchError(str(filepath), "文件不存在")

        try:
            import fitz  # pymupdf
        except ImportError:
            raise FetchError(str(filepath), "pymupdf 未安装，请 pip install pymupdf")

        try:
            doc = fitz.open(str(filepath))
        except Exception as e:
            raise FetchError(str(filepath), f"PDF 打开失败: {e}")

        # 提取文本
        title = self._extract_title(doc, filepath)
        text_pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                text_pages.append(text)

        doc.close()

        full_text = "\n\n".join(text_pages)

        # 如果文本过短，可能是扫描件 → 降级 Vision OCR
        if len(full_text) < self.MIN_TEXT_LENGTH:
            logger.info(f"PDF 文本过短 ({len(full_text)} 字)，尝试 Vision OCR 降级")
            full_text = await self._ocr_fallback(filepath)
            if not full_text:
                raise FetchError(str(filepath), "PDF 文本提取失败，且 OCR 降级也无法提取内容")

        page_count = len(text_pages) or 0
        logger.info(f"PDF 解析完成: {filepath.name}, {page_count} 页, {len(full_text)} 字")

        return RawContent(
            url=str(filepath),
            title=title,
            content=full_text,
            author="",
            source_platform="pdf",
        )

    def _extract_title(self, doc, filepath: Path) -> str:
        """尝试从 PDF 元数据提取标题"""
        metadata = doc.metadata
        if metadata and metadata.get("title"):
            return metadata["title"].strip()

        # 用文件名作为标题
        return filepath.stem.replace("_", " ").replace("-", " ")

    async def _ocr_fallback(self, filepath: Path) -> str:
        """扫描件降级：每页转图片 → Vision OCR"""
        try:
            import fitz
            from .vision_ocr import VisionOCR

            ocr = VisionOCR()
            doc = fitz.open(str(filepath))
            all_text = []

            # 最多处理前 20 页，避免成本过高
            max_pages = min(len(doc), 20)
            for page_num in range(max_pages):
                page = doc[page_num]
                # 页面渲染为图片
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")

                text = await ocr.ocr_image_bytes(img_bytes)
                if text:
                    all_text.append(f"[第 {page_num + 1} 页]\n{text}")

            doc.close()
            return "\n\n".join(all_text)

        except Exception as e:
            logger.error(f"OCR 降级失败: {e}")
            return ""
