"""PDF 解析器 — pymupdf 文本提取 + Vision OCR 表格识别

v0.9 核心改进：
- 检测表格页 → Vision OCR 智能识别 → 输出 Markdown 表格
- 改进目录页检测（"目录" + 章节...页码 格式）
- 清洗页眉（XX大学硕士学位论文）
- 保留表格结构，提升可读性
"""

import re
import logging
from pathlib import Path
from typing import Optional, List, Tuple

from .base import BaseFetcher, RawContent, FetchError

logger = logging.getLogger(__name__)


class PDFParser(BaseFetcher):
    """PDF 文本提取器 — 支持表格识别和学术论文清洗"""

    MIN_TEXT_LENGTH = 100

    # 学术论文封面关键词
    COVER_KEYWORDS = [
        "硕士学位论文", "博士学位论文", "学士论文", "学位论文",
        "研究生学位论文", "本科毕业论文",
        "作者姓名", "指导教师", "学科专业", "学位类型",
        "培养单位", "所在学院", "研究方向",
        "提交日期", "答辩日期", "申请学位",
    ]

    # 声明页关键词
    DECLARATION_KEYWORDS = [
        "原创性声明", "版权使用授权书", "版权声明",
        "本人郑重声明", "学位论文作者签名",
        "不保密", "保密□", "解密后适用",
    ]

    # 页眉模式（学校名+学位论文）
    HEADER_PATTERNS = [
        r"^.+大学硕士学位论文\s*$",
        r"^.+大学博士学位论文\s*$",
        r"^.+学院硕士学位论文\s*$",
        r"^浙江工业大学硕士学位论文\s*$",
        r"^.+大学学士论文\s*$",
    ]

    # 水印关键词
    WATERMARK_KEYWORDS = ["万方数据", "知网", "CNKI", "中国知网"]

    def __init__(self):
        self._is_thesis = False
        self._detected_title = None
        self._vision_ocr = None

    @property
    def vision_ocr(self):
        """延迟加载 Vision OCR"""
        if self._vision_ocr is None:
            from .vision_ocr import VisionOCR
            self._vision_ocr = VisionOCR()
        return self._vision_ocr

    async def fetch(self, url: str) -> RawContent:
        return await self.parse_file(Path(url))

    async def parse_file(self, filepath: Path) -> RawContent:
        if not filepath.exists():
            raise FetchError(str(filepath), "文件不存在")

        try:
            import fitz
        except ImportError:
            raise FetchError(str(filepath), "pymupdf 未安装")

        try:
            doc = fitz.open(str(filepath))
        except Exception as e:
            raise FetchError(str(filepath), f"PDF 打开失败: {e}")

        total_pages = len(doc)
        self._is_thesis = self._detect_thesis(doc)
        logger.info(f"PDF 类型: {'学术论文' if self._is_thesis else '普通文档'} ({filepath.name})")

        title = self._extract_title(doc, filepath)

        # 逐页处理
        text_pages = []
        toc_pages = []  # 记录目录页范围

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text").strip()

            if not text:
                continue

            # 学术论文：检测页面类型
            if self._is_thesis and page_num < 20:
                page_type = self._classify_page(text, page_num)

                if page_type == "cover":
                    logger.debug(f"跳过封面页: 第 {page_num + 1} 页")
                    continue
                elif page_type == "declaration":
                    logger.debug(f"跳过声明页: 第 {page_num + 1} 页")
                    continue
                elif page_type == "toc":
                    # 目录页：记录但不立即跳过，需要检测目录结束
                    toc_pages.append(page_num)
                    logger.debug(f"检测到目录页: 第 {page_num + 1} 页")
                    continue
                elif page_type == "toc_end":
                    # 目录结束标记
                    continue

            # 检测是否是表格页
            has_table = self._detect_table(text)
            if has_table:
                logger.debug(f"检测到表格: 第 {page_num + 1} 页，使用 Vision OCR")
                table_text = await self._extract_table_with_ocr(page)
                if table_text:
                    text_pages.append(table_text)
                    continue

            # 普通文本页：清洗后保留
            text = self._clean_page(text)
            if text.strip():
                text_pages.append(text)

        doc.close()

        full_text = "\n\n".join(text_pages)

        # 文本过短 → 全页 Vision OCR
        if len(full_text) < self.MIN_TEXT_LENGTH:
            logger.info(f"PDF 文本过短 ({len(full_text)} 字)，Vision OCR 全页处理")
            full_text = await self._ocr_all_pages(filepath)
            if not full_text:
                raise FetchError(str(filepath), "PDF 提取失败")

        full_text = self._post_process(full_text)
        logger.info(f"PDF 解析完成: {filepath.name}, {len(text_pages)} 页, {len(full_text)} 字")

        return RawContent(
            url=str(filepath),
            title=title,
            content=full_text,
            author="",
            source_platform="pdf",
        )

    def _detect_table(self, text: str) -> bool:
        """检测页面是否包含表格

        表格特征：
        - 多行包含数字和括号 (如 "1.46 (0.30)")
        - 表格标题 (如 "表4-1", "Table 1", "Table 1.")
        - 多行对齐的数字列
        """
        # 表格标题 (支持 Table 1., Table 4-1, 表4-1 等格式)
        if re.search(r"表\s*\d+[-\.\s]|Table\s*\d+[-\.\s]", text, re.IGNORECASE):
            # 检查是否有数据行（数字+括号）
            data_rows = len(re.findall(r"\d+\.\d+\s*\(\d+\.\d+\)", text))
            if data_rows >= 3:
                return True

        # 检测多列对齐的数字（表格特征）
        lines = text.split("\n")
        numeric_lines = 0
        for line in lines:
            # 一行有多个数字且分布均匀，可能是表格
            numbers = re.findall(r"\d+\.\d+|\d+", line)
            if len(numbers) >= 3:
                numeric_lines += 1

        if numeric_lines >= 8:
            return True

        return False

    async def _extract_table_with_ocr(self, page) -> str:
        """用 Vision OCR 提取表格，输出 Markdown 格式"""
        try:
            import fitz
            pix = page.get_pixmap(dpi=150)  # 适中 DPI，避免图片过大
            img_bytes = pix.tobytes("png")

            # 直接使用表格专用 prompt，不依赖自动分类
            from .vision_ocr import PROMPT_TABLE
            text = await self.vision_ocr._call_vision(
                self.vision_ocr._bytes_to_data_url(img_bytes),
                PROMPT_TABLE
            )
            return text or ""
        except Exception as e:
            logger.warning(f"表格 OCR 失败: {e}")
            return ""

    def _detect_thesis(self, doc) -> bool:
        check_pages = min(5, len(doc))
        thesis_score = 0

        for i in range(check_pages):
            text = doc[i].get_text("text")
            if not text:
                continue

            for kw in self.COVER_KEYWORDS:
                if kw in text:
                    thesis_score += 1

            if re.search(r"(硕士|博士|学士)(学位)?论文", text):
                thesis_score += 2
            if any(wm in text for wm in self.WATERMARK_KEYWORDS):
                thesis_score += 1

        return thesis_score >= 3

    def _extract_title(self, doc, filepath: Path) -> str:
        metadata = doc.metadata
        if metadata and metadata.get("title"):
            title = metadata["title"].strip()
            if 5 < len(title) < 100:
                return title

        if self._is_thesis:
            title = self._extract_thesis_title(doc)
            if title:
                return title

        return self._clean_filename_title(filepath.stem)

    def _extract_thesis_title(self, doc) -> Optional[str]:
        for i in range(min(3, len(doc))):
            text = doc[i].get_text("text")
            if not text:
                continue

            lines = [l.strip() for l in text.split("\n") if l.strip()]

            # 找"论文题目"后面的内容
            for j, line in enumerate(lines):
                if "论文题目" in line or "题    目" in line:
                    if j + 1 < len(lines):
                        next_line = lines[j + 1]
                        if 5 < len(next_line) < 80:
                            return next_line
                    match = re.search(r"[：:]\s*(.+)$", line)
                    if match:
                        title = match.group(1).strip()
                        if len(title) > 5:
                            return title

            # 找最长的中文行
            chinese_lines = []
            for line in lines:
                if any(kw in line for kw in ["硕士", "博士", "学位", "作者", "导师", "专业", "学院"]):
                    continue
                if len(line) < 10 or len(line) > 80:
                    continue
                chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", line))
                if chinese_chars / len(line) > 0.6:
                    chinese_lines.append((line, chinese_chars))

            if chinese_lines:
                chinese_lines.sort(key=lambda x: -x[1])
                return chinese_lines[0][0]

        return None

    def _classify_page(self, text: str, page_num: int) -> str:
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        # 封面页
        cover_hits = sum(1 for kw in self.COVER_KEYWORDS if kw in text)
        if cover_hits >= 3:
            return "cover"

        # 声明页
        decl_hits = sum(1 for kw in self.DECLARATION_KEYWORDS if kw in text)
        if decl_hits >= 2:
            return "declaration"

        # 目录页：检测 "目录" 标题 + 后面多行是 "章节名 ... 页码" 格式
        toc_title_found = False
        toc_entry_count = 0

        for i, line in enumerate(lines[:20]):
            # 目录标题
            if line in ("目录", "目 录", "目    录"):
                toc_title_found = True
                continue

            # 目录条目：章节名 + 点号/空格 + 页码
            # 格式: "1.1 研究背景 ... 5" 或 "第一章 绪论 ... 1"
            if re.match(r"^(第[一二三四五六七八九十]+章|\d+\.\d+|.+)\s*\.+\s*\d+\s*$", line):
                toc_entry_count += 1
            # 或者 "摘 要 ... I" 这种
            if re.match(r"^.+\s*\.+\s*[IVXivx\d]+\s*$", line):
                toc_entry_count += 1

        if toc_title_found and toc_entry_count >= 5:
            return "toc"
        # 即使没有标题，如果大量行符合目录格式
        if toc_entry_count >= 10:
            return "toc"

        # 正文开始
        if any(kw in text[:500] for kw in ["摘要", "引言", "绪论", "Abstract", "第一章", "1 "]):
            return "content_start"

        return "content"

    def _clean_page(self, text: str) -> str:
        """清洗单页文本"""
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            # 跳过空行
            if not stripped:
                cleaned_lines.append("")
                continue

            # 跳过页眉（学校名+学位论文）
            should_skip = False
            for pattern in self.HEADER_PATTERNS:
                if re.match(pattern, stripped):
                    should_skip = True
                    break

            # 跳过水印
            if any(wm in stripped for wm in self.WATERMARK_KEYWORDS):
                should_skip = True

            # 跳过纯数字页码
            if stripped.isdigit() and len(stripped) <= 3:
                should_skip = True

            # 跳过罗马数字页码
            if re.match(r"^[IVXivx]+$", stripped) and len(stripped) <= 5:
                should_skip = True

            if should_skip:
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _clean_filename_title(self, filename: str) -> str:
        filename = re.sub(r"^\d+_\d+_\d+_\d+_\w+_", "", filename)
        filename = re.sub(r"\.(pdf|PDF)$", "", filename)
        filename = filename.replace("_", " ").replace("-", " ")
        return filename.strip()

    def _post_process(self, text: str) -> str:
        # 清理连续空行
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    async def _ocr_all_pages(self, filepath: Path) -> str:
        """全页 Vision OCR（扫描件或文本提取失败时）"""
        try:
            import fitz
            doc = fitz.open(str(filepath))
            all_text = []

            max_pages = min(len(doc), 30)  # 最多 30 页
            for page_num in range(max_pages):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")

                text = await self.vision_ocr.ocr_image_bytes(img_bytes)
                if text:
                    all_text.append(f"[第 {page_num + 1} 页]\n{text}")

            doc.close()
            return "\n\n".join(all_text)
        except Exception as e:
            logger.error(f"OCR 失败: {e}")
            return ""
