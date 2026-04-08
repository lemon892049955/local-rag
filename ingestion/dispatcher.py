"""统一路由分发器 — 支持 URL + 文件

扩展原有的 FetcherRouter，增加文件类型分发能力。
"""

import logging
from pathlib import Path

from .base import BaseFetcher, RawContent, FetchError
from .router import FetcherRouter
from utils.url_utils import detect_source

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
PDF_EXTENSIONS = {".pdf"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm", ".mp4"}


class Dispatcher:
    """统一路由分发 — 支持 URL 和文件"""

    def __init__(self):
        self._url_router = FetcherRouter()
        self._pdf_parser = None
        self._vision_ocr = None
        self._audio_transcriber = None

    @property
    def pdf_parser(self):
        if self._pdf_parser is None:
            from .pdf_parser import PDFParser
            self._pdf_parser = PDFParser()
        return self._pdf_parser

    @property
    def vision_ocr(self):
        if self._vision_ocr is None:
            from .vision_ocr import VisionOCR
            self._vision_ocr = VisionOCR()
        return self._vision_ocr

    @property
    def audio_transcriber(self):
        if self._audio_transcriber is None:
            from .audio_transcriber import AudioTranscriber
            self._audio_transcriber = AudioTranscriber()
        return self._audio_transcriber

    def detect_type(self, input_path: str) -> str:
        """判断输入类型

        Returns:
            "url" | "pdf" | "image" | "audio" | "unknown"
        """
        # URL 判断
        if input_path.startswith("http://") or input_path.startswith("https://"):
            return "url"

        # 文件判断
        path = Path(input_path)
        suffix = path.suffix.lower()

        if suffix in PDF_EXTENSIONS:
            return "pdf"
        elif suffix in IMAGE_EXTENSIONS:
            return "image"
        elif suffix in AUDIO_EXTENSIONS:
            return "audio"
        else:
            return "unknown"

    async def dispatch(self, input_path: str) -> RawContent:
        """统一分发：URL 走网页抓取，文件走对应解析器

        Args:
            input_path: URL 或本地文件路径

        Returns:
            RawContent
        """
        input_type = self.detect_type(input_path)
        logger.info(f"Dispatcher: {input_path[:80]} → {input_type}")

        if input_type == "url":
            return await self._url_router.fetch(input_path)
        elif input_type == "pdf":
            return await self.pdf_parser.parse_file(Path(input_path))
        elif input_type == "image":
            return await self.vision_ocr.fetch(input_path)
        elif input_type == "audio":
            return await self.audio_transcriber.transcribe_file(Path(input_path))
        else:
            raise FetchError(input_path, f"不支持的输入类型: {Path(input_path).suffix}")
