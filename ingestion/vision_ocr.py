"""Vision OCR — Kimi Vision API 图片文字提取

零新依赖，复用已有 OpenAI SDK + Kimi API Key。
成本：~¥0.01/张，零内存开销。
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import get_llm_config
from .base import BaseFetcher, RawContent, FetchError

logger = logging.getLogger(__name__)

OCR_PROMPT = """请提取这张图片中的所有文字内容，保留原始结构和排版。
如果图片是信息图/PPT/长图，请按顺序提取所有文字。
如果图片中没有文字，请描述图片的主要内容。
输出纯文本，不要添加任何额外解释。"""


class VisionOCR(BaseFetcher):
    """图片 OCR — 基于 Vision LLM"""

    def __init__(self):
        config = get_llm_config()
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
        # Kimi Vision 模型
        self.model = config.get("model", "moonshot-v1-8k")

    async def fetch(self, url: str) -> RawContent:
        """url 参数为图片文件路径或 URL"""
        filepath = Path(url)
        if filepath.exists():
            text = await self.ocr_image_file(filepath)
        else:
            text = await self.ocr_image_url(url)

        if not text or len(text.strip()) < 10:
            raise FetchError(url, "图片 OCR 未提取到有效文字")

        return RawContent(
            url=url,
            title="图片内容",
            content=text,
            author="",
            source_platform="image",
        )

    async def ocr_image_file(self, filepath: Path) -> str:
        """OCR 本地图片文件"""
        img_bytes = filepath.read_bytes()
        return await self.ocr_image_bytes(img_bytes, filepath.suffix)

    async def ocr_image_url(self, url: str) -> str:
        """OCR 网络图片 URL"""
        import requests
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return await self.ocr_image_bytes(resp.content)
        except Exception as e:
            raise FetchError(url, f"图片下载失败: {e}")

    async def ocr_image_bytes(self, img_bytes: bytes, suffix: str = ".png") -> str:
        """OCR 图片字节数据

        Args:
            img_bytes: 图片的二进制数据
            suffix: 文件扩展名（用于确定 MIME type）

        Returns:
            提取的文字内容
        """
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".webp": "image/webp", ".gif": "image/gif"}
        mime = mime_map.get(suffix.lower(), "image/png")

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                temperature=0.1,
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Vision OCR 调用失败: {e}")
            return ""

    async def ocr_multiple_images(self, image_urls: list[str]) -> str:
        """批量 OCR 多张图片，拼接结果"""
        results = []
        for i, url in enumerate(image_urls):
            try:
                text = await self.ocr_image_url(url)
                if text:
                    results.append(f"[图片 {i + 1}]\n{text}")
            except Exception as e:
                logger.warning(f"图片 {i + 1} OCR 失败: {e}")

        return "\n\n".join(results)
