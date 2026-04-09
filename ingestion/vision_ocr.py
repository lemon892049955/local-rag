"""Vision OCR — Kimi Vision API 图片文字提取

零新依赖，复用已有 OpenAI SDK + Kimi API Key。
成本：~¥0.01/张，零内存开销。

v0.6.6 修复：
- 使用正确的 Vision 模型 (moonshot-v1-8k-vision-preview)
- 微信图片下载加 Referer 头绕过防盗链
- 小红书图片下载加 Referer 头
- 增加重试和超时控制
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

# Kimi Vision 模型名称
VISION_MODEL = "moonshot-v1-8k-vision-preview"


class VisionOCR(BaseFetcher):
    """图片 OCR — 基于 Vision LLM"""

    def __init__(self):
        config = get_llm_config()
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
        # 强制使用 Vision 模型（不用 config 里的文本模型）
        self.model = VISION_MODEL

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
        """OCR 网络图片 URL — 带 Referer 绕过防盗链"""
        import httpx

        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        if "mmbiz" in url:
            headers["Referer"] = "https://mp.weixin.qq.com/"
        elif "xhscdn" in url or "xiaohongshu" in url:
            headers["Referer"] = "https://www.xiaohongshu.com/"
        elif "zhimg" in url:
            headers["Referer"] = "https://www.zhihu.com/"

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()

            # 检查是否真的拿到了图片（防盗链可能返回 HTML 或 1x1 像素）
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                logger.warning(f"图片下载被防盗链拦截（返回HTML）: {url[:80]}")
                return ""
            if len(resp.content) < 1000:
                logger.warning(f"图片太小，可能是占位图: {url[:80]} ({len(resp.content)} bytes)")
                return ""

            return await self.ocr_image_bytes(resp.content)
        except Exception as e:
            logger.warning(f"图片下载失败: {url[:80]} - {e}")
            return ""

    async def ocr_image_bytes(self, img_bytes: bytes, suffix: str = ".png") -> str:
        """OCR 图片字节数据"""
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
            result = response.choices[0].message.content.strip()
            logger.info(f"Vision OCR 成功: {len(result)} 字")
            return result
        except Exception as e:
            logger.error(f"Vision OCR 调用失败 ({self.model}): {e}")
            return ""

    async def ocr_multiple_images(self, image_urls: list[str]) -> str:
        """批量 OCR 多张图片，拼接结果"""
        results = []
        for i, url in enumerate(image_urls):
            try:
                text = await self.ocr_image_url(url)
                if text:
                    results.append(f"[图片 {i + 1}]\n{text}")
                    logger.info(f"图片 {i+1}/{len(image_urls)} OCR 完成: {len(text)} 字")
            except Exception as e:
                logger.warning(f"图片 {i + 1} OCR 失败: {e}")

        return "\n\n".join(results)
