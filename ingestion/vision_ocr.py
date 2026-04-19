"""Vision OCR — 智能图片内容识别

v0.7 全面增强：
- 图片分类：先判断类型（文字/图表/表格/代码/照片），再用专用 prompt
- 图表理解：柱状图/折线图/饼图等输出结构化数据描述，而非仅提取标签文字
- 表格识别：检测表格后输出 Markdown 表格格式
- 多图关联：支持把同一篇文章的多张图一起发给 Vision 模型，理解图间叙事
- max_images 从 5 提升到 10

零新依赖，复用已有 OpenAI SDK + Kimi API Key。
成本：~¥0.01/张，零内存开销。
"""

import base64
import logging
from pathlib import Path
from typing import Optional, List

from openai import OpenAI

from config import get_vision_config
from .base import BaseFetcher, RawContent, FetchError, ErrorType

logger = logging.getLogger(__name__)

# Kimi Vision 模型名称
VISION_MODEL = "moonshot-v1-8k-vision-preview"

# ===== 分类 Prompt =====
CLASSIFY_PROMPT = """请判断这张图片的类型，只输出一个类型标签（不要其他文字）：
- text: 含大量文字的图片（PPT/长图/截图/信息图/海报）
- chart: 数据图表（柱状图/折线图/饼图/雷达图/热力图/散点图等）
- table: 表格（含行列结构的数据表）
- code: 代码截图
- photo: 照片/插图/实物图（无文字或文字很少）"""

# ===== 专用 OCR Prompts =====
PROMPT_TEXT = """请提取这张图片中的所有文字内容。

规则：
1. 保留原始结构和排版层级（标题用 ## / ###，列表用 - ）
2. 如果是 PPT/长图，按视觉阅读顺序从上到下提取
3. 如果有分栏布局，先左后右
4. 保留重点标记（加粗、高亮等用 **粗体** 表示）
5. 输出纯 Markdown 文本，不要添加额外解释"""

PROMPT_CHART = """请分析这张数据图表，输出结构化描述。

规则：
1. 先说明图表类型（柱状图/折线图/饼图等）和主题
2. 提取所有数据标签、数值和单位
3. 描述数据的核心趋势和关键发现（上升/下降/对比/占比等）
4. 如果有标题、图例、坐标轴标签，全部提取
5. 用 Markdown 格式输出，数据用表格或列表呈现

输出格式：
## [图表标题]
**图表类型**: xxx
**数据摘要**:
| 维度 | 数值 |
|------|------|
| ... | ... |
**趋势分析**: [一句话总结核心发现]"""

PROMPT_TABLE = """请提取这张图片中的表格内容，输出为 Markdown 表格格式。

规则：
1. 保留完整的行列结构，不要遗漏任何单元格
2. 表头用 | --- | 分隔
3. 合并单元格拆开为重复内容
4. 如果表格有标题（如"表4-1 xxx"），用 ## 标注
5. 数字保持原始精度，保留小数位
6. 括号内的数值（如标准差）保留在原位置
7. 表格外的说明文字也要提取，放在表格下方
8. 多级表头用空单元格或合并方式处理
9. 如果表格跨页或被截断，只提取可见部分

输出示例：
## 表4-1 反应时间的均值（标准差）

| 变量 | 人格相似 | 人格不相似 |
|------|----------|------------|
| 注意转移时间(s) | 1.46 (0.30) | 1.61 (0.50) |
| 接管时间(s) | 3.08 (0.68) | 3.03 (0.65) |

注意：只输出 Markdown 表格，不要添加额外解释。"""

PROMPT_CODE = """请提取这张代码截图中的完整代码。

规则：
1. 完整还原代码，保持缩进和空行
2. 用 ```语言名 和 ``` 包裹代码块
3. 如果能识别编程语言，标注语言名
4. 行号不需要保留
5. 如果有注释，保留注释
6. 代码外的 IDE 界面元素忽略"""

PROMPT_PHOTO = """请描述这张图片的内容，重点关注与知识/信息相关的部分。

规则：
1. 描述图片的主题和核心内容（不是外观描述）
2. 如果图片中有文字（少量），也要提取
3. 如果是产品截图/界面，描述功能和交互
4. 如果是流程图/架构图，描述节点和关系
5. 用简洁的中文描述，100-300字
6. 重点提取对知识积累有价值的信息"""

# 多图关联 Prompt
MULTI_IMAGE_PROMPT = """以下是同一篇文章中的 {count} 张图片，请综合理解所有图片的内容。

规则：
1. 理解图片之间的叙事关系和逻辑顺序
2. 每张图片单独描述，用 [图片 N] 标注
3. 对每张图片，根据其类型（文字/图表/表格/代码/照片）采用最合适的提取方式：
   - 文字图：提取所有文字，保留结构
   - 图表：分析数据趋势，提取数值
   - 表格：输出 Markdown 表格
   - 代码：还原完整代码
   - 照片：描述核心内容
4. 最后用 [综合理解] 总结这组图片整体要表达的信息
5. 输出 Markdown 格式"""

PROMPT_MAP = {
    "text": PROMPT_TEXT,
    "chart": PROMPT_CHART,
    "table": PROMPT_TABLE,
    "code": PROMPT_CODE,
    "photo": PROMPT_PHOTO,
}

# 兼容旧版的通用 Prompt（分类失败时兜底）
FALLBACK_PROMPT = PROMPT_TEXT


class VisionOCR(BaseFetcher):
    """智能图片内容识别 — 基于 Vision LLM

    v0.7: 分类→专用prompt→结构化输出
    v0.8: 单例模式，避免重复检测 API 可用性
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if VisionOCR._initialized:
            return
        VisionOCR._initialized = True

        config = get_vision_config()
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
        self.model = config.get("model", VISION_MODEL)
        self._vision_available = None  # 延迟检测

    async def _check_vision_available(self) -> bool:
        """检测 Vision API 是否可用"""
        if self._vision_available is not None:
            return self._vision_available

        try:
            # 发送一个最小请求测试 API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            self._vision_available = True
            logger.info("Vision API 可用")
        except Exception as e:
            self._vision_available = False
            error_msg = str(e)
            # 分类错误
            if "401" in error_msg or "authentication" in error_msg.lower():
                self._vision_error = FetchError.auth("vision-api", error_msg)
            elif "429" in error_msg or "rate" in error_msg.lower():
                self._vision_error = FetchError.rate_limit("vision-api", error_msg)
            elif "404" in error_msg or "not found" in error_msg.lower():
                self._vision_error = FetchError.not_found("vision-api", f"模型不存在: {self.model}")
            else:
                self._vision_error = FetchError.api("vision-api", error_msg)
            logger.warning(f"Vision API 不可用: {self._vision_error.reason}")

        return self._vision_available

    async def fetch(self, url: str) -> RawContent:
        """url 参数为图片文件路径或 URL"""
        filepath = Path(url)
        if filepath.exists():
            text = await self.ocr_image_file(filepath)
        else:
            text = await self.ocr_image_url(url)

        if not text or len(text.strip()) < 10:
            raise FetchError.validation(url, "图片识别未提取到有效内容")

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
        # 检测 Vision API 是否可用
        if not await self._check_vision_available():
            return ""

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

            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                logger.debug(f"图片被防盗链拦截: {url[:60]}")
                return ""
            if len(resp.content) < 1000:
                logger.debug(f"图片过小: {url[:60]} ({len(resp.content)} bytes)")
                return ""

            return await self.ocr_image_bytes(resp.content)
        except httpx.TimeoutException:
            logger.debug(f"图片下载超时: {url[:60]}")
            return ""
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(f"图片不存在: {url[:60]}")
            else:
                logger.debug(f"图片下载失败({e.response.status_code}): {url[:60]}")
            return ""
        except Exception as e:
            logger.debug(f"图片下载异常: {url[:60]} - {type(e).__name__}")
            return ""

    async def ocr_image_bytes(self, img_bytes: bytes, suffix: str = ".png") -> str:
        """智能识别图片：先分类，再用专用 prompt 提取内容"""
        # 检测 Vision API 是否可用
        if not await self._check_vision_available():
            logger.debug("Vision API 不可用，跳过图片识别")
            return ""

        data_url = self._bytes_to_data_url(img_bytes, suffix)

        # Step 1: 分类
        img_type = await self._classify_image(data_url)
        logger.info(f"图片分类: {img_type}")

        # Step 2: 用专用 prompt 提取
        prompt = PROMPT_MAP.get(img_type, FALLBACK_PROMPT)
        return await self._call_vision(data_url, prompt)

    async def ocr_multiple_images(self, image_urls: list[str]) -> str:
        """批量 OCR 多张图片 — 支持多图关联理解"""
        if not image_urls:
            return ""

        # 如果 ≤3 张图，尝试多图关联模式（一次性发给 Vision）
        if 1 < len(image_urls) <= 3:
            result = await self._multi_image_understand(image_urls)
            if result:
                return result

        # >3 张或多图模式失败，逐张处理
        results = []
        for i, url in enumerate(image_urls):
            try:
                text = await self.ocr_image_url(url)
                if text:
                    results.append(f"[图片 {i + 1}]\n{text}")
                    logger.info(f"图片 {i+1}/{len(image_urls)} 识别完成: {len(text)} 字")
            except Exception as e:
                logger.warning(f"图片 {i + 1} 识别失败: {e}")

        return "\n\n".join(results)

    async def _multi_image_understand(self, image_urls: list[str]) -> str:
        """多图关联理解 — 将多张图一起发给 Vision 模型"""
        import httpx

        contents = [{"type": "text", "text": MULTI_IMAGE_PROMPT.format(count=len(image_urls))}]

        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

        async with httpx.AsyncClient(timeout=20) as client:
            for url in image_urls:
                req_headers = dict(headers)
                if "mmbiz" in url:
                    req_headers["Referer"] = "https://mp.weixin.qq.com/"
                elif "xhscdn" in url or "xiaohongshu" in url:
                    req_headers["Referer"] = "https://www.xiaohongshu.com/"

                try:
                    resp = await client.get(url, headers=req_headers)
                    resp.raise_for_status()
                    if len(resp.content) < 1000:
                        continue

                    data_url = self._bytes_to_data_url(resp.content)
                    contents.append({"type": "image_url", "image_url": {"url": data_url}})
                except Exception as e:
                    logger.warning(f"多图下载失败: {url[:60]} - {e}")

        if len(contents) < 2:
            return ""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": contents}],
                temperature=0.1,
                max_tokens=3000,
            )
            result = response.choices[0].message.content.strip()
            logger.info(f"多图关联识别成功: {len(result)} 字, {len(contents)-1} 张图")
            return result
        except Exception as e:
            logger.warning(f"多图关联识别失败，降级为逐张处理: {e}")
            return ""

    async def _classify_image(self, data_url: str) -> str:
        """用 Vision 模型对图片分类"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CLASSIFY_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                temperature=0,
                max_tokens=20,
            )
            raw = response.choices[0].message.content.strip().lower()
            # 提取分类标签
            for label in ["chart", "table", "code", "photo", "text"]:
                if label in raw:
                    return label
            return "text"
        except Exception as e:
            logger.warning(f"图片分类失败，默认 text: {e}")
            return "text"

    async def _call_vision(self, data_url: str, prompt: str) -> str:
        """调用 Vision 模型"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                temperature=0.1,
                max_tokens=2000,
            )
            result = response.choices[0].message.content.strip()
            logger.info(f"Vision 识别成功: {len(result)} 字")
            return result
        except Exception as e:
            logger.error(f"Vision 调用失败 ({self.model}): {e}")
            return ""

    def _bytes_to_data_url(self, img_bytes: bytes, suffix: str = ".png") -> str:
        """图片字节转 data URL（自动压缩大图）"""
        # Kimi Vision API 限制：图片 < 10MB，建议 < 1MB
        MAX_SIZE = 800 * 1024  # 800KB

        if len(img_bytes) > MAX_SIZE:
            img_bytes = self._compress_image(img_bytes, suffix, target_size=MAX_SIZE)

        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".webp": "image/webp", ".gif": "image/gif"}
        mime = mime_map.get(suffix.lower(), "image/png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _compress_image(self, img_bytes: bytes, suffix: str, target_size: int) -> bytes:
        """压缩图片到目标大小"""
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(img_bytes))

            # 转换 RGBA 为 RGB（JPEG 不支持透明通道）
            if img.mode == "RGBA":
                img = img.convert("RGB")

            # 逐步降低质量直到满足大小要求
            quality = 85
            while quality >= 30:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                if len(buf.getvalue()) <= target_size:
                    logger.debug(f"图片压缩: {len(img_bytes)} -> {len(buf.getvalue())} bytes (quality={quality})")
                    return buf.getvalue()
                quality -= 10

            # 如果还是太大，缩小尺寸
            scale = 0.8
            while scale >= 0.3:
                new_size = (int(img.width * scale), int(img.height * scale))
                resized = img.resize(new_size, Image.LANCZOS)
                buf = io.BytesIO()
                resized.save(buf, format="JPEG", quality=60)
                if len(buf.getvalue()) <= target_size:
                    logger.debug(f"图片压缩+缩放: {len(img_bytes)} -> {len(buf.getvalue())} bytes (scale={scale})")
                    return buf.getvalue()
                scale -= 0.1

            return buf.getvalue()
        except Exception as e:
            logger.warning(f"图片压缩失败，使用原图: {e}")
            return img_bytes