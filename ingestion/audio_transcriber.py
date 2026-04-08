"""音频转录器 — OpenAI Whisper API

零新依赖（复用 OpenAI SDK），零内存开销。
成本：~$0.006/分钟 (~¥2.5/小时)。
长音频 >25MB 自动分段。转录后删除原始文件节省磁盘。
"""

import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import get_llm_config
from .base import BaseFetcher, RawContent, FetchError

logger = logging.getLogger(__name__)

# Whisper API 文件大小限制
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm", ".mp4"}


class AudioTranscriber(BaseFetcher):
    """音频转录器 — 基于 Whisper API"""

    def __init__(self):
        config = get_llm_config()
        # Whisper API 必须用 OpenAI 官方端点
        self.client = OpenAI(
            api_key=config.get("api_key", ""),
            base_url="https://api.openai.com/v1",  # Whisper 只支持 OpenAI 官方
        )

    async def fetch(self, url: str) -> RawContent:
        """url 参数为本地音频文件路径"""
        return await self.transcribe_file(Path(url))

    async def transcribe_file(self, filepath: Path) -> RawContent:
        """转录音频文件

        Args:
            filepath: 音频文件路径

        Returns:
            RawContent
        """
        if not filepath.exists():
            raise FetchError(str(filepath), "文件不存在")

        if filepath.suffix.lower() not in AUDIO_EXTENSIONS:
            raise FetchError(str(filepath), f"不支持的音频格式: {filepath.suffix}")

        file_size = filepath.stat().st_size

        if file_size > MAX_FILE_SIZE:
            # 大文件需要分段
            logger.info(f"音频文件 {file_size / 1024 / 1024:.1f}MB 超过 25MB，尝试分段转录")
            text = await self._transcribe_large(filepath)
        else:
            text = await self._transcribe_single(filepath)

        if not text or len(text.strip()) < 10:
            raise FetchError(str(filepath), "音频转录未提取到有效文字")

        logger.info(f"音频转录完成: {filepath.name}, {len(text)} 字")

        return RawContent(
            url=str(filepath),
            title=filepath.stem.replace("_", " ").replace("-", " "),
            content=text,
            author="",
            source_platform="audio",
        )

    async def _transcribe_single(self, filepath: Path) -> str:
        """转录单个文件（≤25MB）"""
        try:
            with open(filepath, "rb") as f:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text",
                    language="zh",
                )
            return response.strip() if isinstance(response, str) else str(response).strip()
        except Exception as e:
            raise FetchError(str(filepath), f"Whisper API 调用失败: {e}")

    async def _transcribe_large(self, filepath: Path) -> str:
        """分段转录大文件"""
        try:
            from pydub import AudioSegment
        except ImportError:
            raise FetchError(str(filepath), "pydub 未安装，无法处理大音频文件。pip install pydub")

        try:
            audio = AudioSegment.from_file(str(filepath))
        except Exception as e:
            raise FetchError(str(filepath), f"音频文件加载失败（可能需要安装 ffmpeg）: {e}")

        # 按 10 分钟分段
        segment_ms = 10 * 60 * 1000
        segments = []
        for i in range(0, len(audio), segment_ms):
            segment = audio[i:i + segment_ms]
            segments.append(segment)

        results = []
        import tempfile
        for i, segment in enumerate(segments):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                segment.export(tmp.name, format="mp3", bitrate="64k")
                try:
                    text = await self._transcribe_single(Path(tmp.name))
                    if text:
                        results.append(text)
                    logger.info(f"分段 {i + 1}/{len(segments)} 转录完成")
                except Exception as e:
                    logger.warning(f"分段 {i + 1} 转录失败: {e}")

        return "\n\n".join(results)
