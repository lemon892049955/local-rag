"""抓取器基类 - 定义统一接口"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RawContent:
    """抓取器输出的原始内容"""
    url: str
    title: str
    content: str          # 纯文本正文
    author: str = ""
    source_platform: str = ""  # xiaohongshu / wechat / general
    original_tags: Optional[List[str]] = None  # 平台原始标签（如小红书标签）
    images: Optional[List[str]] = None  # 图片 URL 列表（供后续 OCR）


class BaseFetcher(ABC):
    """抓取器基类"""

    @abstractmethod
    async def fetch(self, url: str) -> RawContent:
        """抓取并返回原始内容

        Args:
            url: 目标页面的 URL

        Returns:
            RawContent 数据对象

        Raises:
            FetchError: 抓取失败时抛出
        """
        ...


class FetchError(Exception):
    """抓取失败异常"""
    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"抓取失败 [{url}]: {reason}")
