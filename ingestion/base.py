"""抓取器基类 - 定义统一接口"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ErrorType(Enum):
    """错误类型分类"""
    NETWORK = "network"           # 网络错误（超时、连接失败）
    RATE_LIMIT = "rate_limit"     # API 限流
    AUTH = "auth"                 # 认证失败（API Key 无效）
    NOT_FOUND = "not_found"       # 资源不存在（404）
    BLOCKED = "blocked"           # 被反爬拦截
    PARSE = "parse"               # 解析失败
    VALIDATION = "validation"     # 内容校验失败（如内容过短）
    API = "api"                   # API 错误（其他）
    UNKNOWN = "unknown"           # 未知错误


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
    """抓取失败异常 - 支持错误类型分类"""

    def __init__(self, url: str, reason: str, error_type: ErrorType = ErrorType.UNKNOWN):
        self.url = url
        self.reason = reason
        self.error_type = error_type
        super().__init__(f"[{error_type.value}] {url}: {reason}")

    @classmethod
    def network(cls, url: str, reason: str = "网络错误"):
        return cls(url, reason, ErrorType.NETWORK)

    @classmethod
    def rate_limit(cls, url: str, reason: str = "API 限流"):
        return cls(url, reason, ErrorType.RATE_LIMIT)

    @classmethod
    def auth(cls, url: str, reason: str = "认证失败"):
        return cls(url, reason, ErrorType.AUTH)

    @classmethod
    def not_found(cls, url: str, reason: str = "资源不存在"):
        return cls(url, reason, ErrorType.NOT_FOUND)

    @classmethod
    def blocked(cls, url: str, reason: str = "被反爬拦截"):
        return cls(url, reason, ErrorType.BLOCKED)

    @classmethod
    def parse(cls, url: str, reason: str = "解析失败"):
        return cls(url, reason, ErrorType.PARSE)

    @classmethod
    def validation(cls, url: str, reason: str = "内容校验失败"):
        return cls(url, reason, ErrorType.VALIDATION)

    @classmethod
    def api(cls, url: str, reason: str = "API 错误"):
        return cls(url, reason, ErrorType.API)
