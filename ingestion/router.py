"""数据摄入层 - 路由器

根据 URL 自动分发到对应的抓取器。
"""

from .base import BaseFetcher, RawContent, FetchError
from .wechat import WechatFetcher
from .general import GeneralFetcher
from .xiaohongshu import XiaohongshuFetcher
from utils.url_utils import detect_source


class FetcherRouter:
    """抓取器路由 - 根据 URL 域名自动选择抓取器"""

    def __init__(self, xhs_mcp_endpoint=None):
        self._fetchers: dict[str, BaseFetcher] = {
            "wechat": WechatFetcher(),
            "general": GeneralFetcher(),
            "xiaohongshu": XiaohongshuFetcher(mcp_endpoint=xhs_mcp_endpoint),
        }

    async def fetch(self, url: str) -> RawContent:
        """根据 URL 自动路由到对应抓取器"""
        source = detect_source(url)
        fetcher = self._fetchers.get(source, self._fetchers["general"])

        result = await fetcher.fetch(url)
        result.url = url  # 确保 URL 被正确设置
        result.source_platform = source

        return result
