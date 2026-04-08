"""小红书抓取器 - 基于 MCP 协议 (降级为 HTTP 模拟)

注意：完整的 MCP 抓取需要 xiaohongshu-mcp 服务端运行。
此模块提供两种模式：
1. MCP 模式：通过 MCP 协议调用（需要 mcp 服务端）
2. 降级模式：通过 HTTP 请求 + 解析（有反爬限制）
"""

import json
import re
import requests
from urllib.parse import urlparse

from .base import BaseFetcher, RawContent, FetchError
from utils.url_utils import extract_xhs_note_id


class XiaohongshuFetcher(BaseFetcher):
    """小红书笔记抓取器

    优先使用 MCP 模式，失败时降级为基础 HTTP 抓取。
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.0 Mobile/15E148 Safari/604.1"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def __init__(self, mcp_endpoint=None):
        """
        Args:
            mcp_endpoint: MCP 服务的 HTTP 端点 (如 http://localhost:3000)
                         为 None 时使用降级模式
        """
        self.mcp_endpoint = mcp_endpoint

    async def fetch(self, url: str) -> RawContent:
        """抓取小红书笔记"""
        # 优先尝试 MCP 模式
        if self.mcp_endpoint:
            try:
                return await self._fetch_via_mcp(url)
            except Exception:
                pass  # 降级

        # 降级：直接 HTTP 抓取
        return await self._fetch_via_http(url)

    async def _fetch_via_mcp(self, url: str) -> RawContent:
        """通过 MCP 协议抓取（需要 xiaohongshu-mcp 服务端）"""
        note_id = extract_xhs_note_id(url)
        if not note_id:
            raise FetchError(url, "无法从 URL 中提取小红书笔记 ID")

        try:
            resp = requests.post(
                f"{self.mcp_endpoint}/api/note",
                json={"note_id": note_id},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise FetchError(url, f"MCP 调用失败: {e}")

        return RawContent(
            url=url,
            title=data.get("title", ""),
            content=data.get("content", ""),
            author=data.get("author", ""),
            source_platform="xiaohongshu",
            original_tags=data.get("tags", []),
        )

    async def _fetch_via_http(self, url: str) -> RawContent:
        """降级模式：HTTP 直接抓取"""
        # 处理短链跳转
        try:
            resp = requests.get(
                url, headers=self.HEADERS, timeout=15, allow_redirects=True
            )
            resp.raise_for_status()
            resp.encoding = "utf-8"
        except requests.RequestException as e:
            raise FetchError(url, f"HTTP 请求失败: {e}")

        html = resp.text

        # 小红书的 SSR 数据通常在 window.__INITIAL_STATE__ 中
        content_data = self._extract_ssr_data(html)
        if content_data:
            return content_data

        # 兜底：尝试从 HTML 中直接解析
        return self._parse_html_fallback(url, html)

    def _extract_ssr_data(self, html: str):
        """从 SSR 注入的 JSON 中提取数据"""
        pattern = r"window\.__INITIAL_STATE__\s*=\s*({.+?})\s*</script>"
        match = re.search(pattern, html, re.DOTALL)
        if not match:
            return None

        try:
            # 小红书的 JSON 中可能有 undefined，需要替换
            json_str = match.group(1).replace("undefined", "null")
            data = json.loads(json_str)

            # 路径1: noteData.data.noteData (完整数据)
            note = (data.get("noteData", {})
                        .get("data", {})
                        .get("noteData", {}))

            # 路径2: noteData.normalNotePreloadData (预加载数据)
            preload = (data.get("noteData", {})
                           .get("normalNotePreloadData", {}))

            # 路径3: 旧版结构 note.noteDetailMap
            old_note_data = data.get("note", {}).get("noteDetailMap", {})

            title = ""
            desc = ""
            author = ""
            tags = []
            image_urls = []

            if note and note.get("desc"):
                title = note.get("title", "")
                desc = note.get("desc", "")
                user = note.get("user", {})
                author = user.get("nickname", "") if isinstance(user, dict) else ""
                tag_list = note.get("tagList", [])
                if isinstance(tag_list, list):
                    tags = [t.get("name", "") for t in tag_list if t.get("name")]
                # 提取图片列表
                for img in note.get("imageList", []):
                    img_url = img.get("urlDefault") or img.get("url") or ""
                    if img_url and img_url.startswith("http"):
                        image_urls.append(img_url)
            elif preload and preload.get("desc"):
                title = preload.get("title", "")
                desc = preload.get("desc", "")
                for img in preload.get("imageList", []):
                    img_url = img.get("urlDefault") or img.get("url") or ""
                    if img_url and img_url.startswith("http"):
                        image_urls.append(img_url)
            elif old_note_data:
                first_note = next(iter(old_note_data.values()), {})
                note_inner = first_note.get("note", {})
                title = note_inner.get("title", "")
                desc = note_inner.get("desc", "")
                author = note_inner.get("user", {}).get("nickname", "")
                for tag in note_inner.get("tagList", []):
                    tag_name = tag.get("name", "")
                    if tag_name:
                        tags.append(tag_name)
                for img in note_inner.get("imageList", []):
                    img_url = img.get("urlDefault") or img.get("url") or ""
                    if img_url and img_url.startswith("http"):
                        image_urls.append(img_url)

            content = f"{title}\n\n{desc}" if title else desc

            if not content.strip() and not image_urls:
                return None

            return RawContent(
                url="",  # 调用方会填充
                title=title or "小红书笔记",
                content=content or "(图片笔记)",
                author=author,
                source_platform="xiaohongshu",
                original_tags=tags or None,
                images=image_urls[:10] if image_urls else None,
            )
        except (json.JSONDecodeError, StopIteration):
            return None

    def _parse_html_fallback(self, url: str, html: str) -> RawContent:
        """HTML 兜底解析"""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")

        # 尝试获取标题
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"]

        # 尝试获取描述
        content = ""
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            content = og_desc["content"]

        if not content:
            raise FetchError(
                url,
                "小红书反爬限制，建议启用 MCP 模式。请配置 xiaohongshu-mcp 服务端。"
            )

        return RawContent(
            url=url,
            title=title or "小红书笔记",
            content=content,
            author="",
            source_platform="xiaohongshu",
        )
