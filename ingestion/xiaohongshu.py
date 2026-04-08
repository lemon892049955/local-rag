"""小红书抓取器 — MCP 优先 + HTTP 降级

优先通过 xiaohongshu-mcp 服务获取完整数据（文字+图片+标签+互动），
MCP 不可用时降级为 HTTP 直抓（受反爬限制）。

MCP 服务启动方式（Docker）:
  docker pull xpzouying/xiaohongshu-mcp
  docker compose up -d
  # 默认端口: http://localhost:18060/mcp
"""

import json
import re
import logging
import requests
from urllib.parse import urlparse, parse_qs

from .base import BaseFetcher, RawContent, FetchError
from utils.url_utils import extract_xhs_note_id

logger = logging.getLogger(__name__)


class XiaohongshuFetcher(BaseFetcher):
    """小红书笔记抓取器

    优先 MCP 模式（完整数据），失败降级 HTTP（有反爬限制）。
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
            mcp_endpoint: MCP 服务端点 (如 http://localhost:18060/mcp)
        """
        self.mcp_endpoint = mcp_endpoint
        self._mcp_session_id = None
        self._mcp_http_session = None

    async def fetch(self, url: str) -> RawContent:
        """抓取小红书笔记"""
        # 优先 MCP
        if self.mcp_endpoint:
            try:
                result = await self._fetch_via_mcp(url)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"MCP 抓取失败，降级 HTTP: {e}")

        # 降级 HTTP
        return await self._fetch_via_http(url)

    # ===== MCP 模式 =====

    async def _fetch_via_mcp(self, url: str) -> RawContent:
        """通过 xiaohongshu-mcp 服务获取笔记完整数据"""
        feed_id, xsec_token = self._extract_feed_params(url)

        if not feed_id:
            # 尝试先搜索获取
            logger.info(f"URL 中未提取到 feed_id，尝试通过 MCP search 获取")
            raise FetchError(url, "无法从 URL 中提取笔记 ID")

        # 调用 MCP get_feed_detail
        result = self._call_mcp_tool("get_feed_detail", {
            "feed_id": feed_id,
            "xsec_token": xsec_token or "",
        })

        if not result:
            raise FetchError(url, "MCP get_feed_detail 返回空结果")

        # 解析 MCP 返回的数据（结构: {feed_id, data: {note: {...}}}）
        note = result
        if "data" in result and isinstance(result["data"], dict):
            note = result["data"].get("note", result["data"])
        if "note" in result and isinstance(result["note"], dict):
            note = result["note"]

        title = note.get("title", "")
        desc = note.get("desc", "") or note.get("content", "")
        author = ""
        user = note.get("user", {})
        if isinstance(user, dict):
            author = user.get("nickname", "") or user.get("name", "")

        images = note.get("images", []) or note.get("imageList", [])
        tags = []

        # 提取标签
        tag_list = result.get("tagList", []) or result.get("tags", [])
        if isinstance(tag_list, list):
            for t in tag_list:
                if isinstance(t, dict):
                    name = t.get("name", "")
                    if name:
                        tags.append(name)
                elif isinstance(t, str) and t:
                    tags.append(t)

        # 提取图片 URL
        image_urls = []
        if isinstance(images, list):
            for img in images:
                if isinstance(img, str) and img.startswith("http"):
                    image_urls.append(img)
                elif isinstance(img, dict):
                    img_url = img.get("urlDefault") or img.get("url") or img.get("src") or ""
                    if img_url and img_url.startswith("http"):
                        image_urls.append(img_url)

        content = f"{title}\n\n{desc}" if title else desc

        # 图文穿插
        if image_urls and content.strip():
            content = self._interleave_image_placeholders(content, len(image_urls))

        if not content.strip() and not image_urls:
            raise FetchError(url, "MCP 返回的笔记内容为空")

        logger.info(f"MCP 抓取成功: {title[:30]}, {len(desc)}字, {len(image_urls)}张图")

        return RawContent(
            url=url,
            title=title or "小红书笔记",
            content=content or "(图片笔记)",
            author=author,
            source_platform="xiaohongshu",
            original_tags=tags or None,
            images=image_urls[:10] if image_urls else None,
        )

    def _ensure_mcp_session(self):
        """确保 MCP session 已初始化"""
        if self._mcp_session_id:
            return True
        try:
            self._mcp_http_session = requests.Session()
            # Step 1: initialize
            r = self._mcp_http_session.post(self.mcp_endpoint, json={
                "jsonrpc": "2.0", "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "local-rag", "version": "0.6"},
                },
            }, timeout=10)
            self._mcp_session_id = r.headers.get("Mcp-Session-Id", "")
            if not self._mcp_session_id:
                logger.warning("MCP 未返回 Session ID")
                return False
            # Step 2: initialized notification
            self._mcp_http_session.post(self.mcp_endpoint, json={
                "jsonrpc": "2.0", "method": "notifications/initialized",
            }, headers={"Mcp-Session-Id": self._mcp_session_id}, timeout=5)
            logger.info(f"MCP session 已建立: {self._mcp_session_id[:8]}...")
            return True
        except Exception as e:
            logger.warning(f"MCP session 初始化失败: {e}")
            self._mcp_session_id = None
            return False

    def _call_mcp_tool(self, tool_name: str, params: dict) -> dict:
        """调用 MCP 服务的工具（带 session 管理）"""
        if not self._ensure_mcp_session():
            return None
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": params,
                },
            }
            resp = self._mcp_http_session.post(
                self.mcp_endpoint,
                json=payload,
                timeout=60,
                headers={
                    "Content-Type": "application/json",
                    "Mcp-Session-Id": self._mcp_session_id,
                },
            )
            resp.raise_for_status()

            data = resp.json()

            if "error" in data:
                logger.error(f"MCP 错误: {data['error']}")
                # session 可能过期，重置
                if "session" in str(data["error"]).lower():
                    self._mcp_session_id = None
                return None

            result = data.get("result", {})

            # MCP tools/call 的 result 格式: {"content": [{"type": "text", "text": "..."}]}
            content_list = result.get("content", [])
            if content_list:
                for item in content_list:
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            return {"desc": text}
            return result

        except requests.exceptions.ConnectionError:
            logger.warning(f"MCP 服务未运行: {self.mcp_endpoint}")
            self._mcp_session_id = None
            return None
        except Exception as e:
            logger.error(f"MCP 调用失败 [{tool_name}]: {e}")
            return None

    def _extract_feed_params(self, url: str) -> tuple:
        """从小红书 URL 中提取 feed_id 和 xsec_token

        支持格式:
        - https://www.xiaohongshu.com/explore/xxxx
        - https://www.xiaohongshu.com/discovery/item/xxxx
        - https://xhslink.com/xxxx (短链需跳转)
        """
        parsed = urlparse(url)
        query = parse_qs(parsed.query)

        # 提取 xsec_token
        xsec_token = query.get("xsec_token", [""])[0]

        # 提取 feed_id
        feed_id = extract_xhs_note_id(url)

        # 短链: 先跳转获取真实 URL
        if not feed_id and ("xhslink.com" in parsed.netloc or "xhs.cn" in parsed.netloc):
            try:
                resp = requests.head(url, headers=self.HEADERS, allow_redirects=True, timeout=10)
                real_url = resp.url
                feed_id = extract_xhs_note_id(real_url)
                if not xsec_token:
                    real_query = parse_qs(urlparse(real_url).query)
                    xsec_token = real_query.get("xsec_token", [""])[0]
            except Exception:
                pass

        return feed_id, xsec_token

    def _mcp_search(self, keyword: str) -> list:
        """通过 MCP 搜索小红书内容"""
        result = self._call_mcp_tool("search_feeds", {"keyword": keyword})
        if result and isinstance(result, list):
            return result
        if result and isinstance(result, dict):
            return result.get("items", []) or result.get("feeds", [])
        return []

    # ===== HTTP 降级模式 =====

    async def _fetch_via_http(self, url: str) -> RawContent:
        """降级模式：HTTP 直接抓取"""
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
            json_str = match.group(1).replace("undefined", "null")
            data = json.loads(json_str)

            note = (data.get("noteData", {})
                        .get("data", {})
                        .get("noteData", {}))
            preload = (data.get("noteData", {})
                           .get("normalNotePreloadData", {}))
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

            if image_urls and content.strip():
                content = self._interleave_image_placeholders(content, len(image_urls))

            if not content.strip() and not image_urls:
                return None

            return RawContent(
                url="",
                title=title or "小红书笔记",
                content=content or "(图片笔记)",
                author=author,
                source_platform="xiaohongshu",
                original_tags=tags or None,
                images=image_urls[:10] if image_urls else None,
            )
        except (json.JSONDecodeError, StopIteration):
            return None

    # ===== 工具方法 =====

    def _interleave_image_placeholders(self, content: str, image_count: int) -> str:
        """将图片占位符均匀插入正文段落之间"""
        paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
        if not paragraphs:
            return "\n\n".join(f"[IMG_{i+1}]" for i in range(image_count))

        result = []
        img_idx = 0
        step = max(1, len(paragraphs) / max(image_count, 1))

        for i, para in enumerate(paragraphs):
            result.append(para)
            if img_idx < image_count and (i + 1) >= round(step * (img_idx + 1)):
                img_idx += 1
                result.append(f"[IMG_{img_idx}]")

        while img_idx < image_count:
            img_idx += 1
            result.append(f"[IMG_{img_idx}]")

        return "\n\n".join(result)

    def _parse_html_fallback(self, url: str, html: str) -> RawContent:
        """HTML 兜底解析"""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")

        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"]

        content = ""
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            content = og_desc["content"]

        if not content:
            raise FetchError(
                url,
                "小红书反爬限制，建议启用 MCP 模式。"
                "启动方式: docker pull xpzouying/xiaohongshu-mcp && docker compose up -d"
            )

        return RawContent(
            url=url,
            title=title or "小红书笔记",
            content=content,
            author="",
            source_platform="xiaohongshu",
        )
