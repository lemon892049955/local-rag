"""通用网页抓取器 - 基于 Readability 算法"""

import json
import re
import logging
import requests
from readability import Document
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from .base import BaseFetcher, RawContent, FetchError

logger = logging.getLogger(__name__)


class GeneralFetcher(BaseFetcher):
    """通用网页抓取器

    使用 readability-lxml 提取主要内容，
    适用于知乎、博客、新闻等常规网页。
    对知乎等有反爬的站点做特殊处理。
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }

    def _is_zhihu(self, url: str) -> bool:
        return "zhihu.com" in urlparse(url).netloc

    async def fetch(self, url: str) -> RawContent:
        # 知乎走专用路径
        if self._is_zhihu(url):
            return await self._fetch_zhihu(url)

        session = requests.Session()
        session.headers.update(self.HEADERS)
        try:
            resp = session.get(url, timeout=15, allow_redirects=True)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
        except requests.RequestException as e:
            raise FetchError(url, f"HTTP 请求失败: {e}")

        # 使用 Readability 提取主要内容
        doc = Document(resp.text)
        title = doc.title() or "未知标题"
        html_content = doc.summary()

        # HTML -> 结构化文本
        content = self._html_to_text(html_content)

        if len(content.strip()) < 30:
            raise FetchError(url, "Readability 提取内容过短，页面可能需要 JS 渲染")

        # 尝试提取作者
        author = self._extract_author(resp.text)

        return RawContent(
            url=url,
            title=title,
            content=content,
            author=author,
            source_platform="general",
        )

    def _html_to_text(self, html: str) -> str:
        """将 Readability 输出的 HTML 转为 Markdown 风格纯文本"""
        soup = BeautifulSoup(html, "lxml")

        # 移除残余的无用元素
        for tag in soup.find_all(["script", "style", "iframe"]):
            tag.decompose()

        lines = []
        for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "blockquote", "pre", "code"]):
            text = element.get_text(strip=True)
            if not text:
                continue

            tag_name = element.name
            if tag_name in ("h1", "h2", "h3", "h4"):
                prefix = "#" * int(tag_name[1])
                lines.append(f"{prefix} {text}")
            elif tag_name == "li":
                lines.append(f"- {text}")
            elif tag_name == "blockquote":
                lines.append(f"> {text}")
            elif tag_name in ("pre", "code"):
                lines.append(f"```\n{text}\n```")
            else:
                lines.append(text)

        return "\n\n".join(lines)

    def _extract_author(self, html: str) -> str:
        """尝试从页面 meta 中提取作者"""
        soup = BeautifulSoup(html, "lxml")

        # 常见的作者 meta 标签
        for attr in [
            {"name": "author"},
            {"property": "article:author"},
            {"name": "byl"},
        ]:
            tag = soup.find("meta", attr)
            if tag and tag.get("content"):
                return tag["content"].strip()

        # 知乎特殊处理
        author_link = soup.find("a", class_="UserLink-link")
        if author_link:
            return author_link.get_text(strip=True)

        return ""

    async def _fetch_zhihu(self, url: str) -> RawContent:
        """知乎专用抓取 - 通过 API 接口获取回答/文章内容"""
        parsed = urlparse(url)
        path = parsed.path

        # 尝试提取回答 ID
        answer_match = re.search(r"/answer/(\d+)", path)
        # 尝试提取文章 ID (zhuanlan)
        article_match = re.search(r"/p/(\d+)", path)

        if answer_match:
            return await self._fetch_zhihu_answer(url, answer_match.group(1))
        elif article_match:
            return await self._fetch_zhihu_article(url, article_match.group(1))
        else:
            # 问题页面，尝试用 Readability 兜底
            return await self._fetch_zhihu_fallback(url)

    async def _fetch_zhihu_answer(self, url: str, answer_id: str) -> RawContent:
        """通过知乎 API 获取回答内容"""
        api_url = f"https://www.zhihu.com/api/v4/answers/{answer_id}"
        params = {"include": "content,excerpt,author,question"}

        session = requests.Session()
        headers = dict(self.HEADERS)
        headers["Referer"] = "https://www.zhihu.com/"

        try:
            resp = session.get(api_url, headers=headers, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                content_html = data.get("content", "")
                question = data.get("question", {})
                title = question.get("title", data.get("excerpt", "知乎回答")[:50])
                author = data.get("author", {}).get("name", "")

                content = self._html_to_text(content_html)
                if len(content.strip()) > 30:
                    return RawContent(
                        url=url,
                        title=title,
                        content=content,
                        author=author,
                        source_platform="general",
                    )
        except Exception:
            pass

        # API 失败，走页面解析兜底
        return await self._fetch_zhihu_fallback(url)

    async def _fetch_zhihu_article(self, url: str, article_id: str) -> RawContent:
        """通过知乎 API 获取专栏文章"""
        api_url = f"https://www.zhihu.com/api/v4/articles/{article_id}"

        session = requests.Session()
        headers = dict(self.HEADERS)
        headers["Referer"] = "https://www.zhihu.com/"

        try:
            resp = session.get(api_url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                title = data.get("title", "知乎文章")
                content_html = data.get("content", "")
                author = data.get("author", {}).get("name", "")

                content = self._html_to_text(content_html)
                if len(content.strip()) > 30:
                    return RawContent(
                        url=url,
                        title=title,
                        content=content,
                        author=author,
                        source_platform="general",
                    )
        except Exception:
            pass

        return await self._fetch_zhihu_fallback(url)

    async def _fetch_zhihu_fallback(self, url: str) -> RawContent:
        """知乎兜底：尝试从页面 SSR JSON 中提取"""
        session = requests.Session()
        headers = dict(self.HEADERS)
        headers["Referer"] = "https://www.zhihu.com/"
        # 模拟搜索引擎 referrer 有时能绕过反爬
        headers["Referer"] = "https://www.google.com/"

        try:
            resp = session.get(url, headers=headers, timeout=15, allow_redirects=True)
            resp.encoding = "utf-8"
            html = resp.text
        except requests.RequestException as e:
            raise FetchError(url, f"知乎请求失败: {e}")

        # 尝试从 SSR 注入的 initialData 中提取
        pattern = r'<script\s+id="js-initialData"\s+type="text/json">\s*({.+?})\s*</script>'
        match = re.search(pattern, html, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return self._parse_zhihu_initial_data(url, data)
            except (json.JSONDecodeError, KeyError):
                pass

        # 最后尝试 Readability
        if resp.status_code == 200 and len(html) > 500:
            doc = Document(html)
            title = doc.title() or "知乎内容"
            content = self._html_to_text(doc.summary())
            if len(content.strip()) > 30:
                return RawContent(
                    url=url,
                    title=title,
                    content=content,
                    author="",
                    source_platform="general",
                )

        raise FetchError(url, "知乎反爬限制，无法获取内容。建议手动复制文章内容后使用。")

    def _parse_zhihu_initial_data(self, url: str, data: dict) -> RawContent:
        """从知乎 SSR initialData 中提取内容"""
        # 尝试从 initialState.entities.answers 中获取
        entities = data.get("initialState", {}).get("entities", {})

        # 回答
        answers = entities.get("answers", {})
        if answers:
            answer = next(iter(answers.values()))
            content_html = answer.get("content", "")
            question = answer.get("question", {})
            title = question.get("title", "") if isinstance(question, dict) else ""
            author_info = answer.get("author", {})
            author = author_info.get("name", "") if isinstance(author_info, dict) else ""

            content = self._html_to_text(content_html)
            if content.strip():
                return RawContent(
                    url=url,
                    title=title or "知乎回答",
                    content=content,
                    author=author,
                    source_platform="general",
                )

        # 文章
        articles = entities.get("articles", {})
        if articles:
            article = next(iter(articles.values()))
            title = article.get("title", "知乎文章")
            content_html = article.get("content", "")
            author = article.get("author", {}).get("name", "")

            content = self._html_to_text(content_html)
            if content.strip():
                return RawContent(
                    url=url,
                    title=title,
                    content=content,
                    author=author,
                    source_platform="general",
                )

        raise FetchError(url, "无法从知乎 SSR 数据中解析内容")

    def _extract_images(self, html: str) -> list:
        """从 HTML 中提取有意义的图片 URL"""
        soup = BeautifulSoup(html, "lxml")
        urls = []
        for img in soup.find_all("img"):
            src = img.get("data-original") or img.get("data-actualsrc") or img.get("src") or ""
            if not src or not src.startswith("http"):
                continue
            # 过滤小图标/表情/头像
            if any(skip in src for skip in ["emoji", "avatar", "icon", "logo", "gif"]):
                continue
            urls.append(src)
        return urls
