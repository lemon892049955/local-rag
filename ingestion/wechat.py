"""微信公众号文章抓取器

v0.6 图文穿插策略:
  解析 HTML 时将 <img> 替换为 [IMG_N] 占位符，保持图文相对位置。
  后续 OCR 完成后原地替换占位符，语义上下文不会脱节。
"""

import re
import logging
import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from .base import BaseFetcher, RawContent, FetchError

logger = logging.getLogger(__name__)


class WechatFetcher(BaseFetcher):
    """微信公众号文章抓取

    策略：requests 直接请求 + BeautifulSoup 解析
    微信公众号文章是服务端渲染的，不需要 JS 执行。
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    async def fetch(self, url: str) -> RawContent:
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            resp.raise_for_status()
            resp.encoding = "utf-8"
        except requests.RequestException as e:
            raise FetchError(url, f"HTTP 请求失败: {e}")

        soup = BeautifulSoup(resp.text, "lxml")

        # 提取标题
        title = self._extract_title(soup)

        # 提取作者
        author = self._extract_author(soup)

        # 提取正文 - 微信文章正文在 id="js_content" 的 div 中
        content_div = soup.find("div", id="js_content")
        if not content_div:
            content_div = soup.find("div", class_="rich_media_content")

        if not content_div:
            raise FetchError(url, "未找到文章正文区域")

        # 图文穿插提取: img → 占位符 + 收集图片URL
        image_urls = self._replace_images_with_placeholders(content_div)

        # 清洗正文 (此时 img 已变成占位符文本节点)
        content = self._clean_content(content_div)

        # 有图片时降低文本长度阈值（图片密集型文章文字可能很少）
        min_len = 10 if image_urls else 50
        if len(content.strip()) < min_len:
            raise FetchError(url, "提取的正文内容过短，可能是付费/已删除文章")

        return RawContent(
            url=url,
            title=title,
            content=content,
            author=author,
            source_platform="wechat",
            images=image_urls[:10] if image_urls else None,
        )

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """提取标题"""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        h1 = soup.find("h1", id="activity-name")
        if h1:
            return h1.get_text(strip=True)

        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else "未知标题"

    def _extract_author(self, soup: BeautifulSoup) -> str:
        """提取作者/公众号名"""
        author_tag = soup.find("a", id="js_name")
        if author_tag:
            return author_tag.get_text(strip=True)

        meta_author = soup.find("meta", {"name": "author"})
        if meta_author and meta_author.get("content"):
            return meta_author["content"].strip()

        return ""

    def _replace_images_with_placeholders(self, content_div) -> list:
        """将 <img> 替换为 [IMG_N] 占位符，保持图文相对位置。

        返回按顺序排列的图片 URL 列表（index 对应占位符编号）。
        装饰性小图（表情/图标）直接删除，不生成占位符。
        """
        image_urls = []
        img_index = 0

        for img in content_div.find_all("img"):
            src = img.get("data-src") or img.get("src") or ""

            # 非 mmbiz 图片或无 src → 直接删除
            if not src or not src.startswith("http") or "mmbiz" not in src:
                img.decompose()
                continue

            # 过滤装饰性小图（宽度<100px 的表情/图标）
            width = img.get("data-w") or img.get("width") or "999"
            try:
                if int(str(width).replace("px", "")) < 100:
                    img.decompose()
                    continue
            except (ValueError, TypeError):
                pass

            # 有意义的图片 → 替换为占位符
            img_index += 1
            image_urls.append(src)
            placeholder = f"\n\n[IMG_{img_index}]\n\n"
            img.replace_with(NavigableString(placeholder))

        return image_urls

    def _clean_content(self, content_div) -> str:
        """清洗正文 HTML -> 纯文本（保留 [IMG_N] 占位符）"""
        # 移除不需要的元素
        for tag in content_div.find_all(["script", "style", "iframe"]):
            tag.decompose()

        # 移除底部二维码区域和广告
        for div in content_div.find_all("div"):
            text = div.get_text(strip=True)
            if any(keyword in text for keyword in [
                "扫描二维码", "长按识别", "点击关注", "阅读原文",
                "点赞", "在看", "广告",
            ]):
                # 只移除短文本的干扰 div（排除含占位符的）
                if len(text) < 100 and "[IMG_" not in text:
                    div.decompose()

        # 获取纯文本，保留段落结构
        lines = []
        for element in content_div.find_all(["p", "h1", "h2", "h3", "h4", "li", "blockquote"]):
            text = element.get_text(strip=True)
            if not text:
                continue

            # 占位符单独成行
            if "[IMG_" in text:
                # 拆分出占位符和文本
                parts = re.split(r'(\[IMG_\d+\])', text)
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    if re.match(r'\[IMG_\d+\]', part):
                        lines.append(part)
                    else:
                        lines.append(part)
                continue

            tag_name = element.name
            if tag_name in ("h1", "h2", "h3", "h4"):
                prefix = "#" * int(tag_name[1])
                lines.append(f"{prefix} {text}")
            elif tag_name == "li":
                lines.append(f"- {text}")
            elif tag_name == "blockquote":
                lines.append(f"> {text}")
            else:
                lines.append(text)

        content = "\n\n".join(lines)

        # 清理多余空行和空格
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r"[ \t]+", " ", content)

        return content.strip()
