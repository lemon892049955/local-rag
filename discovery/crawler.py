"""热榜抓取器 — 监控 AI 领域热门内容

支持数据源（v2.0 扩展版）：
P0 优先：
- 微信公众号热门（搜狗微信）
- 小红书热门

P1 常规：
- Hacker News (AI 相关)
- GitHub Trending (AI 相关)
- 知乎热榜 (AI 分类)
- V2EX 热门 (技术相关)
- 微博热搜 AI

P2 扩展：
- ProductHunt (AI 产品)
- 36氪 AI
- 量子位
- 机器之心
- 技术头条
"""

import asyncio
import logging
import re
import json
import hashlib
from datetime import datetime
from typing import Optional
from pathlib import Path

import httpx

from config import get_httpx_proxy

logger = logging.getLogger(__name__)

# 存储路径
PENDING_FILE = Path(__file__).parent.parent / "data" / "pending_items.json"

# 重试配置
MAX_RETRIES = 2
RETRY_DELAY = 1.0  # 秒

# 代理配置
_proxy = get_httpx_proxy()


def _generate_id(url: str) -> str:
    """根据 URL 生成唯一 ID"""
    return hashlib.md5(url.encode()).hexdigest()[:8]


def _get_client(timeout: int = 10) -> httpx.AsyncClient:
    """创建带代理的 HTTP 客户端"""
    if _proxy:
        return httpx.AsyncClient(timeout=timeout, proxy=_proxy)
    return httpx.AsyncClient(timeout=timeout)


async def _fetch_with_retry(url: str, client: httpx.AsyncClient, method: str = "GET", **kwargs) -> httpx.Response:
    """带重试的 HTTP 请求"""
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            if method.upper() == "POST":
                resp = await client.post(url, **kwargs)
            else:
                resp = await client.get(url, **kwargs)
            return resp
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                logger.debug(f"请求重试 ({attempt + 1}/{MAX_RETRIES}): {url[:50]}")
    raise last_error


# ===== 微信公众号热门 =====

async def fetch_wechat_hot(top_n: int = 30) -> list[dict]:
    """获取百度热搜"""
    items = []

    try:
        async with _get_client(timeout=15) as client:
            resp = await _fetch_with_retry(
                "https://top.baidu.com/api/board?platform=wise&tab=realtime",
                client,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            if resp.status_code == 200:
                data = resp.json()
                cards = data.get("data", {}).get("cards", [])
                for card in cards:
                    # 数据结构: cards[0]['content'][0]['content'] 是实际列表
                    content_list = card.get("content", [])
                    if isinstance(content_list, list):
                        for content_item in content_list:
                            if isinstance(content_item, dict):
                                hot_list = content_item.get("content", [])
                                if isinstance(hot_list, list):
                                    for item in hot_list[:top_n]:
                                        title = item.get("word", "") or item.get("title", "")
                                        url = item.get("url", "") or item.get("rawUrl", "")
                                        if title and len(title) >= 5:
                                            items.append({
                                                "id": _generate_id(url or f"baidu-{title}"),
                                                "title": title[:100],
                                                "url": url,
                                                "source": "百度热搜",
                                                "score": item.get("hotScore", 0) or item.get("index", 0),
                                                "published_at": datetime.now().strftime("%Y-%m-%d"),
                                                "fetched_at": datetime.now().isoformat(),
                                                "status": "pending",
                                            })
    except Exception as e:
        logger.debug(f"百度热搜抓取失败: {e}")

    # 方案2: 使用 alapi 备选
    if len(items) < 5:
        try:
            async with _get_client(timeout=10) as client:
                resp = await _fetch_with_retry(
                    "https://v2.alapi.cn/api/new/wxHeadlines",
                    client,
                    params={"num": top_n},
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("data", {}).get("list", [])[:top_n]:
                        title = item.get("title", "")
                        url = item.get("url", "")
                        if title:
                            items.append({
                                "id": _generate_id(url or f"wechat-{title}"),
                                "title": title[:100],
                                "url": url,
                                "source": "微信公众号",
                                "score": 0,
                                "published_at": datetime.now().strftime("%Y-%m-%d"),
                                "fetched_at": datetime.now().isoformat(),
                                "status": "pending",
                            })
        except Exception as e:
            logger.debug(f"微信方案2失败: {e}")

    # 方案3: 搜狗微信
    if len(items) < 5:
        try:
            async with _get_client(timeout=10) as client:
                resp = await _fetch_with_retry(
                    "https://weixin.sogou.com/wap/wapsearch.shtml",
                    client,
                    params={"keyword": "人工智能", "type": 2},
                    headers={"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"}
                )
                html = resp.text
                # 简单提取标题
                titles = re.findall(r'<a[^>]*>([^<]{10,100})</a>', html)
                for title in titles[:top_n]:
                    title = title.strip()
                    if title and len(title) >= 5:
                        items.append({
                            "id": _generate_id(f"wechat-{title}"),
                            "title": title[:100],
                            "url": "",
                            "source": "微信公众号",
                            "score": 0,
                            "published_at": datetime.now().strftime("%Y-%m-%d"),
                            "fetched_at": datetime.now().isoformat(),
                            "status": "pending",
                        })
        except Exception as e:
            logger.debug(f"微信方案3失败: {e}")

    logger.info(f"微信公众号: 获取 {len(items)} 篇")
    return items[:top_n]


# ===== 小红书热门 =====

async def fetch_xiaohongshu_hot(top_n: int = 30) -> list[dict]:
    """获取百度热搜（筛选科技/AI相关）"""
    items = []

    try:
        async with _get_client(timeout=15) as client:
            resp = await _fetch_with_retry(
                "https://top.baidu.com/api/board?platform=wise&tab=realtime",
                client,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            if resp.status_code == 200:
                data = resp.json()
                cards = data.get("data", {}).get("cards", [])
                ai_keywords = ["AI", "人工智能", "GPT", "ChatGPT", "大模型", "机器学习",
                              "智能", "机器人", "算法", "科技", "互联网", "手机", "苹果", "华为",
                              "DeepSeek", "宇树", "荣耀"]
                for card in cards:
                    content_list = card.get("content", [])
                    if isinstance(content_list, list):
                        for content_item in content_list:
                            if isinstance(content_item, dict):
                                hot_list = content_item.get("content", [])
                                if isinstance(hot_list, list):
                                    for item in hot_list[:top_n * 2]:
                                        title = item.get("word", "") or item.get("title", "")
                                        url = item.get("url", "") or item.get("rawUrl", "")
                                        # 筛选科技/AI相关
                                        if title and any(kw in title for kw in ai_keywords):
                                            items.append({
                                                "id": _generate_id(url or f"baidu-{title}"),
                                                "title": title[:100],
                                                "url": url,
                                                "source": "百度热搜",
                                                "score": item.get("hotScore", 0) or item.get("index", 0),
                                                "published_at": datetime.now().strftime("%Y-%m-%d"),
                                                "fetched_at": datetime.now().isoformat(),
                                                "status": "pending",
                                            })
                                            if len(items) >= top_n:
                                                break
    except Exception as e:
        logger.debug(f"百度热搜抓取失败: {e}")

    logger.info(f"百度热搜: 获取 {len(items)} 条")
    return items[:top_n]


# ===== Hacker News =====

async def fetch_hn_ai(top_n: int = 30) -> list[dict]:
    """获取 HN AI 相关热门帖子"""
    try:
        async with _get_client(timeout=15) as client:
            resp = await _fetch_with_retry(
                "https://hacker-news.firebaseio.com/v0/topstories.json",
                client
            )
            story_ids = resp.json()[:100]

            items = []
            ai_keywords = ["AI", "GPT", "LLM", "OpenAI", "Claude", "machine learning",
                          "deep learning", "neural", "transformer", "AGI", "artificial intelligence"]

            for story_id in story_ids[:50]:
                try:
                    story_resp = await _fetch_with_retry(
                        f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                        client
                    )
                    story = story_resp.json()
                    if not story:
                        continue

                    title = story.get("title", "")
                    if any(kw.lower() in title.lower() for kw in ai_keywords):
                        items.append({
                            "id": _generate_id(f"hn-{story_id}"),
                            "title": title,
                            "url": story.get("url") or f"https://news.ycombinator.com/item?id={story_id}",
                            "source": "Hacker News",
                            "score": story.get("score", 0),
                            "published_at": datetime.fromtimestamp(story.get("time", 0)).strftime("%Y-%m-%d"),
                            "fetched_at": datetime.now().isoformat(),
                            "status": "pending",
                        })

                    if len(items) >= top_n:
                        break

                except Exception as e:
                    logger.debug(f"HN story {story_id} 获取失败: {e}")

            logger.info(f"HN AI: 获取 {len(items)} 篇")
            return items

    except Exception as e:
        logger.error(f"HN 抓取失败: {e}")
        return []


# ===== GitHub Trending =====

async def fetch_github_ai(top_n: int = 25) -> list[dict]:
    """获取 GitHub AI 相关热门仓库"""
    try:
        async with _get_client(timeout=15) as client:
            resp = await _fetch_with_retry(
                "https://api.gitterapp.com/repositories",
                client,
                params={"language": "python", "since": "daily"},
                headers={"Accept": "application/json"}
            )

            if resp.status_code != 200:
                return await _fetch_github_ai_fallback(top_n)

            repos = resp.json()[:50]
            items = []

            ai_keywords = ["ai", "llm", "gpt", "chatbot", "machine-learning", "deep-learning",
                          "neural", "transformer", "openai", "langchain", "rag", "agent"]

            for repo in repos:
                name = repo.get("name", "").lower()
                desc = (repo.get("description") or "").lower()
                topics = repo.get("topics", [])

                all_text = f"{name} {desc} {' '.join(topics)}"
                if any(kw in all_text for kw in ai_keywords):
                    items.append({
                        "id": _generate_id(repo.get("url", "")),
                        "title": repo.get("name", ""),
                        "url": repo.get("url", ""),
                        "source": "GitHub Trending",
                        "score": repo.get("stars", 0),
                        "summary": repo.get("description", "")[:200] if repo.get("description") else "",
                        "published_at": datetime.now().strftime("%Y-%m-%d"),
                        "fetched_at": datetime.now().isoformat(),
                        "status": "pending",
                    })

                if len(items) >= top_n:
                    break

            logger.info(f"GitHub AI: 获取 {len(items)} 个")
            return items

    except Exception as e:
        logger.error(f"GitHub 抓取失败: {e}")
        return []


async def _fetch_github_ai_fallback(top_n: int) -> list[dict]:
    """GitHub Trending 备选方案：爬取页面"""
    try:
        async with _get_client(timeout=15) as client:
            resp = await _fetch_with_retry(
                "https://github.com/trending/python?since=daily",
                client,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            html = resp.text

            items = []
            repo_pattern = r'<h2[^>]*>.*?<a[^>]*href="(/[^"]+)"[^>]*>([^<]+)</a>'
            matches = list(re.finditer(repo_pattern, html, re.DOTALL))

            for match in matches[:top_n]:
                path, name = match.groups()
                url = f"https://github.com{path}"

                items.append({
                    "id": _generate_id(url),
                    "title": name.strip(),
                    "url": url,
                    "source": "GitHub Trending",
                    "score": 0,
                    "published_at": datetime.now().strftime("%Y-%m-%d"),
                    "fetched_at": datetime.now().isoformat(),
                    "status": "pending",
                })

            logger.info(f"GitHub fallback: 获取 {len(items)} 个")
            return items

    except Exception as e:
        logger.error(f"GitHub fallback 失败: {e}")
        return []


# ===== 知乎热榜 =====

async def fetch_zhihu_hot(top_n: int = 50) -> list[dict]:
    """获取知乎热榜 AI 相关问题"""
    try:
        async with _get_client(timeout=15) as client:
            resp = await _fetch_with_retry(
                "https://www.zhihu.com/api/v3/feed/topstory/hot-lists/total?limit=50",
                client,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "Accept": "application/json, text/plain, */*",
                    "Referer": "https://www.zhihu.com/hot",
                }
            )

            if resp.status_code != 200:
                return []

            data = resp.json()
            items = []

            ai_keywords = ["AI", "人工智能", "GPT", "ChatGPT", "大模型", "机器学习",
                          "深度学习", "神经网络", "自动驾驶", "LLM", "OpenAI", "Claude",
                          "智能", "机器人", "算法"]

            for item in data.get("data", [])[:100]:
                target = item.get("target", {})
                title = target.get("title", "")
                excerpt = target.get("excerpt", "")
                url = target.get("url", "")

                all_text = f"{title} {excerpt}"
                if any(kw in all_text for kw in ai_keywords):
                    items.append({
                        "id": _generate_id(url or f"zhihu-{title}"),
                        "title": title,
                        "url": url,
                        "source": "知乎热榜",
                        "score": item.get("detail_text", "").replace("热度 ", ""),
                        "summary": excerpt[:200] if excerpt else "",
                        "published_at": datetime.now().strftime("%Y-%m-%d"),
                        "fetched_at": datetime.now().isoformat(),
                        "status": "pending",
                    })

                if len(items) >= top_n:
                    break

            logger.info(f"知乎 AI: 获取 {len(items)} 篇")
            return items

    except Exception as e:
        logger.error(f"知乎抓取失败: {e}")
        return []


# ===== V2EX 热门 =====

async def fetch_v2ex_hot(top_n: int = 20) -> list[dict]:
    """获取 V2EX 热门话题"""
    try:
        async with _get_client(timeout=15) as client:
            resp = await client.get(
                "https://www.v2ex.com/api/topics/hot.json",
                headers={"User-Agent": "Mozilla/5.0"}
            )

            if resp.status_code != 200:
                return []

            topics = resp.json()
            items = []

            ai_keywords = ["AI", "GPT", "LLM", "OpenAI", "Claude", "大模型",
                          "机器学习", "深度学习", "ChatGPT", "编程", "程序员"]

            for topic in topics[:top_n]:
                title = topic.get("title", "")
                node = topic.get("node", {}).get("name", "")

                all_text = f"{title} {node}"
                if any(kw in all_text for kw in ai_keywords):
                    items.append({
                        "id": _generate_id(f"v2ex-{topic.get('id')}"),
                        "title": title,
                        "url": topic.get("url", ""),
                        "source": "V2EX",
                        "score": topic.get("replies", 0),
                        "summary": topic.get("content", "")[:200] if topic.get("content") else "",
                        "published_at": datetime.now().strftime("%Y-%m-%d"),
                        "fetched_at": datetime.now().isoformat(),
                        "status": "pending",
                    })

            logger.info(f"V2EX: 获取 {len(items)} 篇")
            return items

    except Exception as e:
        logger.error(f"V2EX 抓取失败: {e}")
        return []


# ===== 微博热搜 AI =====

async def fetch_weibo_hot(top_n: int = 20) -> list[dict]:
    """获取微博热搜（通过百度热搜筛选科技/AI相关）"""
    items = []

    try:
        async with _get_client(timeout=15) as client:
            resp = await _fetch_with_retry(
                "https://top.baidu.com/api/board?platform=wise&tab=realtime",
                client,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            if resp.status_code == 200:
                data = resp.json()
                cards = data.get("data", {}).get("cards", [])
                ai_keywords = ["AI", "人工智能", "GPT", "ChatGPT", "大模型", "机器学习",
                              "智能", "机器人", "算法", "科技", "互联网", "手机", "苹果", "华为",
                              "DeepSeek", "宇树", "荣耀", "程序员"]
                for card in cards:
                    content_list = card.get("content", [])
                    if isinstance(content_list, list):
                        for content_item in content_list:
                            if isinstance(content_item, dict):
                                hot_list = content_item.get("content", [])
                                if isinstance(hot_list, list):
                                    for item in hot_list[:top_n * 2]:
                                        title = item.get("word", "") or item.get("title", "")
                                        url = item.get("url", "") or item.get("rawUrl", "")
                                        # 筛选科技/AI相关
                                        if title and any(kw in title for kw in ai_keywords):
                                            items.append({
                                                "id": _generate_id(url or f"weibo-{title}"),
                                                "title": title[:100],
                                                "url": url,
                                                "source": "微博热搜",
                                                "score": item.get("hotScore", 0) or item.get("index", 0),
                                                "published_at": datetime.now().strftime("%Y-%m-%d"),
                                                "fetched_at": datetime.now().isoformat(),
                                                "status": "pending",
                                            })
                                            if len(items) >= top_n:
                                                break
    except Exception as e:
        logger.error(f"微博抓取失败: {e}")

    logger.info(f"微博热搜: 获取 {len(items)} 条")
    return items


# ===== ProductHunt =====

async def fetch_producthunt_ai(top_n: int = 10) -> list[dict]:
    """获取 ProductHunt AI 产品（通过 GitHub Trending 筛选 AI 产品）"""
    items = []

    try:
        async with _get_client(timeout=15) as client:
            resp = await _fetch_with_retry(
                "https://api.gitterapp.com/repositories",
                client,
                params={"language": "python", "since": "daily"},
                headers={"Accept": "application/json"}
            )

            if resp.status_code != 200:
                return items

            repos = resp.json()[:30]
            ai_keywords = ["ai", "llm", "gpt", "chatbot", "machine-learning", "deep-learning",
                          "neural", "transformer", "openai", "langchain", "rag", "agent", "claude"]

            for repo in repos:
                name = repo.get("name", "").lower()
                desc = (repo.get("description") or "").lower()
                topics = repo.get("topics", [])

                all_text = f"{name} {desc} {' '.join(topics)}"
                if any(kw in all_text for kw in ai_keywords):
                    items.append({
                        "id": _generate_id(repo.get("url", "")),
                        "title": repo.get("name", ""),
                        "url": repo.get("url", ""),
                        "source": "ProductHunt",
                        "score": repo.get("stars", 0),
                        "summary": repo.get("description", "")[:200] if repo.get("description") else "",
                        "published_at": datetime.now().strftime("%Y-%m-%d"),
                        "fetched_at": datetime.now().isoformat(),
                        "status": "pending",
                    })

                if len(items) >= top_n:
                    break

    except Exception as e:
        logger.error(f"ProductHunt 抓取失败: {e}")

    logger.info(f"ProductHunt AI: 获取 {len(items)} 个")
    return items


# ===== 36氪 AI =====

async def fetch_36kr_ai(top_n: int = 20) -> list[dict]:
    """获取 36氪 AI 相关文章"""
    try:
        async with _get_client(timeout=15) as client:
            # 36氪快讯 API
            resp = await _fetch_with_retry(
                "https://36kr.com/api/newsflash",
                client,
                params={"per_page": 50},
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Accept": "application/json",
                }
            )

            if resp.status_code != 200:
                return []

            data = resp.json()
            items = []

            ai_keywords = ["AI", "人工智能", "GPT", "ChatGPT", "大模型", "LLM",
                          "OpenAI", "机器学习", "智能", "机器人", "自动驾驶"]

            for item in data.get("data", {}).get("items", [])[:top_n]:
                title = item.get("title", "")
                desc = item.get("description", "")
                url = item.get("news_url", "") or item.get("source_url", "")

                all_text = f"{title} {desc}"
                if not any(kw in all_text for kw in ai_keywords):
                    continue

                items.append({
                    "id": _generate_id(url or f"36kr-{title}"),
                    "title": title[:100],
                    "url": url,
                    "source": "36氪",
                    "score": 0,
                    "summary": desc[:200] if desc else "",
                    "published_at": datetime.now().strftime("%Y-%m-%d"),
                    "fetched_at": datetime.now().isoformat(),
                    "status": "pending",
                })

            logger.info(f"36氪 AI: 获取 {len(items)} 篇")
            return items

    except Exception as e:
        logger.error(f"36氪抓取失败: {e}")
        return []


# ===== 量子位 =====

async def fetch_qbit_ai(top_n: int = 20) -> list[dict]:
    """获取量子位热门文章"""
    try:
        async with _get_client(timeout=15) as client:
            # 量子位 API
            resp = await _fetch_with_retry(
                "https://www.qbitai.com/wp-json/wp/v2/posts",
                client,
                params={"per_page": 30, "_fields": "title,link,date"},
                headers={"User-Agent": "Mozilla/5.0"}
            )

            if resp.status_code != 200:
                return []

            posts = resp.json()
            items = []

            for post in posts[:top_n]:
                title = post.get("title", {}).get("rendered", "")
                url = post.get("link", "")

                if not title:
                    continue

                items.append({
                    "id": _generate_id(url or f"qbit-{title}"),
                    "title": title[:100],
                    "url": url,
                    "source": "量子位",
                    "score": 0,
                    "published_at": datetime.now().strftime("%Y-%m-%d"),
                    "fetched_at": datetime.now().isoformat(),
                    "status": "pending",
                })

            logger.info(f"量子位: 获取 {len(items)} 篇")
            return items

    except Exception as e:
        logger.error(f"量子位抓取失败: {e}")
        return []


# ===== 机器之心 =====

async def fetch_jqxx_ai(top_n: int = 20) -> list[dict]:
    """获取机器之心热门文章"""
    try:
        async with _get_client(timeout=15) as client:
            # 直接请求首页获取文章列表
            resp = await _fetch_with_retry(
                "https://www.jiqizhixin.com/",
                client,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            )

            if resp.status_code != 200:
                return []

            html = resp.text
            items = []

            # 解析文章链接
            article_pattern = r'<a[^>]*href="(/articles/[^"]+)"[^>]*>([^<]+)</a>'
            matches = list(re.finditer(article_pattern, html))

            seen_urls = set()
            for match in matches[:top_n * 2]:
                path, title = match.groups()
                url = f"https://www.jiqizhixin.com{path}"

                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title = title.strip()
                if len(title) < 5:
                    continue

                items.append({
                    "id": _generate_id(url),
                    "title": title[:100],
                    "url": url,
                    "source": "机器之心",
                    "score": 0,
                    "published_at": datetime.now().strftime("%Y-%m-%d"),
                    "fetched_at": datetime.now().isoformat(),
                    "status": "pending",
                })

                if len(items) >= top_n:
                    break

            logger.info(f"机器之心: 获取 {len(items)} 篇")
            return items

    except Exception as e:
        logger.error(f"机器之心抓取失败: {e}")
        return []


# ===== InfoQ 热门文章 =====

async def fetch_infoq_hot(top_n: int = 15) -> list[dict]:
    """获取 InfoQ 热门技术文章（高质量技术媒体）"""
    items = []
    try:
        async with _get_client(timeout=15) as client:
            resp = await _fetch_with_retry(
                "https://www.infoq.cn/api/v1/article/list",
                client,
                method="POST",
                json={"size": top_n, "type": 1},
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Content-Type": "application/json",
                    "Origin": "https://www.infoq.cn",
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", [])[:top_n]:
                    title = item.get("title", "")
                    uuid = item.get("uuid", "")
                    url = f"https://www.infoq.cn/article/{uuid}" if uuid else ""
                    summary = item.get("summary", "") or item.get("content", "")[:200]
                    if title:
                        items.append({
                            "id": _generate_id(url or f"infoq-{title}"),
                            "title": title[:100],
                            "url": url,
                            "source": "InfoQ",
                            "score": item.get("views", 0) or item.get("likeCount", 0) or 0,
                            "summary": summary[:200] if summary else "",
                            "published_at": datetime.now().strftime("%Y-%m-%d"),
                            "fetched_at": datetime.now().isoformat(),
                            "status": "pending",
                        })
    except Exception as e:
        logger.debug(f"InfoQ抓取失败: {e}")

    logger.info(f"InfoQ: 获取 {len(items)} 条")
    return items


# ===== IT之家热榜 =====

async def fetch_ithome_hot(top_n: int = 20) -> list[dict]:
    """获取 IT之家热榜"""
    items = []
    try:
        async with _get_client(timeout=15) as client:
            resp = await _fetch_with_retry(
                "https://api.ithome.com/json/news/news.json",
                client,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("newslist", [])[:top_n]:
                    title = item.get("title", "")
                    url = item.get("url", "")
                    if title:
                        items.append({
                            "id": _generate_id(url or f"ithome-{title}"),
                            "title": title[:100],
                            "url": url,
                            "source": "IT之家",
                            "score": item.get("commentCount", 0),
                            "summary": item.get("description", "")[:200] if item.get("description") else "",
                            "published_at": datetime.now().strftime("%Y-%m-%d"),
                            "fetched_at": datetime.now().isoformat(),
                            "status": "pending",
                        })
    except Exception as e:
        logger.error(f"IT之家抓取失败: {e}")

    logger.info(f"IT之家: 获取 {len(items)} 条")
    return items


# ===== 掘金热榜 =====

async def fetch_juejin_hot(top_n: int = 20) -> list[dict]:
    """获取掘金热榜"""
    items = []
    try:
        async with _get_client(timeout=15) as client:
            # 掘金需要 POST 请求
            resp = await client.post(
                "https://api.juejin.cn/recommend_api/v1/article/recommend_all_feed",
                json={"id_type": 2, "sort_type": 200, "cursor": "0", "limit": top_n},
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Content-Type": "application/json",
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", [])[:top_n]:
                    article_info = item.get("item_info", {}).get("article_info", {})
                    title = article_info.get("title", "")
                    article_id = article_info.get("article_id", "")
                    url = f"https://juejin.cn/post/{article_id}"
                    if title:
                        items.append({
                            "id": _generate_id(url),
                            "title": title[:100],
                            "url": url,
                            "source": "掘金",
                            "score": article_info.get("view_count", 0),
                            "summary": article_info.get("brief_content", "")[:200] if article_info.get("brief_content") else "",
                            "published_at": datetime.now().strftime("%Y-%m-%d"),
                            "fetched_at": datetime.now().isoformat(),
                            "status": "pending",
                        })
    except Exception as e:
        logger.error(f"掘金抓取失败: {e}")

    logger.info(f"掘金: 获取 {len(items)} 条")
    return items


# ===== 虎嗅热榜 =====

async def fetch_huxiu_hot(top_n: int = 20) -> list[dict]:
    """获取虎嗅热榜"""
    items = []
    try:
        async with _get_client(timeout=15) as client:
            resp = await _fetch_with_retry(
                "https://www.huxiu.com/v2-action/article/tops",
                client,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Referer": "https://www.huxiu.com/"
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", [])[:top_n]:
                    title = item.get("title", "")
                    url = f"https://www.huxiu.com/article/{item.get('aid', '')}.html"
                    if title:
                        items.append({
                            "id": _generate_id(url),
                            "title": title[:100],
                            "url": url,
                            "source": "虎嗅",
                            "score": item.get("count_view", 0),
                            "summary": item.get("summary", "")[:200] if item.get("summary") else "",
                            "published_at": datetime.now().strftime("%Y-%m-%d"),
                            "fetched_at": datetime.now().isoformat(),
                            "status": "pending",
                        })
    except Exception as e:
        logger.error(f"虎嗅抓取失败: {e}")

    logger.info(f"虎嗅: 获取 {len(items)} 条")
    return items


# ===== 哔哩哔哩热榜 =====

async def fetch_bilibili_hot(top_n: int = 10) -> list[dict]:
    """获取哔哩哔哩科技区热榜（关键词预筛选）

    优化策略：
    1. 只抓取标题包含AI/技术关键词的内容
    2. 减少数量，提升质量
    """
    items = []

    # B站内容预筛选关键词（必须匹配才入库）
    REQUIRED_KEYWORDS = [
        "AI", "人工智能", "GPT", "ChatGPT", "Claude", "DeepSeek", "OpenAI",
        "大模型", "LLM", "机器学习", "深度学习", "神经网络",
        "Agent", "智能体", "RAG", "Prompt", "提示词",
        "编程", "代码", "Python", "算法", "开发",
        "Cursor", "Copilot", "Claude Code",
    ]

    try:
        async with _get_client(timeout=15) as client:
            resp = await _fetch_with_retry(
                "https://api.bilibili.com/x/web-interface/ranking/v2?rid=188&type=all",
                client,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Referer": "https://www.bilibili.com/"
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                rank_list = data.get("data", {}).get("list", [])
                for item in rank_list:
                    title = item.get("title", "")
                    # 关键词预筛选
                    if not any(kw.lower() in title.lower() for kw in REQUIRED_KEYWORDS):
                        continue

                    bvid = item.get("bvid", "")
                    url = f"https://www.bilibili.com/video/{bvid}"
                    if title:
                        items.append({
                            "id": _generate_id(url),
                            "title": title[:100],
                            "url": url,
                            "source": "哔哩哔哩",
                            "score": item.get("stat", {}).get("view", 0),
                            "summary": item.get("desc", "")[:200] if item.get("desc") else "",
                            "published_at": datetime.now().strftime("%Y-%m-%d"),
                            "fetched_at": datetime.now().isoformat(),
                            "status": "pending",
                        })
                    if len(items) >= top_n:
                        break
    except Exception as e:
        logger.error(f"哔哩哔哩抓取失败: {e}")

    logger.info(f"哔哩哔哩: 获取 {len(items)} 条（已筛选）")
    return items


# ===== 微信公众号热榜（通过西瓜数据） =====

async def fetch_wechat_articles(top_n: int = 20) -> list[dict]:
    """获取高质量 AI 内容（替代低质量微信热文）"""
    items = []

    # AI 相关关键词（用于筛选高质量内容）
    ai_keywords = [
        "AI", "人工智能", "GPT", "ChatGPT", "Claude", "DeepSeek",
        "大模型", "LLM", "AGI", "机器学习", "深度学习",
        "Agent", "RAG", "MCP", "向量数据库", "Embedding",
        "提示词", "Prompt", "微调", "RAG", "知识库",
        "智能体", "多模态", "推理", "训练", "算力",
        "OpenAI", "Anthropic", "Google", "微软", "字节",
    ]

    # 方案1: 机器之心（高质量 AI 垂直媒体）
    try:
        async with _get_client(timeout=15) as client:
            resp = await _fetch_with_retry(
                "https://api.jiqizhixin.com/articles",
                client,
                params={"page": 1, "per_page": top_n},
                headers={"User-Agent": "Mozilla/5.0"}
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", [])[:top_n]:
                    title = item.get("title", "")
                    url = item.get("url", "") or f"https://www.jiqizhixin.com/articles/{item.get('id', '')}"
                    summary = item.get("summary", "") or item.get("description", "")
                    if title and len(title) >= 5:
                        items.append({
                            "id": _generate_id(url),
                            "title": title[:100],
                            "url": url,
                            "source": "机器之心",
                            "score": item.get("view_count", 0) or 0,
                            "summary": summary[:200] if summary else "",
                            "published_at": datetime.now().strftime("%Y-%m-%d"),
                            "fetched_at": datetime.now().isoformat(),
                            "status": "pending",
                        })
    except Exception as e:
        logger.debug(f"机器之心抓取失败: {e}")

    # 方案2: 量子位（AI 垂直媒体）
    if len(items) < top_n:
        try:
            async with _get_client(timeout=15) as client:
                resp = await _fetch_with_retry(
                    "https://www.qbitai.com/wp-json/wp/v2/posts",
                    client,
                    params={"per_page": top_n, "_fields": "id,title,link,date"},
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data[:top_n]:
                        title = item.get("title", {}).get("rendered", "")
                        url = item.get("link", "")
                        if title and len(title) >= 5:
                            items.append({
                                "id": _generate_id(url),
                                "title": title[:100],
                                "url": url,
                                "source": "量子位",
                                "score": 0,
                                "published_at": item.get("date", "")[:10] or datetime.now().strftime("%Y-%m-%d"),
                                "fetched_at": datetime.now().isoformat(),
                                "status": "pending",
                            })
        except Exception as e:
            logger.debug(f"量子位抓取失败: {e}")

    logger.info(f"AI 垂直媒体: 获取 {len(items)} 条")
    return items[:top_n]


# ===== 少数派热榜 =====

async def fetch_sspai_hot(top_n: int = 15) -> list[dict]:
    """获取少数派热榜"""
    items = []
    try:
        async with _get_client(timeout=15) as client:
            # 少数派首页文章
            resp = await _fetch_with_retry(
                "https://sspai.com/api/v1/article/index/page/1/limit/20",
                client,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Referer": "https://sspai.com/"
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", [])[:top_n]:
                    title = item.get("title", "")
                    aid = item.get("id", "")
                    url = f"https://sspai.com/post/{aid}"
                    if title:
                        items.append({
                            "id": _generate_id(url),
                            "title": title[:100],
                            "url": url,
                            "source": "少数派",
                            "score": item.get("like_count", 0),
                            "summary": item.get("summary", "")[:200] if item.get("summary") else "",
                            "published_at": datetime.now().strftime("%Y-%m-%d"),
                            "fetched_at": datetime.now().isoformat(),
                            "status": "pending",
                        })
    except Exception as e:
        logger.error(f"少数派抓取失败: {e}")

    logger.info(f"少数派: 获取 {len(items)} 条")
    return items


# ===== 统一抓取 =====

async def fetch_all_sources() -> list[dict]:
    """基于知识库比例的分层抓取

    动态读取知识库 taxonomy.yaml，根据实际内容分布调整抓取比例。
    """
    logger.info("开始基于知识库比例抓取热榜...")

    # 动态获取知识库分类比例
    kb_distribution = _get_kb_distribution()

    # 各分类对应的关键词
    CATEGORY_KEYWORDS = {
        "AI技术": [
            "AI", "人工智能", "GPT", "ChatGPT", "Claude", "DeepSeek", "OpenAI", "Anthropic",
            "大模型", "LLM", "AGI", "机器学习", "深度学习", "神经网络", "Transformer",
            "Agent", "智能体", "RAG", "MCP", "向量数据库", "Embedding", "知识库",
            "提示词", "Prompt", "微调", "训练", "推理", "算力", "GPU", "CUDA",
        ],
        "产品设计": [
            "产品经理", "产品设计", "交互设计", "用户体验", "UX", "UI", "原型",
            "需求分析", "用户研究", "可用性", "设计原则", "尼尔森",
        ],
        "商业趋势": [
            "创业", "投资", "融资", "商业模式", "市场分析", "行业趋势", "商业化",
            "变现", "增长", "运营", "营销", "品牌", "出海", "电商",
        ],
        "职业发展": [
            "面试", "简历", "职场", "职业规划", "跳槽", "薪资", "晋升",
            "技能", "学习", "培训", "招聘", "求职",
        ],
        "开发工具": [
            "代码", "编程", "开发", "框架", "库", "工具", "IDE", "Git",
            "Python", "JavaScript", "Go", "Rust", "API", "SDK",
        ],
        "互联网产品": [
            "小红书", "抖音", "微信", "B站", "知乎", "微博", "快手",
            "社交", "内容", "社区", "平台", "流量",
        ],
    }

    tasks = [
        # P0 高质量AI内容源
        fetch_wechat_articles(35),      # 机器之心+量子位
        fetch_juejin_hot(35),           # 掘金（技术+AI，质量最高）
        fetch_hn_ai(30),                # Hacker News
        # P1 技术源
        fetch_github_ai(25),            # GitHub Trending
        # P2 商业/市场
        fetch_36kr_ai(15),              # 36氪
        # 注意：B站内容质量与知识库需求严重不匹配，已移除
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_items = []
    for result in results:
        if isinstance(result, list):
            all_items.extend(result)
        elif isinstance(result, Exception):
            logger.error(f"抓取异常: {result}")

    all_items = [_score_and_classify_item(item, CATEGORY_KEYWORDS) for item in all_items]
    selected = _select_by_distribution(all_items, kb_distribution, total_limit=100)

    logger.info(f"热榜抓取完成: 原始 {len(all_items)} 条 → 筛选 {len(selected)} 条")
    return selected


def _get_kb_distribution() -> dict:
    """从 taxonomy.yaml 动态读取知识库分类比例

    知识库分类 → 抓取分类映射：
    - AI技术发展与应用 → AI技术
    - 产品设计方法 → 产品设计
    - 商业趋势与市场分析 → 商业趋势
    - 职业发展与规划 → 职业发展
    - 开发工具与实践 → 开发工具
    - 互联网产品与平台 → 互联网产品
    - 其他分类 → 其他
    """
    import yaml
    from pathlib import Path

    taxonomy_path = Path(__file__).parent.parent / "wiki" / "_taxonomy.yaml"

    # 知识库分类到抓取分类的映射
    CATEGORY_MAP = {
        "AI技术发展与应用": "AI技术",
        "产品设计方法": "产品设计",
        "商业趋势与市场分析": "商业趋势",
        "职业发展与规划": "职业发展",
        "开发工具与实践": "开发工具",
        "互联网产品与平台": "互联网产品",
        "心理学与行为科学": "其他",
        "AIGC内容创作": "AI技术",
        "生产力工具与方法": "开发工具",
        "项目文档": "其他",
        "数据分析与统计": "开发工具",
        "产品人物与思想": "产品设计",
    }

    try:
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"读取 taxonomy 失败，使用默认比例: {e}")
        return {"AI技术": 0.40, "产品设计": 0.11, "商业趋势": 0.10,
                "职业发展": 0.07, "开发工具": 0.06, "互联网产品": 0.06, "其他": 0.20}

    def count_pages(node: dict) -> int:
        """递归统计分类下的页面数"""
        count = len(node.get("pages", []))
        for child in node.get("children", {}).values():
            count += count_pages(child)
        return count

    categories = data.get("categories", {})
    distribution = {}
    total = 0

    # 统计各分类文章数
    for kb_cat, node in categories.items():
        count = count_pages(node)
        crawl_cat = CATEGORY_MAP.get(kb_cat, "其他")
        distribution[crawl_cat] = distribution.get(crawl_cat, 0) + count
        total += count

    if total == 0:
        return {"AI技术": 0.40, "产品设计": 0.11, "商业趋势": 0.10,
                "职业发展": 0.07, "开发工具": 0.06, "互联网产品": 0.06, "其他": 0.20}

    # 转换为比例
    for cat in distribution:
        distribution[cat] = distribution[cat] / total

    # 确保所有分类都有值
    default_cats = ["AI技术", "产品设计", "商业趋势", "职业发展", "开发工具", "互联网产品", "其他"]
    for cat in default_cats:
        if cat not in distribution:
            distribution[cat] = 0.05  # 默认 5%

    # 归一化
    total_ratio = sum(distribution.values())
    for cat in distribution:
        distribution[cat] = distribution[cat] / total_ratio

    logger.info(f"知识库比例: {distribution}")
    return distribution


def _score_and_classify_item(item: dict, category_keywords: dict) -> dict:
    """计算质量评分并分类"""
    title = item.get("title", "").lower()
    summary = (item.get("summary", "") or "").lower()
    text = f"{title} {summary}"
    source = item.get("source", "")

    # 高质量来源权重
    quality_sources = {
        "机器之心": 10, "量子位": 9, "Hacker News": 8, "GitHub Trending": 8,
        "掘金": 7, "少数派": 7, "36氪": 6,
    }

    score = 0
    categories = []

    # 计算各分类匹配度
    for cat, keywords in category_keywords.items():
        matches = sum(1 for kw in keywords if kw.lower() in text)
        if matches > 0:
            categories.append((cat, matches))
            score += min(matches * 5, 40)

    # 排序取主要分类
    categories.sort(key=lambda x: -x[1])
    item["category"] = categories[0][0] if categories else "其他"
    item["category_score"] = categories[0][1] if categories else 0

    # 来源权重
    score += quality_sources.get(source, 5)

    item["quality_score"] = score
    return item


def _select_by_distribution(items: list[dict], distribution: dict, total_limit: int = 100) -> list[dict]:
    """按知识库比例筛选内容"""
    # 按分类分组
    by_category = {}
    for item in items:
        cat = item.get("category", "其他")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(item)

    # 按质量分排序各组
    for cat in by_category:
        by_category[cat].sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    # 按比例选取
    selected = []
    for cat, ratio in distribution.items():
        limit = int(total_limit * ratio)
        cat_items = by_category.get(cat, [])[:limit]
        selected.extend(cat_items)

    # 补足差额
    if len(selected) < total_limit:
        remaining = [i for i in items if i not in selected]
        remaining.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        selected.extend(remaining[:total_limit - len(selected)])

    # 最终排序
    selected.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
    return selected[:total_limit]
