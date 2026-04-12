"""微信 iLink Bot 核心模块

基于 wechat-ilink-bot SDK，实现微信个人号的消息收发。
替代企微自建应用方案，更轻量、更直接。
"""

import asyncio
import logging
import re
from typing import Optional, Callable, Awaitable

logger = logging.getLogger(__name__)

# 全局 Bot 实例
_bot = None
_bot_running = False


def get_bot():
    """获取全局 Bot 实例"""
    global _bot
    if _bot is None:
        from wechat_bot import Bot
        _bot = Bot(use_current_user=True)  # 优先使用缓存的登录态
    return _bot


async def start_bot():
    """启动 iLink Bot（后台长轮询）

    在 FastAPI startup 事件中调用。
    注册消息 handler，开始监听微信消息。
    """
    global _bot_running
    if _bot_running:
        logger.info("iLink Bot 已在运行中")
        return

    try:
        from wechat_bot import Bot, Filter
        bot = get_bot()

        # ===== 注册消息处理器 =====

        @bot.on_message(Filter.text())
        async def on_text(ctx):
            """处理文本消息"""
            text = ctx.text.strip()
            user_id = ctx.from_user_id
            logger.info(f"[iLink] 收到文本消息: from={user_id}, text={text[:100]}")
            await _handle_text_message(ctx, user_id, text)

        @bot.on_message(Filter.link())
        async def on_link(ctx):
            """处理链接消息（微信分享的文章等）"""
            url = ctx.url or ""
            title = ctx.title or ""
            user_id = ctx.from_user_id
            logger.info(f"[iLink] 收到链接消息: from={user_id}, title={title[:50]}")
            if url:
                await ctx.reply("📥 已收到链接，正在入库...")
                asyncio.create_task(_process_ingest(ctx, user_id, url))

        # 启动 Bot（非阻塞方式）
        asyncio.create_task(_run_bot(bot))
        _bot_running = True
        logger.info("iLink Bot 启动成功，等待消息...")

    except ImportError:
        logger.warning("wechat-ilink-bot 未安装，iLink Bot 功能不可用。pip install wechat-ilink-bot")
    except Exception as e:
        logger.error(f"iLink Bot 启动失败: {e}")


async def _run_bot(bot):
    """后台运行 Bot 长轮询"""
    try:
        await bot.start()
    except Exception as e:
        global _bot_running
        _bot_running = False
        logger.error(f"iLink Bot 运行异常: {e}")


# ===== 消息处理逻辑 =====

def _extract_urls(text: str) -> list:
    """从文本中提取 URL"""
    url_pattern = r'https?://[^\s<>"\'，。！？、）\]】}]+'
    urls = re.findall(url_pattern, text)
    cleaned = []
    for u in urls:
        u = u.rstrip(".,;:!?)]}>")
        if len(u) > 10:
            cleaned.append(u)
    return cleaned


async def _handle_text_message(ctx, user_id: str, text: str):
    """处理文本消息 — 意图识别分发"""
    # 帮助命令
    if text in ("帮助", "help", "/help", "?", "？"):
        await ctx.reply(
            "🤖 BuddyKnow 知识库助手\n\n"
            "📥 发链接 → 自动入库\n"
            "🔍 发文字 → 搜索知识库\n"
            "📊 发「统计」→ 查看库存\n"
            "❓ 发「帮助」→ 查看此菜单"
        )
        return

    # 统计命令
    if text in ("统计", "状态", "stats"):
        await _handle_stats(ctx)
        return

    # 检查是否包含 URL → 入库
    urls = _extract_urls(text)
    if urls:
        await ctx.reply(f"📥 已收到 {len(urls)} 条链接，正在处理中...")
        for url in urls:
            asyncio.create_task(_process_ingest(ctx, user_id, url))
        return

    # 否则 → AI 助手对话
    await ctx.reply("🤔 思考中...")
    asyncio.create_task(_process_assistant_chat(ctx, user_id, text))


async def _handle_stats(ctx):
    """处理统计命令"""
    try:
        from config import DATA_DIR, WIKI_DIR
        file_count = len(list(DATA_DIR.glob("*.md")))
        wiki_count = sum(1 for _ in WIKI_DIR.glob("**/*.md") if not _.name.startswith("_"))
        await ctx.reply(
            f"📊 BuddyKnow 知识库统计\n\n"
            f"📄 文章数: {file_count}\n"
            f"📖 Wiki 页面: {wiki_count}"
        )
    except Exception as e:
        await ctx.reply(f"❌ 获取统计失败: {e}")


async def _process_ingest(ctx, user_id: str, url: str):
    """后台异步入库"""
    try:
        from services.ingest_pipeline import ingest_url
        result = await ingest_url(url)

        if result.get("duplicate"):
            await ctx.reply(f"⚠️ 该链接已入库，无需重复添加")
            return

        if result.get("success"):
            title = result.get("title", "未知")
            tags = result.get("tags", [])
            tags_str = "、".join(tags[:5]) if tags else "待分类"
            await ctx.reply(
                f"✅ 入库成功\n\n"
                f"📝 {title}\n"
                f"🏷️ {tags_str}"
            )
        else:
            error = result.get("error", "未知错误")
            await ctx.reply(f"❌ 入库失败: {error[:100]}")

    except Exception as e:
        logger.error(f"[iLink] 入库失败: {url} - {e}")
        await ctx.reply(f"❌ 入库出错: {str(e)[:100]}")


async def _process_assistant_chat(ctx, user_id: str, message: str):
    """后台 AI 助手对话"""
    try:
        from assistant.intent import detect_intent
        from assistant.chat_engine import chat_once

        session_id = f"ilink_{user_id}"
        intent_result = detect_intent(message)

        # 构建上下文
        context = ""
        if intent_result["intent"] == "search":
            try:
                from retrieval.hybrid_searcher import HybridSearcher
                from retrieval.indexer import VectorIndexer
                searcher = HybridSearcher(indexer=VectorIndexer())
                result = await searcher.search(query=message, top_k=3)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                source_list = "\n".join(
                    f"  [{i+1}] {s.get('title', '未知')}" for i, s in enumerate(sources)
                )
                context = f"知识库搜索结果:\n\n{answer}\n\n参考来源:\n{source_list}"
            except Exception:
                pass
        elif intent_result["intent"] == "stats":
            from config import DATA_DIR, WIKI_DIR
            file_count = len(list(DATA_DIR.glob("*.md")))
            wiki_count = sum(1 for _ in WIKI_DIR.glob("**/*.md") if not _.name.startswith("_"))
            context = f"系统状态: {file_count} 篇文章, {wiki_count} 个 Wiki 页面"

        reply = await chat_once(session_id, message, context)
        if not reply:
            reply = "抱歉，暂时无法回复"

        # 微信消息没有严格长度限制，但太长体验不好
        if len(reply) > 2000:
            reply = reply[:1997] + "..."

        await ctx.reply(reply)

    except Exception as e:
        logger.error(f"[iLink] 助手对话失败: {message[:50]} - {e}")
        # 降级到纯搜索
        await _process_search(ctx, user_id, message)


async def _process_search(ctx, user_id: str, query: str):
    """降级搜索"""
    try:
        from retrieval.hybrid_searcher import HybridSearcher
        searcher = HybridSearcher()
        result = await searcher.search(query=query, top_k=5)

        answer = result.get("answer", "未找到相关内容")
        sources = result.get("sources", [])

        reply = f"📝 {answer}\n"
        if sources:
            reply += "\n📚 参考来源:\n"
            for i, src in enumerate(sources, 1):
                reply += f"  [{i}] {src.get('title', '未知')}\n"

        if len(reply) > 2000:
            reply = reply[:1997] + "..."

        await ctx.reply(reply)
    except Exception as e:
        logger.error(f"[iLink] 搜索失败: {query} - {e}")
        await ctx.reply(f"❌ 搜索出错: {str(e)[:100]}")
