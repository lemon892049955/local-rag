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
_handlers_registered = False


def get_bot(use_current_user=True):
    """获取全局 Bot 实例"""
    global _bot
    if _bot is None:
        from wechat_bot import Bot
        # 尝试从文件读取 token
        token = _load_saved_token()
        if token:
            _bot = Bot(token=token)
            logger.info("[iLink] 使用保存的 token 初始化 Bot")
        else:
            _bot = Bot(use_current_user=use_current_user)
    return _bot


def _load_saved_token() -> Optional[str]:
    """从文件加载保存的 token"""
    import json
    from pathlib import Path
    token_file = Path("/app/.ilink_token.json")
    if not token_file.exists():
        # 也检查当前目录
        token_file = Path(".ilink_token.json")
    if token_file.exists():
        try:
            data = json.loads(token_file.read_text())
            token = data.get("bot_token", "")
            if token:
                logger.info(f"[iLink] 从 {token_file} 加载 token")
                return token
        except Exception as e:
            logger.warning(f"[iLink] 读取 token 文件失败: {e}")
    return None


def _register_handlers(bot):
    """注册消息处理器（只注册一次）"""
    global _handlers_registered
    if _handlers_registered:
        return
    from wechat_bot import Filter

    @bot.on_message(Filter.text())
    async def on_text(ctx):
        """处理所有文本消息（含链接）"""
        text = (ctx.text or "").strip()
        if not text:
            return
        user_id = ctx.from_user_id
        logger.info(f"[iLink] 收到消息: from={user_id}, text={text[:100]}")
        await _handle_text_message(ctx, user_id, text)

    _handlers_registered = True
    logger.info("[iLink] 消息处理器已注册")


async def start_bot():
    """启动 iLink Bot（后台长轮询）

    如果有缓存的登录态则自动启动，否则跳过等待手动登录。
    """
    global _bot_running
    if _bot_running:
        logger.info("iLink Bot 已在运行中")
        return True

    try:
        from wechat_bot import Bot
        bot = get_bot(use_current_user=True)
        _register_handlers(bot)

        # 启动 Bot（非阻塞方式）
        asyncio.create_task(_run_bot(bot))
        _bot_running = True
        logger.info("iLink Bot 启动成功，等待消息...")
        return True

    except ImportError:
        logger.warning("wechat-ilink-bot 未安装，iLink Bot 功能不可用")
        return False
    except Exception as e:
        logger.warning(f"iLink Bot 启动跳过（需要先登录）: {e}")
        return False


async def login_bot():
    """扫码登录并启动 Bot

    返回登录结果，包含 account_id 等信息。
    调用方需要引导用户扫描二维码。
    """
    global _bot, _bot_running
    try:
        from wechat_bot import Bot

        # 重置状态
        if _bot:
            try:
                await _bot.stop()
            except Exception:
                pass
        _bot = Bot(use_current_user=False)
        _bot_running = False

        bot = _bot
        _register_handlers(bot)

        # 登录（会在终端打印二维码）
        result = await bot.login()
        logger.info(f"[iLink] 登录成功: account_id={result.account_id}")

        # 启动消息监听
        asyncio.create_task(_run_bot(bot))
        _bot_running = True

        return {
            "success": True,
            "account_id": result.account_id,
            "user_id": result.user_id,
        }
    except Exception as e:
        logger.error(f"[iLink] 登录失败: {e}")
        return {"success": False, "error": str(e)}


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
