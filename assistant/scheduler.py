"""定时任务调度框架 + 主动推送系统

基于 asyncio 的轻量后台调度器，定期生成推送内容:
- 每周知识摘要
- 知识回顾推荐
- 关联推荐
- 编译状态通知
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Optional

from openai import OpenAI
from config import get_llm_config, DATA_DIR, WIKI_DIR

logger = logging.getLogger(__name__)

# 调度器状态
_scheduler_task: Optional[asyncio.Task] = None
_push_subscribers: set = set()  # 企微 user_id 集合


def add_subscriber(user_id: str):
    """添加推送订阅"""
    _push_subscribers.add(user_id)


def remove_subscriber(user_id: str):
    """移除推送订阅"""
    _push_subscribers.discard(user_id)


# ===== 推送内容生成 =====

async def generate_weekly_digest() -> str:
    """生成每周知识摘要"""
    try:
        from storage.markdown_engine import MarkdownEngine
        engine = MarkdownEngine()
        items = engine.list_all()

        # 最近 7 天入库的文章
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        recent = []
        for item in items:
            created = item.get("created_at", "")
            if created:
                try:
                    dt = datetime.strptime(created.replace("T", " ").split(".")[0], "%Y-%m-%d %H:%M:%S")
                    if dt >= week_ago:
                        recent.append(item)
                except Exception:
                    pass

        if not recent:
            return ""

        # 统计
        wiki_count = sum(1 for _ in WIKI_DIR.glob("**/*.md") if not _.name.startswith("_"))
        total_count = len(items)
        tags_count = {}
        for item in recent:
            for tag in item.get("tags", []):
                tags_count[tag] = tags_count.get(tag, 0) + 1

        top_tags = sorted(tags_count.items(), key=lambda x: -x[1])[:5]
        tag_str = "、".join(f"{t}({c})" for t, c in top_tags)

        titles = "\n".join(f"  · {item.get('title', '未知')}" for item in recent[:10])

        digest = (
            f"📊 本周知识摘要 ({week_ago.strftime('%m/%d')}~{now.strftime('%m/%d')})\n\n"
            f"本周新增 {len(recent)} 篇知识\n"
            f"知识库总计: {total_count} 篇文章 / {wiki_count} 个 Wiki 页面\n\n"
            f"热门标签: {tag_str}\n\n"
            f"新增文章:\n{titles}\n\n"
            f"💡 访问 http://124.222.99.141:8900 查看详情"
        )
        return digest

    except Exception as e:
        logger.error(f"生成周报失败: {e}")
        return ""


async def generate_knowledge_review() -> str:
    """知识回顾 — 随机推荐一篇旧文章重温"""
    try:
        from storage.markdown_engine import MarkdownEngine
        engine = MarkdownEngine()
        items = engine.list_all()

        if len(items) < 3:
            return ""

        item = random.choice(items)
        title = item.get("title", "未知")
        summary = item.get("summary", "")[:150]
        tags = "、".join(item.get("tags", [])[:3])
        platform = item.get("source_platform", "")

        review = (
            f"📖 知识回顾\n\n"
            f"「{title}」\n\n"
            f"{summary}\n\n"
            f"标签: {tags}\n"
            f"来源: {platform}\n\n"
            f"💡 温故知新，访问知识库搜索相关内容"
        )
        return review

    except Exception as e:
        logger.error(f"生成回顾失败: {e}")
        return ""


async def generate_association_recommendation() -> str:
    """关联推荐 — 基于 Wiki 交叉引用发现关联"""
    try:
        from wiki.page_store import list_wiki_pages
        pages = list_wiki_pages()

        if len(pages) < 3:
            return ""

        # 随机选 2 个有共同来源的页面
        import re
        from wiki.page_store import read_page

        candidates = []
        for p in pages:
            data = read_page(p.get("path", ""))
            if data:
                refs = re.findall(r"\[\[(.+?)\]\]", data.get("full_content", ""))
                if refs:
                    candidates.append((p, refs))

        if not candidates:
            return ""

        page, refs = random.choice(candidates)
        ref_list = "、".join(refs[:3])

        rec = (
            f"🔗 关联推荐\n\n"
            f"Wiki 页面「{page.get('title', '')}」与以下概念有交叉引用:\n"
            f"  {ref_list}\n\n"
            f"💡 这些知识之间可能存在你未注意到的联系"
        )
        return rec

    except Exception as e:
        logger.error(f"生成推荐失败: {e}")
        return ""


async def generate_monthly_health() -> str:
    """月度健康报告"""
    try:
        from wiki.inspector import inspect
        report = inspect()
        summary = report.get("summary", "")
        orphans = report.get("orphan_pages", [])
        missing = report.get("missing_pages", [])

        health = (
            f"🏥 月度知识库健康报告\n\n"
            f"{summary}\n"
        )
        if orphans:
            health += f"\n⚠️ 孤立页面 ({len(orphans)}): {', '.join(orphans[:5])}"
        if missing:
            health += f"\n❌ 缺失引用 ({len(missing)}): {', '.join(missing[:5])}"
        if not orphans and not missing:
            health += "\n✅ 所有页面状态良好!"

        return health

    except Exception as e:
        logger.error(f"生成健康报告失败: {e}")
        return ""


# ===== 推送分发 =====

async def push_to_all(content: str, channel: str = "both"):
    """向所有订阅者推送

    channel: wecom | web | both
    """
    if not content:
        return

    # 企微推送
    if channel in ("wecom", "both") and _push_subscribers:
        try:
            from wecom.sender import send_text_msg
            for user_id in _push_subscribers:
                try:
                    send_text_msg(user_id, content)
                except Exception as e:
                    logger.warning(f"推送失败 {user_id}: {e}")
        except ImportError:
            pass

    # Web 推送通过 SSE 通知列表（存内存，前端轮询获取）
    if channel in ("web", "both"):
        _pending_notifications.append({
            "content": content,
            "time": datetime.now().isoformat(),
            "read": False,
        })
        # 保留最近 20 条
        while len(_pending_notifications) > 20:
            _pending_notifications.pop(0)


# 待推送通知队列（Web 端轮询）
_pending_notifications: list = []


def get_pending_notifications(mark_read: bool = True) -> list:
    """获取待推送通知"""
    result = [n for n in _pending_notifications if not n.get("read")]
    if mark_read:
        for n in result:
            n["read"] = True
    return result


# ===== 调度器 =====

async def scheduler_loop():
    """主调度循环 — 每小时检查一次是否有任务需要执行"""
    logger.info("推送调度器已启动")

    while True:
        try:
            now = datetime.now()
            weekday = now.weekday()  # 0=Mon
            hour = now.hour

            # 每周一 9:00 — 周报
            if weekday == 0 and hour == 9:
                digest = await generate_weekly_digest()
                if digest:
                    await push_to_all(digest)
                    logger.info("已推送每周摘要")

            # 每周三/五 12:00 — 知识回顾
            if weekday in (2, 4) and hour == 12:
                review = await generate_knowledge_review()
                if review:
                    await push_to_all(review)
                    logger.info("已推送知识回顾")

            # 每周六 10:00 — 关联推荐
            if weekday == 5 and hour == 10:
                rec = await generate_association_recommendation()
                if rec:
                    await push_to_all(rec)
                    logger.info("已推送关联推荐")

            # 每月 1 号 9:00 — 月度健康
            if now.day == 1 and hour == 9:
                health = await generate_monthly_health()
                if health:
                    await push_to_all(health)
                    logger.info("已推送月度健康报告")

        except Exception as e:
            logger.error(f"调度器异常: {e}")

        # 每小时检查一次
        await asyncio.sleep(3600)


async def start_scheduler():
    """启动调度器"""
    global _scheduler_task
    if _scheduler_task is None or _scheduler_task.done():
        _scheduler_task = asyncio.create_task(scheduler_loop())
        logger.info("推送调度器任务已创建")


async def trigger_push(push_type: str) -> str:
    """手动触发推送（供 API 和测试用）

    push_type: weekly | review | association | health
    """
    generators = {
        "weekly": generate_weekly_digest,
        "review": generate_knowledge_review,
        "association": generate_association_recommendation,
        "health": generate_monthly_health,
    }

    gen = generators.get(push_type)
    if not gen:
        return f"未知推送类型: {push_type}，可选: {', '.join(generators.keys())}"

    content = await gen()
    if content:
        await push_to_all(content)
        return content
    return "生成内容为空，可能知识库数据不足"
