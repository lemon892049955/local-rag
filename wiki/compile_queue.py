"""Wiki 编译任务队列

入库 (Ingest) 可高并发，但 Wiki 编译 (Compile) 必须串行执行。
使用 asyncio.Queue + 单消费者协程，保证同一时间只有一个 LLM 在修改 Wiki。
"""

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_compile_queue: asyncio.Queue = None
_worker_task: asyncio.Task = None


def get_queue() -> asyncio.Queue:
    """获取全局编译队列（懒初始化）"""
    global _compile_queue
    if _compile_queue is None:
        _compile_queue = asyncio.Queue()
    return _compile_queue


async def enqueue_compile(article_path: Path, user_id: str = ""):
    """入库完成后调用：将文章加入编译队列

    Args:
        article_path: data/ 目录下新入库文章的路径
        user_id: 企微用户 ID（可选，用于编译完成通知）
    """
    queue = get_queue()
    await queue.put({"article_path": article_path, "user_id": user_id})
    logger.info(f"编译任务入队: {article_path.name} (队列长度: {queue.qsize()})")


async def compile_worker():
    """单线程编译消费者 — 严格串行执行

    从队列中逐个取出任务，调用 WikiCompiler 编译。
    保证同一时间只有一个编译在运行，杜绝并发冲突。
    """
    from wiki.compiler import WikiCompiler
    compiler = WikiCompiler()
    queue = get_queue()

    logger.info("Wiki 编译 Worker 已启动，等待任务...")

    while True:
        task = await queue.get()
        article_path = task["article_path"]
        user_id = task.get("user_id", "")

        try:
            logger.info(f"开始编译: {article_path.name}")
            result = await compiler.compile(article_path)
            logger.info(
                f"编译完成: {article_path.name}, "
                f"新建 {len(result.get('new_pages', []))} 个页面, "
                f"更新 {len(result.get('updated_pages', []))} 个页面"
            )

            # 如果有企微用户 ID，推送编译完成通知
            if user_id and (result.get("new_pages") or result.get("updated_pages")):
                try:
                    from wecom.sender import send_text_msg
                    pages_info = []
                    for p in result.get("new_pages", []):
                        pages_info.append(f"  新建: {p}")
                    for p in result.get("updated_pages", []):
                        pages_info.append(f"  更新: {p}")
                    msg = f"📝 Wiki 编译完成\n" + "\n".join(pages_info[:5])
                    send_text_msg(user_id, msg)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"编译失败: {article_path.name} - {e}", exc_info=True)
        finally:
            queue.task_done()


async def start_compile_worker():
    """在应用启动时调用，启动编译 Worker"""
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(compile_worker())
        logger.info("Wiki 编译 Worker 任务已创建")
