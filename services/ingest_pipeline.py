"""统一入库管线 (Ingest Pipeline)

封装完整的 fetch → OCR → clean → save → index → compile 流程。
所有入库入口（main.py / wecom/callback.py / assistant/router.py）统一调用此模块。

v0.9: 合并 ingest_url/ingest_raw 重复逻辑，OCR 并行处理，VisionOCR 单例复用。
v1.0: 精确错误处理，区分网络错误/API限流/认证失败等。
"""

import re
import logging
from pathlib import Path
from typing import Optional, Callable, Awaitable

from ingestion.base import RawContent, FetchError, ErrorType
from utils.url_utils import normalize_url, check_duplicate
from config import DATA_DIR

logger = logging.getLogger(__name__)

# 进度回调类型: async def callback(stage, status, message, progress)
ProgressCallback = Optional[Callable[[str, str, str, int], Awaitable[None]]]


def _get_router():
    from main import get_router
    return get_router()


def _get_cleaner():
    from main import get_cleaner
    return get_cleaner()


def _get_engine():
    from main import get_engine
    return get_engine()


def _get_indexer():
    from main import get_indexer
    return get_indexer()


def _format_error(e: Exception, context: str = "") -> str:
    """格式化错误信息，提供用户友好的提示"""
    if isinstance(e, FetchError):
        type_msgs = {
            ErrorType.NETWORK: "网络连接失败，请检查网络后重试",
            ErrorType.RATE_LIMIT: "API 调用频率超限，请稍后重试",
            ErrorType.AUTH: "API 认证失败，请检查 API Key 配置",
            ErrorType.NOT_FOUND: "资源不存在或已被删除",
            ErrorType.BLOCKED: "被目标网站反爬拦截，建议稍后重试",
            ErrorType.PARSE: "内容解析失败",
            ErrorType.VALIDATION: "内容校验失败",
            ErrorType.API: "API 服务异常",
            ErrorType.UNKNOWN: "未知错误",
        }
        base_msg = type_msgs.get(e.error_type, e.reason)
        return f"{base_msg}" + (f" ({e.url[:50]})" if e.url else "")

    error_msg = str(e)
    # 识别常见错误模式
    if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
        return f"请求超时，请检查网络后重试"
    if "connection" in error_msg.lower():
        return f"网络连接失败"
    if "401" in error_msg or "unauthorized" in error_msg.lower():
        return f"API 认证失败，请检查配置"
    if "429" in error_msg or "rate limit" in error_msg.lower():
        return f"API 调用频率超限，请稍后重试"
    if "502" in error_msg or "503" in error_msg:
        return f"服务暂时不可用，请稍后重试"

    return error_msg[:100] if len(error_msg) > 100 else error_msg


async def ocr_images(raw: RawContent, max_images: int = 10):
    """对抓取结果中的图片执行智能识别，原地修改 raw.content

    v0.8: 并行处理图片，大幅提升速度
    """
    if not raw.images:
        return

    try:
        from ingestion.vision_ocr import VisionOCR
        import asyncio

        ocr = VisionOCR()  # 单例复用
        images_to_process = raw.images[:max_images]
        has_placeholders = "[IMG_" in raw.content

        if has_placeholders:
            # 有占位符：并行处理所有图片
            async def process_one(idx: int, img_url: str) -> tuple[int, str]:
                try:
                    text = await ocr.ocr_image_url(img_url)
                    if text:
                        formatted = f"\n\n> **📷 图片 {idx+1}**\n" + "\n".join(f"> {line}" for line in text.strip().split("\n")) + "\n"
                        return (idx, formatted)
                except Exception as e:
                    logger.debug(f"图片 {idx+1} OCR 失败: {e}")
                return (idx, "")

            # 并行处理
            tasks = [process_one(i, url) for i, url in enumerate(images_to_process)]
            results = await asyncio.gather(*tasks)

            # 按顺序替换占位符
            for idx, formatted in sorted(results):
                if formatted:
                    raw.content = raw.content.replace(f"[IMG_{idx+1}]", formatted)

            # 清理未替换的占位符
            raw.content = re.sub(r'\[IMG_\d+\]', '', raw.content)
        else:
            # 无占位符：使用多图关联理解（≤3张）或逐张处理
            result = await ocr.ocr_multiple_images(images_to_process)
            if result:
                raw.content = raw.content + "\n\n---\n\n## 图片内容\n\n" + result
    except Exception as e:
        logger.warning(f"OCR 处理失败: {e}")


async def _run_pipeline(raw: RawContent, source_url: str, force: bool,
                        on_progress: ProgressCallback, start_progress: int = 0) -> dict:
    """核心入库管线（被 ingest_url 和 ingest_raw 共用）"""
    async def _progress(stage: str, status: str, message: str, progress: int):
        if on_progress:
            try:
                await on_progress(stage, status, message, progress)
            except Exception as e:
                logger.debug(f"进度回调失败: {e}")

    progress_offset = start_progress
    progress_scale = 100 - start_progress

    # 1. 图片 OCR
    if raw.images:
        await _progress("ocr", "running", f"OCR 识别中（{len(raw.images)}张图片）...", progress_offset + int(progress_scale * 0.15))
        await ocr_images(raw)
        await _progress("ocr", "ok", "OCR 识别完成", progress_offset + int(progress_scale * 0.30))
    else:
        await _progress("ocr", "skip", "无图片，跳过 OCR", progress_offset + int(progress_scale * 0.30))

    # 2. LLM 清洗
    await _progress("clean", "running", "LLM 提取元数据...", progress_offset + int(progress_scale * 0.35))
    try:
        knowledge = await _get_cleaner().clean(
            title=raw.title, content=raw.content,
            source=raw.source_platform, author=raw.author,
            original_tags=raw.original_tags,
        )
        tags_str = "、".join(knowledge.tags[:3]) if knowledge.tags else ""
        await _progress("clean", "ok", f"清洗完成: 《{knowledge.title[:20]}》 {tags_str}", progress_offset + int(progress_scale * 0.50))
    except Exception as e:
        error_msg = _format_error(e, "LLM")
        await _progress("clean", "error", error_msg, progress_offset + int(progress_scale * 0.50))
        return {"success": False, "error": error_msg}

    # 3. 落库
    await _progress("save", "running", "保存到知识库...", progress_offset + int(progress_scale * 0.55))
    try:
        filepath = _get_engine().save(
            knowledge=knowledge, source_url=source_url,
            source_platform=raw.source_platform, author=raw.author,
        )
        await _progress("save", "ok", f"已保存: {filepath.name}", progress_offset + int(progress_scale * 0.65))
    except Exception as e:
        error_msg = _format_error(e, "保存")
        await _progress("save", "error", error_msg, progress_offset + int(progress_scale * 0.65))
        return {"success": False, "error": error_msg}

    # 4. 索引
    await _progress("index", "running", "构建向量索引...", progress_offset + int(progress_scale * 0.70))
    chunk_count = 0
    try:
        chunk_count = _get_indexer().index_file(filepath)
        await _progress("index", "ok", f"生成 {chunk_count} 个索引切片", progress_offset + int(progress_scale * 0.85))
        # 重建 BM25 索引
        try:
            from main import get_searcher
            get_searcher().rebuild_bm25()
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"索引构建失败: {e}")
        await _progress("index", "error", f"索引构建失败: {e}", progress_offset + int(progress_scale * 0.85))

    # 5. Wiki 编译入队
    await _progress("compile", "running", "Wiki 编译入队...", progress_offset + int(progress_scale * 0.90))
    try:
        # force 模式：清除 Wiki 页面中该文件的 sources 引用
        if force:
            try:
                from wiki.page_store import list_wiki_pages
                from config import WIKI_DIR
                fname = filepath.name
                for p in list_wiki_pages():
                    if fname in p.get("sources", []):
                        pp = WIKI_DIR / p["path"]
                        if pp.exists():
                            content = pp.read_text(encoding="utf-8")
                            content = content.replace(f"  - {fname}\n", "").replace(f"  - {fname}", "")
                            pp.write_text(content, encoding="utf-8")
                logger.debug(f"已清除 Wiki sources 中的 {fname}")
            except Exception as e:
                logger.debug(f"清除 Wiki sources 失败: {e}")

        from wiki.compile_queue import enqueue_compile, get_queue
        await enqueue_compile(filepath)
        queue_size = get_queue().qsize()
        await _progress("compile", "queued", f"Wiki 编译已排队（队列第{queue_size}位）", progress_offset + int(progress_scale * 0.95))
    except Exception:
        await _progress("compile", "skip", "Wiki 编译入队失败", progress_offset + int(progress_scale * 0.95))

    await _progress("done", "ok", "入库完成！", 100)

    return {
        "success": True,
        "file_path": str(filepath),
        "title": knowledge.title,
        "tags": knowledge.tags,
        "message": f"入库成功，生成 {chunk_count} 个索引切片，Wiki 编译已排队",
    }


async def ingest_url(url: str, skip_duplicate: bool = True, force: bool = False,
                     on_progress: ProgressCallback = None) -> dict:
    """完整入库流程：URL → 抓取 → OCR → 清洗 → 落库 → 索引 → Wiki 编译"""
    async def _progress(stage: str, status: str, message: str, progress: int):
        if on_progress:
            try:
                await on_progress(stage, status, message, progress)
            except Exception as e:
                logger.debug(f"进度回调失败: {e}")

    # 1. 去重检查
    await _progress("dedup", "running", "去重检查中...", 5)
    if force:
        existing = check_duplicate(url, DATA_DIR)
        if existing:
            old_path = Path(existing) if Path(existing).is_absolute() else DATA_DIR / existing
            if old_path.exists():
                old_path.unlink()
                logger.info(f"强制重入库: 已删除旧文件 {old_path.name}")
        await _progress("dedup", "ok", "强制模式: 已处理", 10)
    elif skip_duplicate:
        existing = check_duplicate(url, DATA_DIR)
        if existing:
            await _progress("dedup", "duplicate", "该链接已入库", 100)
            return {"success": True, "file_path": existing, "title": "(已存在)",
                    "tags": [], "message": f"该 URL 已入库: {existing}", "duplicate": True}
    await _progress("dedup", "ok", "去重检查通过", 10)

    # 2. 抓取
    await _progress("fetch", "running", "正在抓取...", 15)
    try:
        raw = await _get_router().fetch(url)
        img_count = len(raw.images) if raw.images else 0
        await _progress("fetch", "ok", f"抓取完成，{len(raw.content)}字" + (f"，{img_count}张图" if img_count else ""), 30)
    except Exception as e:
        error_msg = _format_error(e, "抓取")
        await _progress("fetch", "error", error_msg, 30)
        return {"success": False, "error": error_msg, "error_type": getattr(e, 'error_type', ErrorType.UNKNOWN).value if isinstance(e, FetchError) else "unknown"}

    # 3. 执行核心管线
    return await _run_pipeline(raw, url, force, on_progress, start_progress=30)


async def ingest_raw(raw: RawContent, source_url: str = "", force: bool = False,
                     on_progress: ProgressCallback = None) -> dict:
    """对已解析的 RawContent 执行后续管线"""
    return await _run_pipeline(raw, source_url, force, on_progress, start_progress=0)
