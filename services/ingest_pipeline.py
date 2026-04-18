"""统一入库管线 (Ingest Pipeline)

封装完整的 fetch → OCR → clean → save → index → compile 流程。
所有入库入口（main.py / wecom/callback.py / assistant/router.py）统一调用此模块。

v0.8: 增加 on_progress 回调，支持分阶段进度推送。
"""

import re
import logging
from pathlib import Path
from typing import Optional, Callable, Awaitable

from ingestion.router import FetcherRouter
from ingestion.base import RawContent
from transform.llm_cleaner import LLMCleaner, CleanedKnowledge
from storage.markdown_engine import MarkdownEngine
from retrieval.indexer import VectorIndexer
from utils.url_utils import normalize_url, check_duplicate
from config import DATA_DIR

logger = logging.getLogger(__name__)

# 进度回调类型: async def callback(stage, status, message, progress)
ProgressCallback = Optional[Callable[[str, str, str, int], Awaitable[None]]]

# 全局单例（避免每次请求重复实例化）
_router = None
_cleaner = None
_engine = None
_indexer = None


def _get_router():
    global _router
    if _router is None:
        _router = FetcherRouter()
    return _router


def _get_cleaner():
    global _cleaner
    if _cleaner is None:
        _cleaner = LLMCleaner()
    return _cleaner


def _get_engine():
    global _engine
    if _engine is None:
        _engine = MarkdownEngine()
    return _engine


def _get_indexer():
    global _indexer
    if _indexer is None:
        _indexer = VectorIndexer()
    return _indexer


async def ocr_images(raw: RawContent, max_images: int = 10):
    """对抓取结果中的图片执行智能识别，原地修改 raw.content

    v0.7: 支持图片分类→专用 prompt、多图关联理解
    """
    if not raw.images:
        return

    try:
        from ingestion.vision_ocr import VisionOCR
        ocr = VisionOCR()
        images_to_process = raw.images[:max_images]
        has_placeholders = "[IMG_" in raw.content

        if has_placeholders:
            # 有占位符：逐张处理，保持图文位置关系
            for i, img_url in enumerate(images_to_process):
                try:
                    text = await ocr.ocr_image_url(img_url)
                    if text:
                        # 用 Markdown 引用块格式，视觉上与正文区分
                        formatted = f"\n\n> **📷 图片 {i+1}**\n" + "\n".join(f"> {line}" for line in text.strip().split("\n")) + "\n"
                        raw.content = raw.content.replace(f"[IMG_{i+1}]", formatted)
                except Exception as e:
                    logger.debug(f"单张图片 OCR 失败: {img_url[:50]}... - {e}")
            raw.content = re.sub(r'\[IMG_\d+\]', '', raw.content)
        else:
            # 无占位符：使用多图关联理解（≤3张）或逐张处理
            result = await ocr.ocr_multiple_images(images_to_process)
            if result:
                raw.content = raw.content + "\n\n---\n\n## 图片内容\n\n" + result
    except Exception as e:
        logger.warning(f"OCR 处理失败: {e}")


async def ingest_url(url: str, skip_duplicate: bool = True, force: bool = False,
                     on_progress: ProgressCallback = None) -> dict:
    """完整入库流程：URL → 抓取 → OCR → 清洗 → 落库 → 索引 → Wiki 编译

    Args:
        url: 要入库的 URL
        skip_duplicate: 是否跳过已入库的 URL
        force: 强制重新入库（删除旧文件后重新走全流程）
        on_progress: 可选的异步进度回调 (stage, status, message, progress_percent)

    Returns:
        {"success": bool, "file_path": str, "title": str, "tags": list, "message": str}
    """
    async def _progress(stage: str, status: str, message: str, progress: int):
        if on_progress:
            try:
                await on_progress(stage, status, message, progress)
            except Exception as e:
                logger.debug(f"进度回调失败: {e}")

    # 1. 去重检查（force 模式下删除旧文件）
    await _progress("dedup", "running", "去重检查中...", 5)
    if force:
        existing = check_duplicate(url, DATA_DIR)
        if existing:
            from pathlib import Path
            old_path = Path(existing) if Path(existing).is_absolute() else DATA_DIR / existing
            if old_path.exists():
                old_path.unlink()
                logger.info(f"强制重入库: 已删除旧文件 {old_path.name}")
                await _progress("dedup", "ok", f"强制模式: 已删除旧文件 {old_path.name}", 10)
            else:
                await _progress("dedup", "ok", "强制模式: 无旧文件", 10)
        else:
            await _progress("dedup", "ok", "强制模式: 无旧文件", 10)
    elif skip_duplicate:
        existing = check_duplicate(url, DATA_DIR)
        if existing:
            await _progress("dedup", "duplicate", "该链接已入库", 100)
            return {"success": True, "file_path": existing, "title": "(已存在)",
                    "tags": [], "message": f"该 URL 已入库: {existing}", "duplicate": True}
    await _progress("dedup", "ok", "去重检查通过", 10)

    # 2. 抓取
    await _progress("fetch", "running", "正在抓取网页内容...", 15)
    try:
        raw = await _get_router().fetch(url)
        content_len = len(raw.content) if raw.content else 0
        img_count = len(raw.images) if raw.images else 0
        await _progress("fetch", "ok", f"抓取完成，{content_len}字" + (f"，{img_count}张图片" if img_count else ""), 30)
    except Exception as e:
        await _progress("fetch", "error", f"抓取失败: {e}", 30)
        return {"success": False, "error": f"抓取失败: {e}"}

    # 3. 图片 OCR
    if raw.images:
        await _progress("ocr", "running", f"OCR 识别中（{len(raw.images)}张图片）...", 35)
        await ocr_images(raw)
        await _progress("ocr", "ok", "OCR 识别完成", 45)
    else:
        await _progress("ocr", "skip", "无图片，跳过 OCR", 45)

    # 4. LLM 清洗
    await _progress("clean", "running", "LLM 提取标题/摘要/标签...", 50)
    try:
        knowledge = await _get_cleaner().clean(
            title=raw.title, content=raw.content,
            source=raw.source_platform, author=raw.author,
            original_tags=raw.original_tags,
        )
        tags_str = "、".join(knowledge.tags[:3]) if knowledge.tags else ""
        await _progress("clean", "ok", f"清洗完成: 《{knowledge.title[:20]}》 标签: {tags_str}", 65)
    except Exception as e:
        await _progress("clean", "error", f"LLM 清洗失败: {e}", 65)
        return {"success": False, "error": f"LLM 清洗失败: {e}"}

    # 5. 落库
    await _progress("save", "running", "保存到知识库...", 70)
    try:
        filepath = _get_engine().save(
            knowledge=knowledge, source_url=url,
            source_platform=raw.source_platform, author=raw.author,
        )
        await _progress("save", "ok", f"已保存: {filepath.name}", 75)
    except Exception as e:
        await _progress("save", "error", f"保存失败: {e}", 75)
        return {"success": False, "error": f"文件保存失败: {e}"}

    # 6. 索引
    await _progress("index", "running", "构建向量索引...", 80)
    chunk_count = 0
    try:
        chunk_count = _get_indexer().index_file(filepath)
        await _progress("index", "ok", f"生成 {chunk_count} 个索引切片", 90)
    except Exception as e:
        logger.warning(f"索引构建失败: {e}")
        await _progress("index", "error", f"索引构建失败: {e}", 90)

    # 7. Wiki 编译入队
    await _progress("compile", "running", "Wiki 编译入队...", 93)
    try:
        # force 模式：清除 Wiki 页面中该文件的 sources 引用，确保重编译不被跳过
        if force:
            try:
                from wiki.page_store import list_wiki_pages, read_page
                from config import WIKI_DIR
                fname = filepath.name
                for p in list_wiki_pages():
                    if fname in p.get("sources", []):
                        pp = WIKI_DIR / p["path"]
                        if pp.exists():
                            content = pp.read_text(encoding="utf-8")
                            content = content.replace(f"  - {fname}\n", "").replace(f"  - {fname}", "")
                            pp.write_text(content, encoding="utf-8")
                logger.info(f"强制重入库: 已清除 Wiki sources 中的 {fname}")
            except Exception as e:
                logger.warning(f"清除 Wiki sources 失败: {e}")
        from wiki.compile_queue import enqueue_compile, get_queue
        await enqueue_compile(filepath)
        queue_size = get_queue().qsize()
        await _progress("compile", "queued", f"Wiki 编译已排队（队列第{queue_size}位）", 95)
    except Exception:
        await _progress("compile", "skip", "Wiki 编译入队失败", 95)

    await _progress("done", "ok", "入库完成！", 100)

    return {
        "success": True,
        "file_path": str(filepath),
        "title": knowledge.title,
        "tags": knowledge.tags,
        "message": f"入库成功，生成 {chunk_count} 个索引切片，Wiki 编译已排队",
    }


async def ingest_raw(raw: RawContent, source_url: str = "") -> dict:
    """对已解析的 RawContent 执行后续管线"""
    await ocr_images(raw)

    try:
        knowledge = await _get_cleaner().clean(
            title=raw.title, content=raw.content,
            source=raw.source_platform, author=raw.author,
            original_tags=raw.original_tags,
        )
    except Exception as e:
        return {"success": False, "error": f"LLM 清洗失败: {e}"}

    try:
        filepath = _get_engine().save(
            knowledge=knowledge, source_url=source_url,
            source_platform=raw.source_platform, author=raw.author,
        )
    except Exception as e:
        return {"success": False, "error": f"文件保存失败: {e}"}

    chunk_count = 0
    try:
        chunk_count = _get_indexer().index_file(filepath)
    except Exception as e:
        logger.warning(f"索引构建失败: {e}")

    try:
        from wiki.compile_queue import enqueue_compile
        await enqueue_compile(filepath)
    except Exception:
        pass

    return {
        "success": True,
        "file_path": str(filepath),
        "title": knowledge.title,
        "tags": knowledge.tags,
        "message": f"入库成功，{chunk_count} 个切片",
    }
