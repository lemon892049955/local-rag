"""统一入库管线 (Ingest Pipeline)

封装完整的 fetch → OCR → clean → save → index → compile 流程。
所有入库入口（main.py / wecom/callback.py / assistant/router.py）统一调用此模块。
"""

import re
import logging
from pathlib import Path

from ingestion.router import FetcherRouter
from ingestion.base import RawContent
from transform.llm_cleaner import LLMCleaner, CleanedKnowledge
from storage.markdown_engine import MarkdownEngine
from retrieval.indexer import VectorIndexer
from utils.url_utils import normalize_url, check_duplicate
from config import DATA_DIR

logger = logging.getLogger(__name__)

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
                except Exception:
                    pass
            raw.content = re.sub(r'\[IMG_\d+\]', '', raw.content)
        else:
            # 无占位符：使用多图关联理解（≤3张）或逐张处理
            result = await ocr.ocr_multiple_images(images_to_process)
            if result:
                raw.content = raw.content + "\n\n---\n\n## 图片内容\n\n" + result
    except Exception as e:
        logger.warning(f"OCR 处理失败: {e}")


async def ingest_url(url: str, skip_duplicate: bool = True) -> dict:
    """完整入库流程：URL → 抓取 → OCR → 清洗 → 落库 → 索引 → Wiki 编译

    Returns:
        {"success": bool, "file_path": str, "title": str, "tags": list, "message": str}
    """
    if skip_duplicate:
        existing = check_duplicate(url, DATA_DIR)
        if existing:
            return {"success": True, "file_path": existing, "title": "(已存在)",
                    "tags": [], "message": f"该 URL 已入库: {existing}", "duplicate": True}

    try:
        raw = await _get_router().fetch(url)
    except Exception as e:
        return {"success": False, "error": f"抓取失败: {e}"}

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
            knowledge=knowledge, source_url=url,
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
