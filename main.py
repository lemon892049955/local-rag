"""FastAPI 主入口

暴露核心端点:
- POST /ingest       - 入库 (URL -> 抓取 -> 清洗 -> 落库 -> 索引)
- POST /ingest/text  - 手动入库
- POST /search       - 检索 (查询 -> 向量召回 -> LLM 答案生成)
- GET  /api/knowledge      - 知识列表
- GET  /api/knowledge/{id} - 知识详情
"""

from pathlib import Path
import json
import logging
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional

from config import DATA_DIR, BASE_DIR, API_TOKEN
from ingestion.router import FetcherRouter
from transform.llm_cleaner import LLMCleaner
from storage.markdown_engine import MarkdownEngine
from retrieval.indexer import VectorIndexer
from retrieval.hybrid_searcher import HybridSearcher
from utils.url_utils import normalize_url, check_duplicate


def _safe_filename(filename: str) -> str:
    """校验文件名安全性，防止路径穿越"""
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    if not filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="仅支持 .md 文件")
    return filename


def _safe_wiki_path(subdir: str, filename: str) -> str:
    """校验 Wiki 路径安全性"""
    if subdir not in ("topics", "entities", "concepts", "moc"):
        raise HTTPException(status_code=400, detail="非法目录")
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    return f"{subdir}/{filename}"


# ===== API 认证中间件 =====

async def verify_api_token(authorization: str = Header(None)):
    """验证 API Token（如果配置了的话）"""
    if not API_TOKEN:
        return True  # 未配置 token，跳过认证

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="缺少认证信息。请在请求头添加 Authorization: Bearer <your-token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    if token != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="认证失败，Token 无效",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ===== Startup =====
    from wiki.compile_queue import start_compile_worker
    await start_compile_worker()
    from assistant.scheduler import start_scheduler
    await start_scheduler()
    # 预热 Reranker + Embedding
    try:
        searcher = get_searcher()
        _ = searcher.reranker.model
        logger.info("Reranker 模型预热完成")
    except Exception as e:
        logger.warning(f"Reranker 预热失败（首次搜索时加载）: {e}")
    # 启动 iLink Bot
    try:
        from ilink.bot import start_bot
        await start_bot()
    except Exception as e:
        logger.warning(f"iLink Bot 启动跳过: {e}")
    yield
    # ===== Shutdown =====
    try:
        from ilink.bot import _bot, _bot_running
        if _bot:
            await _bot.stop()
    except Exception as e:
        logger.debug(f"iLink Bot 关闭: {e}")


app = FastAPI(
    title="BuddyKnow - 个人碎片知识落库系统",
    description="本地化优先的个人知识资产管理与 RAG 问答系统（v2.2 最优组合版）",
    version="2.2.0",
    lifespan=lifespan,
)

# CORS 配置（允许跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册企业微信回调路由（保留兼容）
from wecom.callback import router as wecom_router
app.include_router(wecom_router)

# 注册 iLink Bot 管理路由
from ilink.api import router as ilink_router
app.include_router(ilink_router)

# 注册 AI 助手路由
from assistant.router import router as assistant_router
app.include_router(assistant_router)


# 全局组件（延迟初始化以加速启动）
_router = None
_cleaner = None
_engine = None
_indexer = None
_searcher = None


def get_router():
    global _router
    if _router is None:
        _router = FetcherRouter()
    return _router


def get_cleaner():
    global _cleaner
    if _cleaner is None:
        _cleaner = LLMCleaner()
    return _cleaner


def get_engine():
    global _engine
    if _engine is None:
        _engine = MarkdownEngine()
    return _engine


def get_indexer():
    global _indexer
    if _indexer is None:
        _indexer = VectorIndexer()
    return _indexer


def get_searcher():
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher(indexer=get_indexer())
    return _searcher


# ===== 请求/响应模型 =====

class IngestRequest(BaseModel):
    url: str
    force: bool = False  # 强制重新入库（删除旧文件后重新走全流程）

class BatchIngestRequest(BaseModel):
    urls: list[str]

class IngestResponse(BaseModel):
    success: bool
    file_path: str
    title: str
    tags: list[str]
    message: str = ""

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchResponse(BaseModel):
    answer: str
    sources: list
    debug: dict = {}
    highlight_keywords: list[str] = []


# ===== 端点 =====

@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_token)])
async def ingest(req: IngestRequest):
    """入库接口 - URL -> 抓取 -> 清洗 -> 落库 -> 索引"""
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL 不能为空")

    import time
    start_time = time.time()

    from services.ingest_pipeline import ingest_url
    result = await ingest_url(url, force=req.force)

    duration_ms = int((time.time() - start_time) * 1000)

    # 埋点：记录入库
    from utils.analytics import track_ingest
    track_ingest(
        url=url,
        title=result.get("title"),
        success=result.get("success", False) or result.get("duplicate", False),
        duration_ms=duration_ms,
        source="web",
    )

    if not result.get("success") and not result.get("duplicate"):
        raise HTTPException(status_code=422, detail=result.get("error", "入库失败"))

    return IngestResponse(
        success=True,
        file_path=result.get("file_path", ""),
        title=result.get("title", ""),
        tags=result.get("tags", []),
        message=result.get("message", ""),
    )


@app.post("/ingest/stream", dependencies=[Depends(verify_api_token)])
async def ingest_stream(req: IngestRequest):
    """入库接口（SSE 流式） — 分阶段推送进度

    返回 text/event-stream，每个阶段推送一条 JSON 事件：
    {"stage": "fetch", "status": "running", "message": "...", "progress": 20}
    """
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL 不能为空")

    import asyncio

    # 用 asyncio.Queue 做 SSE 桥接
    progress_queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(stage: str, status: str, message: str, progress: int):
        await progress_queue.put({
            "stage": stage, "status": status,
            "message": message, "progress": progress,
        })

    async def event_generator():
        # 启动入库任务
        from services.ingest_pipeline import ingest_url
        task = asyncio.create_task(ingest_url(url, force=req.force, on_progress=on_progress))

        # 持续读取进度事件
        while True:
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=120)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("stage") == "done" or event.get("status") == "error":
                    break
                if event.get("stage") == "dedup" and event.get("status") == "duplicate":
                    break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'stage': 'timeout', 'status': 'error', 'message': '入库超时', 'progress': 0})}\n\n"
                break

        # 等待任务完成，推送最终结果
        try:
            result = await task
            yield f"data: {json.dumps({'stage': 'result', 'status': 'ok', 'result': result, 'progress': 100}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'stage': 'result', 'status': 'error', 'message': str(e), 'progress': 0}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class IngestUploadRequest(BaseModel):
    force: bool = False


@app.post("/ingest/upload", response_model=IngestResponse, dependencies=[Depends(verify_api_token)])
async def ingest_upload(file: UploadFile = File(...), force: bool = Form(False)):
    """文件上传入库 — 支持 PDF/图片/音频

    Args:
        force: 强制重新入库（删除旧文件后重新处理）
    """
    import tempfile

    if file is None:
        raise HTTPException(status_code=400, detail="请上传文件")

    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        from ingestion.dispatcher import Dispatcher
        dispatcher = Dispatcher()
        file_type = dispatcher.detect_type(str(tmp_path))

        if file_type == "unknown":
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix}")

        raw = await dispatcher.dispatch(str(tmp_path))

        from services.ingest_pipeline import ingest_raw
        result = await ingest_raw(raw, source_url=f"file://{file.filename}", force=force)

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "入库失败"))

        return IngestResponse(
            success=True,
            file_path=result.get("file_path", ""),
            title=result.get("title", ""),
            tags=result.get("tags", []),
            message=f"{file_type.upper()} {result.get('message', '')}",
        )
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


@app.post("/ingest/upload/stream", dependencies=[Depends(verify_api_token)])
async def ingest_upload_stream(file: UploadFile = File(...), force: bool = Form(False)):
    """文件上传入库（SSE 流式）— 分阶段推送进度

    返回 text/event-stream，每个阶段推送一条 JSON 事件
    """
    import tempfile
    import asyncio

    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    progress_queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(stage: str, status: str, message: str, progress: int):
        await progress_queue.put({
            "stage": stage, "status": status,
            "message": message, "progress": progress,
        })

    async def event_generator():
        try:
            # 推送解析阶段
            await progress_queue.put({"stage": "parse", "status": "running", "message": f"正在解析 {file.filename}...", "progress": 10})

            from ingestion.dispatcher import Dispatcher
            dispatcher = Dispatcher()
            file_type = dispatcher.detect_type(str(tmp_path))

            if file_type == "unknown":
                await progress_queue.put({"stage": "parse", "status": "error", "message": f"不支持的文件类型: {suffix}", "progress": 0})
                return

            await progress_queue.put({"stage": "parse", "status": "ok", "message": f"文件类型: {file_type.upper()}", "progress": 20})

            raw = await dispatcher.dispatch(str(tmp_path))

            from services.ingest_pipeline import ingest_raw
            result = await ingest_raw(raw, source_url=f"file://{file.filename}", force=force, on_progress=on_progress)

            await progress_queue.put({"stage": "done", "status": "ok", "progress": 100, "result": result})
        except Exception as e:
            await progress_queue.put({"stage": "error", "status": "error", "message": str(e), "progress": 0})
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass

    async def stream_generator():
        task = asyncio.create_task(event_generator())

        while True:
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=120)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("stage") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'stage': 'timeout', 'status': 'error', 'message': '处理超时', 'progress': 0})}\n\n"
                break

        try:
            await task
        except Exception:
            pass

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/ingest/batch", dependencies=[Depends(verify_api_token)])
async def ingest_batch(req: BatchIngestRequest):
    """批量入库 — 多个 URL 逐条入库"""
    if not req.urls:
        raise HTTPException(status_code=400, detail="URL 列表不能为空")

    import asyncio
    from services.ingest_pipeline import ingest_url

    results = []
    for url in req.urls:
        url = url.strip()
        if not url:
            continue
        try:
            result = await ingest_url(url)
            results.append({"url": url, **result})
        except Exception as e:
            results.append({"url": url, "success": False, "error": str(e)})

    success_count = sum(1 for r in results if r.get("success"))
    return {
        "total": len(results),
        "success": success_count,
        "failed": len(results) - success_count,
        "results": results,
    }


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """检索接口 - 语义搜索 + RAG 答案生成"""

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    import time
    start_time = time.time()

    result = await get_searcher().search(
        query=req.query,
        top_k=req.top_k,
    )

    duration_ms = int((time.time() - start_time) * 1000)

    # 埋点：记录搜索
    from utils.analytics import track_search
    track_search(
        query=req.query,
        sources_count=len(result.get("sources", [])),
        duration_ms=duration_ms,
    )

    return SearchResponse(
        answer=result["answer"],
        sources=result["sources"],
        debug=result.get("debug", {}),
        highlight_keywords=result.get("highlight_keywords", []),
    )


@app.post("/search/stream")
async def search_stream(req: SearchRequest):
    """SSE 流式搜索 — 分阶段推送 sources → answer tokens"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    async def event_generator():
        async for chunk in get_searcher().search_stream(
            query=req.query,
            top_k=req.top_k,
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/reindex", dependencies=[Depends(verify_api_token)])
async def reindex():
    """重建索引 - 从 Markdown 文件恢复 ChromaDB"""
    total = get_indexer().reindex_all()
    return {"success": True, "total_chunks": total}


@app.get("/health")
async def health():
    """健康检查 — 快速探活端点"""
    checks = {"api": "ok"}
    # ChromaDB
    try:
        count = get_indexer().get_stats().get("total_chunks", 0)
        checks["vectordb"] = "ok" if count >= 0 else "error"
        checks["total_chunks"] = count
    except Exception as e:
        checks["vectordb"] = f"error: {e}"
    # LLM
    try:
        from config import get_llm_config
        config = get_llm_config()
        checks["llm"] = "ok" if config.get("api_key") else "no_key"
    except Exception as e:
        checks["llm"] = f"error: {e}"
    # Data dir
    checks["data_dir"] = "ok" if DATA_DIR.exists() else "missing"
    from config import WIKI_DIR
    checks["wiki_dir"] = "ok" if WIKI_DIR.exists() else "missing"
    checks["knowledge_files"] = len(list(DATA_DIR.glob("*.md")))

    ok_values = {"ok", True}
    overall = "ok" if all(
        (v in ok_values) or isinstance(v, int) for v in checks.values()
    ) else "degraded"
    return {"status": overall, **checks}


@app.get("/stats")
async def stats():
    """系统状态"""
    indexer_stats = get_indexer().get_stats()
    file_count = len(list(DATA_DIR.glob("*.md")))
    from config import WIKI_DIR
    wiki_count = sum(1 for _ in WIKI_DIR.glob("**/*.md") if not _.name.startswith("_"))
    # 编译队列状态
    try:
        from wiki.compile_queue import get_queue
        queue_size = get_queue().qsize()
    except Exception:
        queue_size = 0
    return {
        "knowledge_files": file_count,
        "wiki_pages": wiki_count,
        "compile_queue": queue_size,
        **indexer_stats,
    }


@app.get("/")
async def root():
    """首页 - 返回 Web UI"""
    index_path = BASE_DIR / "web" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"name": "BuddyKnow", "version": "2.2.0", "hint": "Web UI not found, visit /docs"}


@app.get("/discover")
async def discover_page():
    """发现页面 - AI 热榜监控"""
    discover_path = BASE_DIR / "web" / "discover.html"
    if discover_path.exists():
        return FileResponse(discover_path)
    return {"error": "Discover page not found"}


# ===== Wiki API =====

@app.get("/api/wiki/pages")
async def list_wiki_pages():
    """Wiki 页面列表"""
    from wiki.page_store import list_wiki_pages as _list
    pages = _list()
    return {"pages": pages, "total": len(pages)}


@app.get("/api/wiki/graph")
async def get_wiki_graph():
    """知识图谱 — 标签聚合视角

    节点 = 标签（知识主题）
    边 = 两个标签共现于同一篇文章（共现越多线越粗）
    点击标签 → 展示该标签下的文章列表
    """
    files = get_engine().list_all()

    # 1. 建立标签→文章映射
    tag_articles = {}  # tag -> [{title, file_path, summary, platform}]
    for item in files:
        for tag in (item.get("tags") or []):
            tag_articles.setdefault(tag, []).append({
                "title": item.get("title", ""),
                "file_path": item.get("file_path", ""),
                "summary": item.get("summary", ""),
                "platform": item.get("source_platform", ""),
                "author": item.get("author", ""),
            })

    # 2. 构建节点（只保留有文章的标签）
    nodes = []
    for tag, articles in tag_articles.items():
        nodes.append({
            "id": tag,
            "label": tag,
            "count": len(articles),
            "articles": articles,
        })

    # 3. 构建边（两个标签共现于同一篇文章）
    edges = []
    tags = list(tag_articles.keys())
    # 建立文章→标签集合的反向映射
    article_tags = {}
    for item in files:
        fp = item.get("file_path", "")
        article_tags[fp] = set(item.get("tags") or [])

    # 计算标签对的共现次数
    from collections import Counter
    pair_count = Counter()
    for fp, tag_set in article_tags.items():
        tag_list = sorted(tag_set)
        for i in range(len(tag_list)):
            for j in range(i + 1, len(tag_list)):
                pair_count[(tag_list[i], tag_list[j])] += 1

    for (t1, t2), weight in pair_count.items():
        edges.append({"source": t1, "target": t2, "weight": weight})

    return {"nodes": nodes, "edges": edges}


@app.get("/api/wiki/tree")
async def get_wiki_tree():
    """Wiki 目录树 — 优先读 Taxonomy 语义分类，fallback 到标签聚类

    v0.8: 改为读 _taxonomy.yaml 语义分类结果，确保前端分类与编译时一致。
    """
    from wiki.page_store import list_wiki_pages as _list
    from config import WIKI_DIR
    import yaml

    pages = _list()
    if not pages:
        return {"folders": [], "total_pages": 0}

    # 建立 path → page 的映射
    path_to_page = {p.get("path", ""): p for p in pages}
    classified_paths = set()

    # 优先读 taxonomy
    taxonomy_path = WIKI_DIR / "_taxonomy.yaml"
    result = []

    if taxonomy_path.exists():
        try:
            taxonomy = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
            categories = taxonomy.get("categories", {}) if taxonomy else {}

            for cat_name, cat_data in categories.items():
                if not isinstance(cat_data, dict):
                    continue
                cat_pages = []
                for p in cat_data.get("pages", []):
                    if p in path_to_page:
                        cat_pages.append(path_to_page[p])
                        classified_paths.add(p)
                # 子分类的页面也归入父分类展示
                for child_name, child_data in (cat_data.get("children") or {}).items():
                    if isinstance(child_data, dict):
                        for p in child_data.get("pages", []):
                            if p in path_to_page:
                                cat_pages.append(path_to_page[p])
                                classified_paths.add(p)
                if cat_pages:
                    result.append({
                        "name": cat_name,
                        "count": len(cat_pages),
                        "pages": cat_pages,
                    })
        except Exception:
            pass

    # 未被 taxonomy 覆盖的页面归入"待分类"（MOC 导航页除外，它们本身就是分类索引）
    uncategorized = [p for p in pages
                     if p.get("path", "") not in classified_paths
                     and not p.get("path", "").startswith("moc/")]
    if uncategorized:
        result.append({
            "name": "待分类",
            "count": len(uncategorized),
            "pages": uncategorized,
        })

    # 按页面数降序
    result.sort(key=lambda x: -x["count"])

    return {"folders": result, "total_pages": len(pages)}


@app.post("/api/wiki/reclassify", dependencies=[Depends(verify_api_token)])
async def reclassify_wiki():
    """重新分类所有 Wiki 页面 — 清空 taxonomy 后逐个用 LLM 动态分类（保留 pinned）"""
    from wiki.taxonomy import init_taxonomy_from_existing, _reclassify_running
    if _reclassify_running:
        raise HTTPException(status_code=409, detail="重分类正在进行中，请稍后再试")
    import asyncio
    asyncio.create_task(init_taxonomy_from_existing())
    return {"success": True, "message": "重分类已启动（后台执行中，pinned 页面保留）"}


@app.post("/api/wiki/taxonomy/move", dependencies=[Depends(verify_api_token)])
async def move_wiki_category(req: dict):
    """手动调整 Wiki 页面分类 — 标记 pinned，AI 重分类时跳过

    Body: {"page_path": "topics/xxx.md", "category": "分类名", "subcategory": "子分类名"}
    """
    page_path = req.get("page_path", "").strip()
    category = req.get("category", "").strip()
    if not page_path or not category:
        raise HTTPException(status_code=400, detail="需要 page_path 和 category")

    from wiki.taxonomy import move_page_category
    result = move_page_category(
        page_path=page_path,
        category=category,
        subcategory=req.get("subcategory", "").strip(),
    )
    return result


@app.get("/api/wiki/taxonomy/categories")
async def list_taxonomy_categories():
    """获取当前分类体系（含子分类列表）— 前端分类选择器用"""
    from wiki.taxonomy import load_taxonomy
    taxonomy = load_taxonomy()
    categories = taxonomy.get("categories", {})
    pinned = set(taxonomy.get("pinned_pages", []))

    result = []
    for cat_name, cat_data in categories.items():
        if not isinstance(cat_data, dict):
            continue
        children = list(cat_data.get("children", {}).keys())
        result.append({"name": cat_name, "children": children})
    return {"categories": result, "pinned_count": len(pinned)}


@app.delete("/api/wiki/page/{subdir}/{filename}", dependencies=[Depends(verify_api_token)])
async def delete_wiki_page(subdir: str, filename: str):
    """删除 Wiki 页面 — 同时清理分类树和向量索引"""
    page_path = _safe_wiki_path(subdir, filename)
    from config import WIKI_DIR
    filepath = WIKI_DIR / page_path
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Wiki 页面不存在")

    # 1. 删除文件
    filepath.unlink()

    # 2. 从分类树中移除 + 清理 pinned
    from wiki.taxonomy import load_taxonomy, save_taxonomy, _remove_page_from_all
    taxonomy = load_taxonomy()
    _remove_page_from_all(taxonomy, page_path)
    pinned = set(taxonomy.get("pinned_pages", []))
    pinned.discard(page_path)
    taxonomy["pinned_pages"] = sorted(pinned)
    save_taxonomy(taxonomy)

    # 3. 重建向量索引（移除该文件的 chunks）
    try:
        get_indexer().reindex_all()
    except Exception as e:
        logging.warning(f"Wiki 索引重建失败: {e}")

    return {"success": True, "message": f"已删除: {page_path}"}


# ===== 概念订阅 =====

@app.get("/api/wiki/subscriptions")
async def list_subscriptions():
    """获取概念订阅列表"""
    from config import WIKI_DIR
    import yaml
    sub_file = WIKI_DIR / "_subscriptions.yaml"
    if not sub_file.exists():
        return {"concepts": []}
    data = yaml.safe_load(sub_file.read_text(encoding="utf-8"))
    return {"concepts": data.get("concepts", []) if data else []}


@app.post("/api/wiki/subscriptions", dependencies=[Depends(verify_api_token)])
async def add_subscription(req: dict):
    """添加概念订阅 — 后续入库文章涉及该概念时自动关联

    Body: {"name": "概念名", "keywords": ["关键词1", "关键词2"], "description": "概念描述"}
    keywords 可选，默认用 name 做匹配
    """
    name = req.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="概念名不能为空")

    from config import WIKI_DIR
    import yaml
    sub_file = WIKI_DIR / "_subscriptions.yaml"

    if sub_file.exists():
        data = yaml.safe_load(sub_file.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    concepts = data.setdefault("concepts", [])
    # 去重
    if any(c.get("name") == name if isinstance(c, dict) else c == name for c in concepts):
        return {"success": True, "message": f"「{name}」已存在"}

    entry = {"name": name}
    keywords = req.get("keywords", [])
    if keywords:
        entry["keywords"] = keywords
    desc = req.get("description", "")
    if desc:
        entry["description"] = desc

    concepts.append(entry)
    yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    sub_file.write_text(yaml_str, encoding="utf-8")

    return {"success": True, "message": f"已订阅「{name}」", "total": len(concepts)}


@app.delete("/api/wiki/subscriptions/{name}", dependencies=[Depends(verify_api_token)])
async def remove_subscription(name: str):
    """取消概念订阅"""
    from config import WIKI_DIR
    import yaml
    sub_file = WIKI_DIR / "_subscriptions.yaml"
    if not sub_file.exists():
        raise HTTPException(status_code=404, detail="无订阅记录")

    data = yaml.safe_load(sub_file.read_text(encoding="utf-8")) or {}
    concepts = data.get("concepts", [])
    new_concepts = [c for c in concepts if (c.get("name") if isinstance(c, dict) else c) != name]
    if len(new_concepts) == len(concepts):
        raise HTTPException(status_code=404, detail=f"未找到订阅: {name}")

    data["concepts"] = new_concepts
    yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    sub_file.write_text(yaml_str, encoding="utf-8")
    return {"success": True, "message": f"已取消订阅「{name}」"}


@app.get("/api/wiki/log")
async def get_wiki_log():
    """获取 Wiki 操作日志"""
    from config import WIKI_DIR
    log_path = WIKI_DIR / "_log.md"
    if not log_path.exists():
        return {"content": ""}
    return {"content": log_path.read_text(encoding="utf-8")}


@app.get("/api/wiki/page/{subdir}/{filename}")
async def get_wiki_page(subdir: str, filename: str):
    """Wiki 页面详情"""
    from wiki.page_store import read_page
    page_path = _safe_wiki_path(subdir, filename)
    page = read_page(page_path)
    if not page:
        raise HTTPException(status_code=404, detail="Wiki 页面不存在")
    return page


@app.post("/api/wiki/compile-all", dependencies=[Depends(verify_api_token)])
async def compile_all_articles():
    """对 data/ 中所有文章执行全量 Wiki 编译（存量迁移用）"""
    from wiki.compile_queue import enqueue_compile
    count = 0
    for md_file in sorted(DATA_DIR.glob("*.md")):
        await enqueue_compile(md_file)
        count += 1
    return {"success": True, "queued": count, "message": f"已将 {count} 篇文章加入编译队列"}


@app.post("/api/wiki/inspect")
async def wiki_inspect():
    """Wiki 健康检查"""
    from wiki.inspector import inspect
    return inspect()


@app.put("/api/wiki/page/{subdir}/{filename}", dependencies=[Depends(verify_api_token)])
async def update_wiki_page(subdir: str, filename: str, req: dict):
    """编辑 Wiki 页面 — 修改正文内容"""
    page_path = _safe_wiki_path(subdir, filename)
    from wiki.page_store import read_page, create_page
    page = read_page(page_path)
    if not page:
        raise HTTPException(status_code=404, detail="Wiki 页面不存在")

    new_body = req.get("content", "").strip()
    if not new_body:
        raise HTTPException(status_code=400, detail="内容不能为空")

    # 重建页面：保留原 meta + 新 body
    import yaml
    from datetime import datetime
    meta = page.get("meta", {})
    meta["updated_at"] = datetime.now().strftime("%Y-%m-%d")
    yaml_str = yaml.dump(meta, default_flow_style=False, allow_unicode=True, sort_keys=False).strip()
    new_content = f"---\n{yaml_str}\n---\n\n{new_body}\n"

    from config import WIKI_DIR
    filepath = WIKI_DIR / page_path
    filepath.write_text(new_content, encoding="utf-8")

    # 重建该文件的向量索引
    try:
        get_indexer().index_file(filepath)
    except Exception as e:
        logging.warning(f"Wiki 索引重建失败: {e}")

    return {"success": True, "message": "已保存", "path": page_path}


# ===== 推送 & 通知 API =====

@app.get("/api/assistant/notifications")
async def get_notifications():
    """获取待推送通知（Web 端轮询）"""
    from assistant.scheduler import get_pending_notifications
    notifs = get_pending_notifications()
    return {"notifications": notifs}


@app.post("/api/assistant/push/{push_type}")
async def trigger_push_api(push_type: str):
    """手动触发推送 — weekly / review / association / health"""
    from assistant.scheduler import trigger_push
    content = await trigger_push(push_type)
    return {"success": bool(content), "content": content}


# ===== 知识库 API =====

@app.get("/api/knowledge")
async def list_knowledge():
    """知识列表 - 返回所有已落库文件的元数据"""
    files = get_engine().list_all()
    return {"items": files, "total": len(files)}


@app.get("/api/knowledge/{filename}")
async def get_knowledge(filename: str):
    """知识详情 - 返回某篇文件的元数据 + 正文"""
    filename = _safe_filename(filename)
    filepath = DATA_DIR / filename
    if not filepath.exists() or not filepath.suffix == ".md":
        raise HTTPException(status_code=404, detail="文件不存在")

    content = filepath.read_text(encoding="utf-8")

    # 解析 front-matter 和正文
    meta = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            import yaml
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()

    return {
        "filename": filename,
        "meta": meta,
        "content": body,
    }


@app.put("/api/knowledge/{filename}", dependencies=[Depends(verify_api_token)])
async def update_knowledge(filename: str, req: dict):
    """编辑知识文件 — 支持修改正文内容"""
    filename = _safe_filename(filename)
    filepath = DATA_DIR / filename
    if not filepath.exists() or not filepath.suffix == ".md":
        raise HTTPException(status_code=404, detail="文件不存在")

    import yaml
    from datetime import datetime

    content = filepath.read_text(encoding="utf-8")
    meta = {}
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}

    # 更新正文
    new_body = req.get("content", "").strip()
    if not new_body:
        raise HTTPException(status_code=400, detail="内容不能为空")

    # 更新 meta
    meta["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 重建文件
    yaml_str = yaml.dump(meta, default_flow_style=False, allow_unicode=True, sort_keys=False).strip()
    new_content = f"---\n{yaml_str}\n---\n\n{new_body}\n"
    filepath.write_text(new_content, encoding="utf-8")

    # 重建该文件的向量索引
    try:
        get_indexer().index_file(filepath)
    except Exception as e:
        logging.warning(f"索引重建失败: {e}")

    return {"success": True, "message": "已保存", "filename": filename}


@app.delete("/api/knowledge/{filename}", dependencies=[Depends(verify_api_token)])
async def delete_knowledge(filename: str):
    """删除知识文件"""
    filename = _safe_filename(filename)
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    filepath.unlink()
    # 重建索引
    try:
        get_indexer().reindex_all()
    except Exception as e:
        logging.warning(f"索引重建失败: {e}")
    return {"success": True, "message": f"已删除: {filename}"}


@app.get("/api/categories")
async def get_categories():
    """分类聚合 - 基于标签自动聚类"""
    files = get_engine().list_all()
    categories = {}
    category_rules = {
        "AI & 技术": ["AI", "人工智能", "技术", "深度学习", "机器学习", "模型", "算法", "编程", "开发", "架构", "工程"],
        "产品 & 方法论": ["产品", "方法论", "设计", "工作流", "SOP", "流程", "策略", "框架", "Coding"],
        "职场 & 行业": ["职场", "行业", "就业", "求职", "薪资", "焦虑", "转型", "职业"],
        "运营 & 增长": ["运营", "增长", "营销", "推广", "电商", "转化", "投放"],
        "内容 & 创作": ["内容", "创作", "写作", "文案", "短剧", "视频", "自媒体", "角色"],
        "工具 & 资源": ["工具", "资源", "分享", "推荐", "教程", "指南", "科技动态"],
    }
    for item in files:
        tags = item.get("tags", [])
        matched = False
        for cat_name, keywords in category_rules.items():
            for tag in tags:
                if any(kw in tag for kw in keywords):
                    categories.setdefault(cat_name, []).append(item)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            categories.setdefault("其他", []).append(item)

    result = [{"name": n, "count": len(items), "items": items}
              for n, items in categories.items()]
    result.sort(key=lambda x: -x["count"])
    return {"categories": result, "total": len(files)}


@app.get("/api/graph")
async def get_knowledge_graph():
    """知识图谱 - 节点(文章+标签) + 边(关联)"""
    files = get_engine().list_all()
    nodes, edges, tag_set, ftmap = [], [], set(), {}

    for item in files:
        fn = item.get("file_path", "").split("/")[-1]
        tags = item.get("tags", [])
        ftmap[fn] = set(tags)
        nodes.append({
            "id": fn, "label": item.get("title", "")[:30], "type": "article",
            "tags": tags, "summary": item.get("summary", ""),
            "platform": item.get("source_platform", ""),
            "date": item.get("created_at", ""), "url": item.get("source_url", ""),
        })
        for tag in tags:
            if tag not in tag_set:
                tag_set.add(tag)
                nodes.append({"id": f"tag:{tag}", "label": tag, "type": "tag"})
            edges.append({"source": fn, "target": f"tag:{tag}", "type": "has_tag"})

    fns = list(ftmap.keys())
    for i in range(len(fns)):
        for j in range(i + 1, len(fns)):
            shared = ftmap[fns[i]] & ftmap[fns[j]]
            if shared:
                edges.append({"source": fns[i], "target": fns[j],
                              "type": "related", "shared_tags": list(shared),
                              "weight": len(shared)})
    return {"nodes": nodes, "edges": edges}


@app.get("/api/tags")
async def get_tag_stats():
    """标签统计 - 用于标签云"""
    files = get_engine().list_all()
    tc = {}
    for item in files:
        for tag in item.get("tags", []):
            tc[tag] = tc.get(tag, 0) + 1
    tags = sorted([{"name": k, "count": v} for k, v in tc.items()],
                  key=lambda x: -x["count"])
    return {"tags": tags, "total_tags": len(tags)}


@app.get("/api/timeline")
async def get_timeline():
    """时间轴"""
    files = get_engine().list_all()
    items = [{"title": f.get("title", ""), "date": f.get("created_at", ""),
              "tags": f.get("tags", []), "platform": f.get("source_platform", ""),
              "file_path": f.get("file_path", ""), "summary": f.get("summary", "")}
             for f in files]
    items.sort(key=lambda x: x["date"], reverse=True)
    return {"items": items}


# ===== 数据导出 =====

@app.get("/api/export")
async def export_data():
    """导出知识库 — 打包为 ZIP（data/ + wiki/ 目录）"""
    import io
    import zipfile
    from fastapi.responses import StreamingResponse

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # data/ 目录
        for md_file in sorted(DATA_DIR.glob("*.md")):
            zf.write(md_file, f"data/{md_file.name}")
        # wiki/ 目录
        from config import WIKI_DIR
        for md_file in sorted(WIKI_DIR.glob("**/*.md")):
            rel = md_file.relative_to(WIKI_DIR)
            zf.write(md_file, f"wiki/{rel}")
        # taxonomy
        tax_path = WIKI_DIR / "_taxonomy.yaml"
        if tax_path.exists():
            zf.write(tax_path, "wiki/_taxonomy.yaml")

    buf.seek(0)
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=knowledge_export_{date_str}.zip"},
    )


# ===== 反馈闭环 =====

@app.post("/api/feedback")
async def submit_feedback(req: dict):
    """提交搜索反馈 — 用于 Few-shot 优化

    Body: {"query": "...", "answer": "...", "rating": "good"|"bad", "comment": "..."}
    """
    query = req.get("query", "").strip()
    rating = req.get("rating", "").strip()
    if not query or rating not in ("good", "bad"):
        raise HTTPException(status_code=400, detail="需要 query 和 rating(good/bad)")

    # 埋点：记录反馈
    from utils.analytics import track_feedback
    track_feedback(query=query, rating=rating)

    from config import BASE_DIR
    from datetime import datetime
    import json

    fb_dir = BASE_DIR / "outputs" / "feedback"
    fb_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "query": query,
        "answer": req.get("answer", ""),
        "rating": rating,
        "comment": req.get("comment", ""),
        "timestamp": datetime.now().isoformat(),
    }

    # 追加写入 JSONL
    fb_file = fb_dir / "feedback.jsonl"
    with open(fb_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Good 反馈自动写入 QA 缓存，下次相同问题直接返回
    if rating == "good" and record["answer"]:
        try:
            import hashlib
            qa_dir = BASE_DIR / "outputs" / "qa"
            qa_dir.mkdir(parents=True, exist_ok=True)
            qhash = hashlib.md5(query.encode()).hexdigest()[:8]
            date_str = datetime.now().strftime("%Y%m%d")
            cache_file = qa_dir / f"{date_str}_{qhash}.md"
            if not cache_file.exists():
                cache_file.write_text(
                    f"---\nquery: '{query}'\ncached_at: '{datetime.now().strftime('%Y-%m-%d %H:%M')}'\nsources_count: 0\nfrom_feedback: true\n---\n\n# Q: {query}\n\n{record['answer']}\n",
                    encoding="utf-8",
                )
                logger.info(f"Good 反馈已缓存: {cache_file.name}")
        except Exception as e:
            logger.debug(f"反馈缓存写入失败: {e}")

    return {"success": True, "message": "感谢反馈!"}


@app.get("/api/feedback")
async def list_feedback(limit: int = 50):
    """获取反馈记录 — 用于 Few-shot 注入"""
    from config import BASE_DIR
    fb_file = BASE_DIR / "outputs" / "feedback" / "feedback.jsonl"
    if not fb_file.exists():
        return {"items": [], "total": 0}

    records = []
    with open(fb_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    return {"items": records[-limit:], "total": len(records)}


# ===== 埋点统计 API =====

@app.get("/api/analytics/summary")
async def get_analytics_summary(days: int = 7):
    """获取统计概览"""
    from utils.analytics import get_stats_summary
    return get_stats_summary(days)


@app.get("/api/analytics/top-queries")
async def get_analytics_top_queries(days: int = 7, limit: int = 20):
    """获取热门搜索词"""
    from utils.analytics import get_top_queries
    return {"queries": get_top_queries(days, limit)}


@app.get("/api/analytics/no-result-queries")
async def get_analytics_no_result_queries(days: int = 7, limit: int = 20):
    """获取无结果的搜索词（内容缺口）"""
    from utils.analytics import get_no_result_queries
    return {"queries": get_no_result_queries(days, limit)}


@app.get("/api/analytics/trend")
async def get_analytics_trend(days: int = 7):
    """获取每日趋势"""
    from utils.analytics import get_daily_trend
    return {"trend": get_daily_trend(days)}


@app.get("/api/analytics/events")
async def get_analytics_events(limit: int = 50, event_type: str = None):
    """获取最近事件列表"""
    from utils.analytics import get_recent_events
    return {"events": get_recent_events(limit, event_type)}


# ===== 发现模块 API =====

@app.get("/api/discover")
async def get_discover_items(
    status: str = "pending",
    page: int = 1,
    page_size: int = 20,
    source: Optional[str] = None
):
    """获取待审核内容列表"""
    from discovery.store import get_pending_items
    return get_pending_items(status=status, page=page, page_size=page_size, source=source)


@app.post("/api/discover/refresh", dependencies=[Depends(verify_api_token)])
async def refresh_discover():
    """手动刷新热榜（抓取新内容）"""
    try:
        from discovery.crawler import fetch_all_sources
        from discovery.store import add_items

        items = await fetch_all_sources()
        added = add_items(items)

        return {
            "success": True,
            "fetched": len(items),
            "added": added,
            "message": f"抓取 {len(items)} 条，新增 {added} 条"
        }
    except Exception as e:
        logger.error(f"刷新热榜失败: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/discover/ingest", dependencies=[Depends(verify_api_token)])
async def ingest_discover_item(req: dict):
    """入库选中的内容"""
    item_id = req.get("id")
    if not item_id:
        raise HTTPException(status_code=400, detail="缺少 id 参数")

    from discovery.store import get_item_by_id, update_status

    item = get_item_by_id(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="内容不存在")

    url = item.get("url", "")
    if not url:
        raise HTTPException(status_code=400, detail="该内容无有效链接")

    # 调用入库流程
    try:
        from services.ingest_pipeline import ingest_url
        result = await ingest_url(url, skip_duplicate=False, force=True)

        if result.get("success"):
            update_status(item_id, "ingested")
            return {"success": True, "message": "入库成功", "result": result}
        else:
            return {"success": False, "error": result.get("error", "入库失败")}

    except Exception as e:
        logger.error(f"入库失败: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/discover/ingest-batch", dependencies=[Depends(verify_api_token)])
async def ingest_discover_batch(req: dict):
    """批量入库 — 并发执行（限制并发数）"""
    item_ids = req.get("ids", [])
    if not item_ids:
        raise HTTPException(status_code=400, detail="缺少 ids 参数")

    from discovery.store import get_item_by_id, update_status
    import asyncio

    # 并发限制：最多 3 个并发入库
    MAX_CONCURRENT = 3
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results = []

    async def ingest_one(item_id: str) -> dict:
        """入库单条内容"""
        item = get_item_by_id(item_id)
        if not item or not item.get("url"):
            return {"id": item_id, "success": False, "error": "内容不存在或无链接"}

        async with semaphore:
            try:
                from services.ingest_pipeline import ingest_url
                result = await ingest_url(item["url"], skip_duplicate=False, force=False)
                if result.get("success"):
                    update_status(item_id, "ingested")
                    return {"id": item_id, "success": True, "title": result.get("title")}
                else:
                    return {"id": item_id, "success": False, "error": result.get("error")}
            except Exception as e:
                return {"id": item_id, "success": False, "error": str(e)}

    # 并发执行所有入库任务
    tasks = [ingest_one(item_id) for item_id in item_ids]
    results = await asyncio.gather(*tasks)

    success_count = sum(1 for r in results if r["success"])
    return {
        "success": True,
        "total": len(item_ids),
        "ingested": success_count,
        "results": results
    }


@app.post("/api/discover/ingest-batch-stream", dependencies=[Depends(verify_api_token)])
async def ingest_discover_batch_stream(req: dict):
    """批量入库 — SSE 流式进度推送"""
    item_ids = req.get("ids", [])
    if not item_ids:
        raise HTTPException(status_code=400, detail="缺少 ids 参数")

    from discovery.store import get_item_by_id, update_status
    import asyncio

    MAX_CONCURRENT = 3
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def event_generator():
        results = []
        completed = 0
        total = len(item_ids)

        async def ingest_one(item_id: str) -> dict:
            nonlocal completed
            item = get_item_by_id(item_id)
            if not item or not item.get("url"):
                completed += 1
                return {"id": item_id, "success": False, "error": "内容不存在或无链接"}

            async with semaphore:
                try:
                    from services.ingest_pipeline import ingest_url
                    result = await ingest_url(item["url"], skip_duplicate=False, force=False)
                    completed += 1
                    if result.get("success"):
                        update_status(item_id, "ingested")
                        return {"id": item_id, "success": True, "title": result.get("title")}
                    else:
                        return {"id": item_id, "success": False, "error": result.get("error")}
                except Exception as e:
                    completed += 1
                    return {"id": item_id, "success": False, "error": str(e)}

        # 创建所有任务
        tasks = []
        for item_id in item_ids:
            task = asyncio.create_task(ingest_one(item_id))
            tasks.append((item_id, task))

        # 等待所有任务完成，定期推送进度
        while any(not t.done() for _, t in tasks):
            progress_data = {
                "type": "progress",
                "total": total,
                "completed": completed,
                "percent": int(completed / total * 100) if total > 0 else 0
            }
            yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.5)

        # 收集结果
        for item_id, task in tasks:
            try:
                results.append(task.result())
            except Exception as e:
                results.append({"id": item_id, "success": False, "error": str(e)})

        # 发送最终结果
        success_count = sum(1 for r in results if r["success"])
        final_data = {
            "type": "done",
            "total": total,
            "ingested": success_count,
            "failed": total - success_count,
            "results": results
        }
        yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/discover/ignore", dependencies=[Depends(verify_api_token)])
async def ignore_discover_item(req: dict):
    """忽略内容"""
    item_id = req.get("id")
    if not item_id:
        raise HTTPException(status_code=400, detail="缺少 id 参数")

    from discovery.store import update_status
    success = update_status(item_id, "ignored")
    return {"success": success}


@app.post("/api/discover/ignore-batch", dependencies=[Depends(verify_api_token)])
async def ignore_discover_batch(req: dict):
    """批量忽略"""
    item_ids = req.get("ids", [])
    if not item_ids:
        raise HTTPException(status_code=400, detail="缺少 ids 参数")

    from discovery.store import batch_update_status
    updated = batch_update_status(item_ids, "ignored")
    return {"success": True, "updated": updated}


@app.get("/api/discover/stats")
async def get_discover_stats():
    """获取发现模块统计"""
    from discovery.store import get_stats
    return get_stats()


@app.get("/api/discover/sources")
async def get_discover_sources():
    """获取可用数据源"""
    return {
        "sources": [
            {"id": "hackernews", "name": "Hacker News AI", "type": "api"},
            {"id": "github", "name": "GitHub Trending", "type": "api"},
            {"id": "zhihu", "name": "知乎热榜", "type": "crawler"},
            {"id": "v2ex", "name": "V2EX 热门", "type": "api"},
        ]
    }


# ===== 静态文件 =====
WEB_DIR = BASE_DIR / "web"
WEB_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
