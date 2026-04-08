"""FastAPI 主入口

暴露核心端点:
- POST /ingest       - 入库 (URL -> 抓取 -> 清洗 -> 落库 -> 索引)
- POST /ingest/text  - 手动入库
- POST /search       - 检索 (查询 -> 向量召回 -> LLM 答案生成)
- GET  /api/knowledge      - 知识列表
- GET  /api/knowledge/{id} - 知识详情
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional

from config import DATA_DIR, BASE_DIR
from ingestion.router import FetcherRouter
from transform.llm_cleaner import LLMCleaner
from storage.markdown_engine import MarkdownEngine
from retrieval.indexer import VectorIndexer
from retrieval.hybrid_searcher import HybridSearcher
from utils.url_utils import normalize_url, check_duplicate

app = FastAPI(
    title="Local RAG - 个人碎片知识落库系统",
    description="本地化优先的个人知识资产管理与 RAG 问答系统（v0.5 智能助手）",
    version="0.5.0",
)

# 注册企业微信回调路由
from wecom.callback import router as wecom_router
app.include_router(wecom_router)

# 注册 AI 助手路由
from assistant.router import router as assistant_router
app.include_router(assistant_router)


# ===== Wiki 编译 Worker + 推送调度器启动 =====
@app.on_event("startup")
async def startup_event():
    from wiki.compile_queue import start_compile_worker
    await start_compile_worker()
    from assistant.scheduler import start_scheduler
    await start_scheduler()


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


# ===== 端点 =====

@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """入库接口 - URL -> 抓取 -> 清洗 -> 落库 -> 索引"""

    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL 不能为空")

    # 1. 去重检查
    existing = check_duplicate(url, DATA_DIR)
    if existing:
        return IngestResponse(
            success=True,
            file_path=existing,
            title="(已存在)",
            tags=[],
            message=f"该 URL 已入库: {existing}",
        )

    # 2. 抓取
    try:
        raw = await get_router().fetch(url)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"抓取失败: {e}")

    # 3. LLM 清洗
    try:
        knowledge = await get_cleaner().clean(
            title=raw.title,
            content=raw.content,
            source=raw.source_platform,
            author=raw.author,
            original_tags=raw.original_tags,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 清洗失败: {e}")

    # 4. Markdown 落库
    try:
        filepath = get_engine().save(
            knowledge=knowledge,
            source_url=url,
            source_platform=raw.source_platform,
            author=raw.author,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    # 5. 向量索引 + BM25 索引
    try:
        chunk_count = get_indexer().index_file(filepath)
        get_searcher().rebuild_bm25()
    except Exception as e:
        chunk_count = 0

    # 6. Wiki 编译入队（异步，不阻塞响应）
    try:
        from wiki.compile_queue import enqueue_compile
        await enqueue_compile(filepath)
    except Exception:
        pass

    return IngestResponse(
        success=True,
        file_path=str(filepath),
        title=knowledge.title,
        tags=knowledge.tags,
        message=f"入库成功，生成 {chunk_count} 个索引切片，Wiki 编译已排队",
    )


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(file: UploadFile = File(...)):
    """文件上传入库 — 支持 PDF/图片/音频

    multipart/form-data 上传文件，自动识别类型并解析。
    """
    from fastapi import UploadFile
    import tempfile

    if file is None:
        raise HTTPException(status_code=400, detail="请上传文件")

    # 保存到临时文件
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

        # 解析文件 → RawContent
        raw = await dispatcher.dispatch(str(tmp_path))

        # LLM 清洗
        knowledge = await get_cleaner().clean(
            title=raw.title,
            content=raw.content,
            source=raw.source_platform,
            author=raw.author,
            original_tags=raw.original_tags,
        )

        # 落库
        filepath = get_engine().save(
            knowledge=knowledge,
            source_url=f"file://{file.filename}",
            source_platform=raw.source_platform,
            author=raw.author,
        )

        # 索引
        chunk_count = get_indexer().index_file(filepath)
        get_searcher().rebuild_bm25()

        # Wiki 编译入队
        try:
            from wiki.compile_queue import enqueue_compile
            await enqueue_compile(filepath)
        except Exception:
            pass

        return IngestResponse(
            success=True,
            file_path=str(filepath),
            title=knowledge.title,
            tags=knowledge.tags,
            message=f"{file_type.upper()} 入库成功，{len(raw.content)} 字，Wiki 编译已排队",
        )
    finally:
        # 删除临时文件节省磁盘
        try:
            tmp_path.unlink()
        except Exception:
            pass


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """检索接口 - 语义搜索 + RAG 答案生成"""

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    result = await get_searcher().search(
        query=req.query,
        top_k=req.top_k,
    )

    return SearchResponse(
        answer=result["answer"],
        sources=result["sources"],
        debug=result.get("debug", {}),
    )


@app.post("/reindex")
async def reindex():
    """重建索引 - 从 Markdown 文件恢复 ChromaDB"""
    total = get_indexer().reindex_all()
    return {"success": True, "total_chunks": total}


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
    return {"name": "Local RAG", "version": "0.5.0", "hint": "Web UI not found, visit /docs"}


# ===== Wiki API =====

@app.get("/api/wiki/pages")
async def list_wiki_pages():
    """Wiki 页面列表"""
    from wiki.page_store import list_wiki_pages as _list
    pages = _list()
    return {"pages": pages, "total": len(pages)}


@app.get("/api/wiki/graph")
async def get_wiki_graph():
    """Wiki 知识图谱 — 节点=Wiki页面，边=[[交叉引用]]关系"""
    import re
    from wiki.page_store import list_wiki_pages, read_page

    pages = list_wiki_pages()
    nodes = []
    edges = []
    title_to_path = {}

    for p in pages:
        path = p.get("path", "")
        title = p.get("title", "")
        title_to_path[title] = path
        nodes.append({
            "id": path,
            "label": title[:20],
            "title": title,
            "type": p.get("type", "topic"),
            "summary": p.get("summary", ""),
            "sources_count": len(p.get("sources", [])),
            "updated_at": p.get("updated_at", ""),
        })

    # 扫描交叉引用 [[]]
    for p in pages:
        path = p.get("path", "")
        page_data = read_page(path)
        if not page_data:
            continue
        refs = re.findall(r"\[\[(.+?)\]\]", page_data.get("full_content", ""))
        for ref in refs:
            target_path = title_to_path.get(ref, "")
            if target_path and target_path != path:
                edges.append({"source": path, "target": target_path, "type": "reference"})

    # 共享来源关联
    path_sources = {}
    for p in pages:
        path_sources[p.get("path", "")] = set(p.get("sources", []))
    paths = list(path_sources.keys())
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            shared = path_sources[paths[i]] & path_sources[paths[j]]
            if shared:
                edges.append({
                    "source": paths[i], "target": paths[j],
                    "type": "shared_source", "weight": len(shared),
                })

    return {"nodes": nodes, "edges": edges}


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
    page_path = f"{subdir}/{filename}"
    page = read_page(page_path)
    if not page:
        raise HTTPException(status_code=404, detail="Wiki 页面不存在")
    return page


@app.post("/api/wiki/compile-all")
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


@app.delete("/api/knowledge/{filename}")
async def delete_knowledge(filename: str):
    """删除知识文件"""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    filepath.unlink()
    # 重建索引
    try:
        get_indexer().reindex_all()
    except Exception:
        pass
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


# ===== 静态文件 =====
WEB_DIR = BASE_DIR / "web"
WEB_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
