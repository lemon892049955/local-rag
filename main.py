"""FastAPI 主入口

暴露核心端点:
- POST /ingest       - 入库 (URL -> 抓取 -> 清洗 -> 落库 -> 索引)
- POST /ingest/text  - 手动入库
- POST /search       - 检索 (查询 -> 向量召回 -> LLM 答案生成)
- GET  /api/knowledge      - 知识列表
- GET  /api/knowledge/{id} - 知识详情
"""

from pathlib import Path
import logging
import re
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


def _safe_filename(filename: str) -> str:
    """校验文件名安全性，防止路径穿越"""
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    if not filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="仅支持 .md 文件")
    return filename


def _safe_wiki_path(subdir: str, filename: str) -> str:
    """校验 Wiki 路径安全性"""
    if subdir not in ("topics", "entities", "insights"):
        raise HTTPException(status_code=400, detail="非法目录")
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    return f"{subdir}/{filename}"

app = FastAPI(
    title="Local RAG - 个人碎片知识落库系统",
    description="本地化优先的个人知识资产管理与 RAG 问答系统（v0.5 智能助手）",
    version="0.6.3",
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

    from services.ingest_pipeline import ingest_url
    result = await ingest_url(url)

    if not result.get("success") and not result.get("duplicate"):
        raise HTTPException(status_code=422, detail=result.get("error", "入库失败"))

    return IngestResponse(
        success=True,
        file_path=result.get("file_path", ""),
        title=result.get("title", ""),
        tags=result.get("tags", []),
        message=result.get("message", ""),
    )


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(file: UploadFile = File(...)):
    """文件上传入库 — 支持 PDF/图片/音频"""
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
        result = await ingest_raw(raw, source_url=f"file://{file.filename}")

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
    return {"name": "Local RAG", "version": "0.6.3", "hint": "Web UI not found, visit /docs"}


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
    """Wiki 目录树 — 按标签自动聚类

    遍历 Wiki 页面的 sources 字段，找到对应文章的标签，
    用出现频次最高的标签作为该 Wiki 页面的分类文件夹。
    """
    from wiki.page_store import list_wiki_pages as _list
    from collections import Counter

    pages = _list()
    files = get_engine().list_all()

    # 建立 filename → tags 的映射
    file_tags = {}
    for item in files:
        fp = item.get("file_path", "")
        fn = fp.split("/")[-1] if fp else ""
        file_tags[fn] = item.get("tags", [])

    # 为每个 Wiki 页面计算最佳归属文件夹
    folders = {}  # folder_name -> [page, ...]
    uncategorized = []

    for page in pages:
        sources = page.get("sources", [])
        # 统计该页面所有来源文章的标签频次
        tag_counter = Counter()
        for src in sources:
            for tag in file_tags.get(src, []):
                tag_counter[tag] += 1

        # 也考虑页面标题本身包含的关键词
        page_title = page.get("title", "")

        if tag_counter:
            # 取频次最高的标签作为分类
            best_tag = tag_counter.most_common(1)[0][0]
            folders.setdefault(best_tag, []).append(page)
        else:
            uncategorized.append(page)

    # 按文件夹内页面数排序（多的排前面）
    sorted_folders = sorted(folders.items(), key=lambda x: -len(x[1]))

    # 合并过小的文件夹（只有1个页面的合并到"其他"）
    result = []
    other_pages = list(uncategorized)
    for folder_name, folder_pages in sorted_folders:
        if len(folder_pages) >= 1:
            result.append({
                "name": folder_name,
                "count": len(folder_pages),
                "pages": folder_pages,
            })
        else:
            other_pages.extend(folder_pages)

    if other_pages:
        result.append({
            "name": "其他",
            "count": len(other_pages),
            "pages": other_pages,
        })

    return {"folders": result, "total_pages": len(pages)}


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


@app.put("/api/knowledge/{filename}")
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


@app.delete("/api/knowledge/{filename}")
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


# ===== 静态文件 =====
WEB_DIR = BASE_DIR / "web"
WEB_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
