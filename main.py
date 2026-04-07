"""FastAPI 主入口

暴露两个核心端点:
- POST /ingest  - 入库 (URL -> 抓取 -> 清洗 -> 落库 -> 索引)
- POST /search  - 检索 (查询 -> 向量召回 -> LLM 答案生成)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Optional

from config import DATA_DIR
from ingestion.router import FetcherRouter
from transform.llm_cleaner import LLMCleaner
from storage.markdown_engine import MarkdownEngine
from retrieval.indexer import VectorIndexer
from retrieval.searcher import RAGSearcher
from utils.url_utils import normalize_url, check_duplicate

app = FastAPI(
    title="Local RAG - 个人碎片知识落库系统",
    description="本地化优先的个人知识资产管理与 RAG 问答系统",
    version="1.0.0",
)

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
        _searcher = RAGSearcher(indexer=get_indexer())
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
    sources: list[dict]


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

    # 5. 向量索引
    try:
        chunk_count = get_indexer().index_file(filepath)
    except Exception as e:
        # 索引失败不阻断流程，文件已保存
        chunk_count = 0

    return IngestResponse(
        success=True,
        file_path=str(filepath),
        title=knowledge.title,
        tags=knowledge.tags,
        message=f"入库成功，生成 {chunk_count} 个索引切片",
    )


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
    return {
        "knowledge_files": file_count,
        **indexer_stats,
    }


@app.get("/")
async def root():
    return {
        "name": "Local RAG - 个人碎片知识落库系统",
        "version": "1.0.0",
        "endpoints": {
            "POST /ingest": "入库 - 发送 URL 自动抓取清洗落库",
            "POST /search": "检索 - 自然语言查询知识库",
            "POST /reindex": "重建索引 - 从 .md 文件恢复",
            "GET /stats": "系统状态统计",
        },
    }
