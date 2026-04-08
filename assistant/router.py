"""AI 助手 API 路由

提供:
- POST /api/assistant/chat      — 流式对话 (SSE)
- POST /api/assistant/chat-sync — 非流式对话
- GET  /api/assistant/sessions   — 会话管理
- DELETE /api/assistant/sessions/{id} — 清空会话
"""

import uuid
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from assistant.intent import detect_intent
from assistant.chat_engine import (
    chat_stream, chat_once, clear_session,
    get_session, cleanup_expired_sessions,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/assistant", tags=["智能助手"])


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatSyncResponse(BaseModel):
    session_id: str
    intent: str
    answer: str
    action: Optional[dict] = None


# ===== 意图分发 → 上下文构建 =====

async def _build_context(intent_result: dict) -> tuple[str, Optional[dict]]:
    """根据意图构建上下文信息 + 可选 action

    Returns:
        (context_str, action_dict_or_none)
    """
    intent = intent_result["intent"]
    params = intent_result["params"]
    action = None

    if intent == "search":
        query = params.get("query", "")
        if query:
            try:
                from retrieval.hybrid_searcher import HybridSearcher
                from retrieval.indexer import VectorIndexer
                searcher = HybridSearcher(indexer=VectorIndexer())
                result = await searcher.search(query=query, top_k=3)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                source_list = "\n".join(
                    f"  [{i+1}] {s.get('title', '未知')}" for i, s in enumerate(sources)
                )
                context = f"知识库搜索结果:\n\n{answer}\n\n参考来源:\n{source_list}"
                return context, None
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return f"搜索出错: {e}", None

    elif intent == "ingest":
        urls = params.get("urls", [])
        if urls:
            action = {"type": "ingest", "urls": urls}
            return f"用户想入库以下链接: {', '.join(urls)}", action

    elif intent == "stats":
        try:
            from config import DATA_DIR, WIKI_DIR
            from retrieval.indexer import VectorIndexer
            indexer = VectorIndexer()
            indexer_stats = indexer.get_stats()
            file_count = len(list(DATA_DIR.glob("*.md")))
            wiki_count = sum(1 for _ in WIKI_DIR.glob("**/*.md") if not _.name.startswith("_"))
            try:
                from wiki.compile_queue import get_queue
                queue_size = get_queue().qsize()
            except Exception:
                queue_size = 0
            context = (
                f"系统状态:\n"
                f"- 知识文章: {file_count} 篇\n"
                f"- Wiki 页面: {wiki_count} 个\n"
                f"- 向量切片: {indexer_stats.get('total_chunks', 0)} 个\n"
                f"- 编译队列: {queue_size} 个任务\n"
            )
            return context, None
        except Exception as e:
            return f"获取状态失败: {e}", None

    elif intent == "wiki":
        try:
            from wiki.page_store import list_wiki_pages
            pages = list_wiki_pages()
            page_list = "\n".join(f"  - {p.get('title', '')} ({p.get('type', '')})" for p in pages[:10])
            context = f"Wiki 页面列表 ({len(pages)} 个):\n{page_list}"
            return context, None
        except Exception as e:
            return f"Wiki 查询失败: {e}", None

    return "", None


# ===== API 端点 =====

@router.post("/chat")
async def chat_sse(req: ChatRequest):
    """流式对话 — SSE"""
    session_id = req.session_id or str(uuid.uuid4())
    message = req.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    # 清理过期会话
    cleanup_expired_sessions()

    # 意图识别
    intent_result = detect_intent(message)
    logger.info(f"Intent: {intent_result['intent']} | Message: {message[:50]}")

    # 构建上下文
    context, action = await _build_context(intent_result)

    async def event_generator():
        # 先发送元信息
        meta = {
            "type": "meta",
            "session_id": session_id,
            "intent": intent_result["intent"],
        }
        if action:
            meta["action"] = action
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        # 如果是入库动作，先触发入库
        if action and action.get("type") == "ingest":
            context_extra = ""
            for url in action.get("urls", []):
                try:
                    yield f"data: {json.dumps({'type': 'status', 'text': f'正在入库: {url[:60]}...'}, ensure_ascii=False)}\n\n"
                    from ingestion.router import FetcherRouter
                    from transform.llm_cleaner import LLMCleaner
                    from storage.markdown_engine import MarkdownEngine
                    from retrieval.indexer import VectorIndexer
                    from utils.url_utils import check_duplicate
                    from config import DATA_DIR

                    existing = check_duplicate(url, DATA_DIR)
                    if existing:
                        yield f"data: {json.dumps({'type': 'status', 'text': f'⚠️ 该链接已入库'}, ensure_ascii=False)}\n\n"
                        context_extra += f"\n链接 {url} 已存在于知识库中。"
                        continue

                    router_inst = FetcherRouter()
                    raw = await router_inst.fetch(url)
                    cleaner = LLMCleaner()
                    knowledge = await cleaner.clean(
                        title=raw.title, content=raw.content,
                        source=raw.source_platform, author=raw.author,
                        original_tags=raw.original_tags,
                    )
                    engine = MarkdownEngine()
                    filepath = engine.save(
                        knowledge=knowledge, source_url=url,
                        source_platform=raw.source_platform, author=raw.author,
                    )
                    indexer = VectorIndexer()
                    indexer.index_file(filepath)

                    yield f"data: {json.dumps({'type': 'status', 'text': f'✅ 入库成功: {knowledge.title}'}, ensure_ascii=False)}\n\n"
                    context_extra += f"\n入库结果: 「{knowledge.title}」入库成功，标签: {', '.join(knowledge.tags)}"

                    # Wiki 编译入队
                    try:
                        from wiki.compile_queue import enqueue_compile
                        await enqueue_compile(filepath)
                    except Exception:
                        pass

                except Exception as e:
                    yield f"data: {json.dumps({'type': 'status', 'text': f'❌ 入库失败: {str(e)[:80]}'}, ensure_ascii=False)}\n\n"
                    context_extra += f"\n入库失败: {str(e)[:100]}"
            final_context = context + context_extra
        else:
            final_context = context

        # 流式 LLM 回复
        async for chunk in chat_stream(session_id, message, final_context):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat-sync", response_model=ChatSyncResponse)
async def chat_sync(req: ChatRequest):
    """非流式对话"""
    session_id = req.session_id or str(uuid.uuid4())
    message = req.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    cleanup_expired_sessions()
    intent_result = detect_intent(message)
    context, action = await _build_context(intent_result)

    # 入库动作
    if action and action.get("type") == "ingest":
        for url in action.get("urls", []):
            try:
                from ingestion.router import FetcherRouter
                from transform.llm_cleaner import LLMCleaner
                from storage.markdown_engine import MarkdownEngine
                from retrieval.indexer import VectorIndexer
                from utils.url_utils import check_duplicate
                from config import DATA_DIR

                existing = check_duplicate(url, DATA_DIR)
                if existing:
                    continue
                router_inst = FetcherRouter()
                raw = await router_inst.fetch(url)
                cleaner = LLMCleaner()
                knowledge = await cleaner.clean(
                    title=raw.title, content=raw.content,
                    source=raw.source_platform, author=raw.author,
                    original_tags=raw.original_tags,
                )
                engine = MarkdownEngine()
                filepath = engine.save(
                    knowledge=knowledge, source_url=url,
                    source_platform=raw.source_platform, author=raw.author,
                )
                indexer = VectorIndexer()
                indexer.index_file(filepath)
            except Exception:
                pass

    answer = await chat_once(session_id, message, context)
    return ChatSyncResponse(
        session_id=session_id,
        intent=intent_result["intent"],
        answer=answer,
        action=action,
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """清空会话"""
    clear_session(session_id)
    return {"success": True, "message": f"会话 {session_id} 已清空"}
