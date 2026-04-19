"""AI 助手对话引擎

支持多轮对话、意图分发、流式输出。
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, List, Dict, Optional

from openai import OpenAI
from config import get_llm_config

logger = logging.getLogger(__name__)

# 复用 OpenAI client（避免每次对话 new 一个）
_client = None

def _get_client():
    global _client
    if _client is None:
        config = get_llm_config()
        _client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    return _client


SYSTEM_PROMPT = """你是 BuddyKnow 的智能助手，一个个人知识库管理系统的 AI 伙伴。
你帮助用户管理和查询从公众号、小红书、知乎等平台收集的碎片知识。

你的能力:
1. **知识搜索** — 用户提问时，基于知识库中的文章和 Wiki 页面回答
2. **入库协助** — 用户粘贴链接时，自动触发入库流程
3. **系统查询** — 查看知识库统计、Wiki 页面列表等
4. **智能闲聊** — 在没有明确知识库需求时，友好交流

规则:
- 优先使用知识库中的内容回答，不要编造
- **回答中必须标注信息来源**，格式为 `[来源: 文章标题]`，放在相关段落末尾
- 如果检索结果中有多个来源，分别标注
- 如果知识库中没有相关信息，坦诚说明"知识库中暂未找到相关内容"
- 回答简洁直接，使用中文，适当使用 emoji
- 每次回复控制在 500 字以内"""

# 会话存储 (内存级，重启清空)
_sessions: Dict[str, List[dict]] = {}
_SESSION_MAX_TURNS = 20
_SESSION_TTL = 3600  # 1 小时


def get_session(session_id: str) -> List[dict]:
    """获取或创建会话历史"""
    if session_id not in _sessions:
        _sessions[session_id] = []
    return _sessions[session_id]


def add_message(session_id: str, role: str, content: str):
    """添加消息到会话"""
    history = get_session(session_id)
    history.append({"role": role, "content": content, "ts": time.time()})
    # 保留最近 N 轮
    if len(history) > _SESSION_MAX_TURNS * 2:
        _sessions[session_id] = history[-_SESSION_MAX_TURNS * 2:]


def clear_session(session_id: str):
    """清空会话"""
    _sessions.pop(session_id, None)


def cleanup_expired_sessions():
    """清理过期会话"""
    now = time.time()
    expired = []
    for sid, msgs in _sessions.items():
        if msgs and msgs[-1].get("ts", 0) < now - _SESSION_TTL:
            expired.append(sid)
    for sid in expired:
        del _sessions[sid]


async def chat_stream(
    session_id: str,
    user_message: str,
    context: str = "",
) -> AsyncGenerator[str, None]:
    """流式对话 — 返回 SSE 数据块

    Args:
        session_id: 会话 ID
        user_message: 用户输入
        context: 额外上下文 (搜索结果/系统状态等)

    Yields:
        SSE 格式字符串
    """
    config = get_llm_config()
    client = _get_client()

    # 构建消息列表
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 加入会话历史
    history = get_session(session_id)
    for msg in history[-_SESSION_MAX_TURNS * 2:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 当前用户消息
    if context:
        user_content = f"{user_message}\n\n---以下是系统提供的参考信息---\n{context}"
    else:
        user_content = user_message

    messages.append({"role": "user", "content": user_content})

    # 记录用户消息
    add_message(session_id, "user", user_message)

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            temperature=0.5,
            max_tokens=1500,
            stream=True,
        )

        full_answer = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                full_answer += delta.content
                yield f"data: {json.dumps({'type': 'content', 'text': delta.content}, ensure_ascii=False)}\n\n"

        # 记录助手回复
        add_message(session_id, "assistant", full_answer)

        yield f"data: {json.dumps({'type': 'done', 'text': full_answer}, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        error_msg = f"抱歉，AI 回复出错了: {str(e)[:100]}"
        add_message(session_id, "assistant", error_msg)
        yield f"data: {json.dumps({'type': 'error', 'text': error_msg}, ensure_ascii=False)}\n\n"


async def chat_once(
    session_id: str,
    user_message: str,
    context: str = "",
) -> str:
    """非流式对话 — 返回完整回复"""
    full = ""
    async for chunk in chat_stream(session_id, user_message, context):
        if chunk.startswith("data: "):
            try:
                data = json.loads(chunk[6:].strip())
                if data.get("type") == "content":
                    full += data.get("text", "")
            except Exception:
                pass
    return full
