"""企业微信消息回调处理

接收微信转发的消息 → 提取链接 → 调用入库 → 推送结果通知
"""

import re
import time
import logging
from typing import Optional
from collections import OrderedDict

from fastapi import APIRouter, Request, Response, BackgroundTasks

from wecom.crypto import WXBizMsgCrypt, parse_xml_msg
from wecom.sender import notify_ingest_success, notify_ingest_fail, send_text_msg
from config import get_wecom_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/wecom", tags=["企业微信"])

# 消息去重缓存：MsgId -> 处理时间，防止企微重试导致重复处理
_msg_cache = OrderedDict()
_MSG_CACHE_MAX = 500
_MSG_CACHE_TTL = 300  # 5 分钟


def _is_duplicate(msg_id: str) -> bool:
    """检查消息是否重复"""
    if not msg_id:
        return False
    now = time.time()
    # 清理过期缓存
    while _msg_cache and next(iter(_msg_cache.values())) < now - _MSG_CACHE_TTL:
        _msg_cache.popitem(last=False)
    if msg_id in _msg_cache:
        return True
    _msg_cache[msg_id] = now
    if len(_msg_cache) > _MSG_CACHE_MAX:
        _msg_cache.popitem(last=False)
    return False


def _get_crypt() -> WXBizMsgCrypt:
    cfg = get_wecom_config()
    return WXBizMsgCrypt(
        token=cfg["token"],
        encoding_aes_key=cfg["encoding_aes_key"],
        corp_id=cfg["corp_id"],
    )


# ===== URL 验证 (GET) =====

@router.get("/callback")
async def verify_url(msg_signature: str, timestamp: str, nonce: str, echostr: str):
    """企微回调 URL 验证"""
    try:
        crypt = _get_crypt()
        reply = crypt.verify_url(msg_signature, timestamp, nonce, echostr)
        return Response(content=reply, media_type="text/plain")
    except Exception as e:
        logger.error(f"URL 验证失败: {e}")
        return Response(content="error", status_code=403)


# ===== 消息接收 (POST) =====

@router.post("/callback")
async def receive_msg(
    request: Request,
    background_tasks: BackgroundTasks,
    msg_signature: str = "",
    timestamp: str = "",
    nonce: str = "",
):
    """接收企微回调消息，异步处理入库"""
    body = await request.body()
    body_str = body.decode("utf-8")

    try:
        crypt = _get_crypt()
        xml_str = crypt.decrypt_msg(msg_signature, timestamp, nonce, body_str)
        msg = parse_xml_msg(xml_str)
    except Exception as e:
        logger.error(f"消息解密失败: {e}")
        return Response(content="success", media_type="text/plain")

    msg_type = msg.get("MsgType", "")
    from_user = msg.get("FromUserName", "")
    content = msg.get("Content", "")
    msg_id = msg.get("MsgId", "")

    # 去重：同一条消息只处理一次
    if _is_duplicate(msg_id):
        return Response(content="success", media_type="text/plain")

    logger.info(f"收到消息: type={msg_type}, from={from_user}, content={content[:100]}")

    # 处理文本消息 — 意图识别分发
    if msg_type == "text" and content:
        # 推送订阅管理
        if content.strip() in ("订阅推送", "开启推送"):
            from assistant.scheduler import add_subscriber
            add_subscriber(from_user)
            send_text_msg(from_user, "🔔 已开启推送通知！你将收到每周摘要、知识回顾等推送。\n发「取消推送」可关闭。")
            return Response(content="success", media_type="text/plain")
        if content.strip() in ("取消推送", "关闭推送"):
            from assistant.scheduler import remove_subscriber
            remove_subscriber(from_user)
            send_text_msg(from_user, "🔕 已关闭推送通知。发「订阅推送」可重新开启。")
            return Response(content="success", media_type="text/plain")

        urls = extract_urls(content)
        if urls:
            for url in urls:
                background_tasks.add_task(process_ingest, from_user, url)
            send_text_msg(from_user, f"📥 已收到 {len(urls)} 条链接，正在处理中...")
            # 自动订阅推送
            from assistant.scheduler import add_subscriber
            add_subscriber(from_user)
        else:
            # 走 AI 助手对话
            background_tasks.add_task(process_assistant_chat, from_user, content)
            send_text_msg(from_user, f"🤔 思考中...")

    # 处理链接消息 (微信转发的公众号文章等)
    elif msg_type == "link":
        url = msg.get("Url", "")
        title = msg.get("Title", "")
        if url:
            background_tasks.add_task(process_ingest, from_user, url)
            send_text_msg(from_user, f"📥 已收到「{title[:30]}」，正在入库...")

    # 处理图片消息 → Vision OCR
    elif msg_type == "image":
        pic_url = msg.get("PicUrl", "")
        if pic_url:
            background_tasks.add_task(process_file_ingest, from_user, pic_url, "image")
            send_text_msg(from_user, "🖼 已收到图片，正在 OCR 识别...")

    # 处理语音消息 → Whisper 转录
    elif msg_type == "voice":
        media_id = msg.get("MediaId", "")
        if media_id:
            background_tasks.add_task(process_media_ingest, from_user, media_id, "voice")
            send_text_msg(from_user, "🎤 已收到语音，正在转录...")

    # 处理文件消息 → 按类型分发
    elif msg_type == "file":
        media_id = msg.get("MediaId", "")
        filename = msg.get("FileName", "")
        if media_id:
            background_tasks.add_task(process_media_ingest, from_user, media_id, "file", filename)
            send_text_msg(from_user, f"📄 已收到文件「{filename[:30]}」，正在处理...")

    # 处理视频消息 → 暂仅提示
    elif msg_type in ("video", "shortvideo"):
        send_text_msg(from_user, "🎬 暂不支持视频入库，请发送视频的链接或截图。\n视频转录功能即将上线~")

    # 事件消息静默处理
    elif msg_type == "event":
        pass

    # 其他消息类型
    else:
        send_text_msg(from_user, "💡 支持：发链接入库、发文字搜索、发图片OCR、发语音转录、发PDF文件")

    return Response(content="success", media_type="text/plain")


# ===== 后台任务 =====

async def process_file_ingest(user_id: str, file_url: str, file_type: str):
    """后台异步处理文件入库（图片 OCR / 音频转录 / PDF 等）

    file_url 可以是 HTTP URL 或本地文件路径。
    """
    try:
        import tempfile, requests as req
        from ingestion.dispatcher import Dispatcher
        from pathlib import Path

        # 判断是本地路径还是 URL
        if file_url.startswith("/") or file_url.startswith("C:"):
            tmp_path = file_url  # 已经是本地路径
        else:
            # 下载文件
            resp = req.get(file_url, timeout=30)
            resp.raise_for_status()

            suffix_map = {
                "image": ".png", "voice": ".amr", "file": ".tmp",
                "pdf": ".pdf", "audio": ".mp3",
            }
            suffix = suffix_map.get(file_type, ".tmp")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name

        dispatcher = Dispatcher()
        raw = await dispatcher.dispatch(tmp_path)

        # 图片 OCR 补充（如果抓取器提取了图片列表）
        if raw.images:
            try:
                from ingestion.vision_ocr import VisionOCR
                ocr = VisionOCR()
                ocr_results = []
                for i, img_url in enumerate(raw.images[:5]):
                    try:
                        text = await ocr.ocr_image_url(img_url)
                        if text:
                            ocr_results.append((i + 1, text))
                    except Exception as e:
                        logger.debug(f"图片 OCR 失败: {e}")

                if ocr_results:
                    has_placeholders = "[IMG_" in raw.content
                    if has_placeholders:
                        import re as _re
                        for idx, text in ocr_results:
                            raw.content = raw.content.replace(f"[IMG_{idx}]", f"[图片{idx}内容: {text}]")
                        raw.content = _re.sub(r'\[IMG_\d+\]', '[图片: 无法识别]', raw.content)
                    else:
                        ocr_text = "\n\n".join(f"[图片{idx}内容: {text}]" for idx, text in ocr_results)
                        raw.content = raw.content + "\n\n---以下是图片内容---\n\n" + ocr_text
            except Exception as e:
                logger.warning(f"OCR 处理失败: {e}")

        # 入库管线
        from services.ingest_pipeline import ingest_raw
        result = await ingest_raw(raw, source_url=file_url)

        if not result.get("success"):
            raise Exception(result.get("error", "入库失败"))

        try:
            from wiki.compile_queue import enqueue_compile
            await enqueue_compile(result["file_path"], user_id=user_id)
        except Exception as e:
            logger.warning(f"Wiki 编译入队失败: {e}")

        notify_ingest_success(user_id, result.get("title", ""), result.get("tags", []), file_url)

        # 删除临时文件（跳过原始本地路径）
        if tmp_path != file_url:
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"文件入库失败: {file_type} - {e}")
        notify_ingest_fail(user_id, str(e), file_url)


async def process_media_ingest(user_id: str, media_id: str, media_type: str, filename: str = ""):
    """后台异步处理企微媒体文件（需要通过 media_id 下载）"""
    try:
        from wecom.sender import get_access_token
        import requests as req, tempfile
        from pathlib import Path

        # 通过企微 API 下载媒体文件
        token = get_access_token()
        url = f"https://qyapi.weixin.qq.com/cgi-bin/media/get?access_token={token}&media_id={media_id}"
        resp = req.get(url, timeout=30)

        if resp.status_code != 200:
            send_text_msg(user_id, "❌ 媒体文件下载失败")
            return

        # 根据类型确定后缀
        suffix_map = {"voice": ".amr", "file": ".tmp"}
        suffix = suffix_map.get(media_type, ".tmp")
        # 文件消息尝试从文件名获取后缀
        if media_type == "file" and filename:
            ext = Path(filename).suffix.lower()
            if ext:
                suffix = ext

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        # 直接传本地路径（修复 file:// Bug）
        await process_file_ingest(user_id, tmp_path, media_type)
        Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"媒体入库失败: {media_type}/{media_id} - {e}")
        send_text_msg(user_id, f"❌ 处理失败: {str(e)[:100]}")

async def process_ingest(user_id: str, url: str):
    """后台异步入库"""
    try:
        from services.ingest_pipeline import ingest_url
        result = await ingest_url(url)

        if result.get("duplicate"):
            send_text_msg(user_id, "⚠️ 该链接已入库，无需重复添加")
            return

        if not result.get("success"):
            notify_ingest_fail(user_id, result.get("error", "未知错误"), url)
            return

        notify_ingest_success(user_id, result.get("title", ""), result.get("tags", []), url)

    except Exception as e:
        logger.error(f"入库失败: {url} - {e}")
        notify_ingest_fail(user_id, str(e), url)


async def process_search(user_id: str, query: str):
    """后台异步搜索"""
    try:
        from main import get_searcher
        searcher = get_searcher()
        result = await searcher.search(query=query, top_k=5)

        answer = result.get("answer", "未找到相关内容")
        sources = result.get("sources", [])

        # 组装回复
        reply = f"📝 {answer}\n"
        if sources:
            reply += "\n📚 参考来源:\n"
            for i, src in enumerate(sources, 1):
                reply += f"  [{i}] {src.get('title', '未知')}\n"

        # 截断（企微消息有长度限制）
        if len(reply) > 2000:
            reply = reply[:1997] + "..."

        send_text_msg(user_id, reply)

    except Exception as e:
        logger.error(f"搜索失败: {query} - {e}")
        send_text_msg(user_id, f"❌ 搜索出错: {str(e)[:100]}")


async def process_assistant_chat(user_id: str, message: str):
    """后台 AI 助手对话 — 复用助手引擎，多轮对话"""
    try:
        from assistant.intent import detect_intent
        from assistant.chat_engine import chat_once

        # 使用企微 user_id 作为 session_id，保持多轮对话
        session_id = f"wecom_{user_id}"
        intent_result = detect_intent(message)

        # 构建上下文
        context = ""
        if intent_result["intent"] == "search":
            try:
                from main import get_searcher
                searcher = get_searcher()
                result = await searcher.search(query=message, top_k=3)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                source_list = "\n".join(f"  [{i+1}] {s.get('title', '未知')}" for i, s in enumerate(sources))
                context = f"知识库搜索结果:\n\n{answer}\n\n参考来源:\n{source_list}"
            except Exception as e:
                logger.warning(f"OCR 处理失败: {e}")
        elif intent_result["intent"] == "stats":
            from config import DATA_DIR, WIKI_DIR
            file_count = len(list(DATA_DIR.glob("*.md")))
            wiki_count = sum(1 for _ in WIKI_DIR.glob("**/*.md") if not _.name.startswith("_"))
            context = f"系统状态: {file_count} 篇文章, {wiki_count} 个 Wiki 页面"

        reply = await chat_once(session_id, message, context)
        if not reply:
            reply = "抱歉，暂时无法回复"

        if len(reply) > 2000:
            reply = reply[:1997] + "..."

        send_text_msg(user_id, reply)

    except Exception as e:
        logger.error(f"助手对话失败: {message[:50]} - {e}")
        # 降级到搜索
        await process_search(user_id, message)


# ===== 工具函数 =====

def extract_urls(text: str) -> list:
    """从文本中提取 URL"""
    url_pattern = r'https?://[^\s<>"\'，。！？、）\]】}]+'
    urls = re.findall(url_pattern, text)
    # 清理尾部标点
    cleaned = []
    for u in urls:
        u = u.rstrip(".,;:!?)]}>")
        if len(u) > 10:
            cleaned.append(u)
    return cleaned
