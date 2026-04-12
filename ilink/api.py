"""iLink Bot 管理 API

提供扫码登录、状态查看等 HTTP 端点，方便在 Web UI 或终端中管理 Bot。
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ilink", tags=["iLink Bot"])


@router.get("/status")
async def bot_status():
    """Bot 运行状态"""
    from ilink.bot import _bot_running, _bot
    status = "running" if _bot_running else "stopped"
    return {
        "status": status,
        "bot_initialized": _bot is not None,
        "message": "iLink Bot 运行中" if _bot_running else "iLink Bot 未启动",
    }


@router.post("/login")
async def bot_login():
    """触发扫码登录

    返回二维码信息，用户需用微信扫码确认。
    登录成功后 Bot 自动开始监听消息。
    """
    try:
        from ilink.bot import get_bot, start_bot

        bot = get_bot()

        # 尝试登录
        result = await bot.login()
        logger.info(f"[iLink] 登录成功: account_id={result.account_id}")

        # 启动消息监听
        await start_bot()

        return {
            "success": True,
            "account_id": result.account_id,
            "user_id": result.user_id,
            "message": "登录成功，Bot 已开始监听消息",
        }
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="wechat-ilink-bot 未安装。请运行: pip install wechat-ilink-bot",
        )
    except Exception as e:
        logger.error(f"[iLink] 登录失败: {e}")
        raise HTTPException(status_code=500, detail=f"登录失败: {str(e)}")


@router.post("/start")
async def bot_start():
    """启动 Bot（使用缓存的登录态）"""
    try:
        from ilink.bot import start_bot
        await start_bot()
        return {"success": True, "message": "Bot 已启动"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动失败: {str(e)}")


@router.post("/stop")
async def bot_stop():
    """停止 Bot"""
    try:
        from ilink.bot import _bot, _bot_running
        import ilink.bot as bot_module
        if _bot:
            await _bot.stop()
            bot_module._bot_running = False
            bot_module._bot = None
        return {"success": True, "message": "Bot 已停止"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止失败: {str(e)}")
