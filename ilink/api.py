"""iLink Bot 管理 API

提供扫码登录、状态查看等 HTTP 端点，方便在 Web UI 或终端中管理 Bot。
"""

import logging

from fastapi import APIRouter, HTTPException

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
        "message": "iLink Bot 运行中，可接收微信消息" if _bot_running else "iLink Bot 未启动，请先登录",
    }


@router.post("/login")
async def bot_login():
    """触发扫码登录

    会在服务器终端打印二维码，需要用微信扫码确认。
    登录成功后 Bot 自动开始监听消息。
    
    注意：需要在服务器终端查看二维码，或通过 docker logs 查看。
    """
    try:
        from ilink.bot import login_bot
        result = await login_bot()

        if result.get("success"):
            return {
                "success": True,
                "account_id": result.get("account_id"),
                "user_id": result.get("user_id"),
                "message": "登录成功，Bot 已开始监听消息",
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"登录失败: {result.get('error', '未知错误')}",
            )
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="wechat-ilink-bot 未安装。请运行: pip install wechat-ilink-bot",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[iLink] 登录失败: {e}")
        raise HTTPException(status_code=500, detail=f"登录失败: {str(e)}")


@router.post("/start")
async def bot_start():
    """启动 Bot（使用缓存的登录态）"""
    try:
        from ilink.bot import start_bot
        ok = await start_bot()
        if ok:
            return {"success": True, "message": "Bot 已启动"}
        else:
            raise HTTPException(
                status_code=500,
                detail="启动失败，可能需要先登录。请调用 POST /ilink/login",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动失败: {str(e)}")


@router.post("/stop")
async def bot_stop():
    """停止 Bot"""
    try:
        import ilink.bot as bot_module
        if bot_module._bot:
            await bot_module._bot.stop()
            bot_module._bot_running = False
            bot_module._bot = None
            bot_module._handlers_registered = False
        return {"success": True, "message": "Bot 已停止"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止失败: {str(e)}")
