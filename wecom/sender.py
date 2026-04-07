"""企业微信主动推送消息

通过企微 API 向用户发送入库结果通知。
"""

import time
import requests
from config import get_wecom_config


_access_token_cache = {"token": "", "expires_at": 0}


def get_access_token() -> str:
    """获取企微 access_token (自动缓存)"""
    now = int(time.time())
    if _access_token_cache["token"] and _access_token_cache["expires_at"] > now + 60:
        return _access_token_cache["token"]

    cfg = get_wecom_config()
    url = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
    resp = requests.get(url, params={
        "corpid": cfg["corp_id"],
        "corpsecret": cfg["secret"],
    }, timeout=10)
    data = resp.json()

    if data.get("errcode", 0) != 0:
        raise Exception(f"获取 access_token 失败: {data}")

    _access_token_cache["token"] = data["access_token"]
    _access_token_cache["expires_at"] = now + data.get("expires_in", 7200)
    return _access_token_cache["token"]


def send_text_msg(user_id: str, content: str, agent_id: int = None):
    """发送文本消息给用户"""
    cfg = get_wecom_config()
    token = get_access_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"

    payload = {
        "touser": user_id,
        "msgtype": "text",
        "agentid": agent_id or cfg["agent_id"],
        "text": {"content": content},
    }

    resp = requests.post(url, json=payload, timeout=10)
    return resp.json()


def send_text_card(user_id: str, title: str, description: str,
                   url: str = "", btn_txt: str = "详情", agent_id: int = None):
    """发送文本卡片消息"""
    cfg = get_wecom_config()
    token = get_access_token()
    api_url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"

    payload = {
        "touser": user_id,
        "msgtype": "textcard",
        "agentid": agent_id or cfg["agent_id"],
        "textcard": {
            "title": title,
            "description": description,
            "url": url or "https://work.weixin.qq.com",
            "btntxt": btn_txt,
        },
    }

    resp = requests.post(api_url, json=payload, timeout=10)
    return resp.json()


def notify_ingest_success(user_id: str, title: str, tags: list, source_url: str = ""):
    """入库成功通知"""
    tags_str = "、".join(tags[:5]) if tags else "待分类"
    desc = (
        f"<div class=\"gray\">知识库入库通知</div>"
        f"<div class=\"normal\">标题：{title}</div>"
        f"<div class=\"normal\">标签：{tags_str}</div>"
        f"<div class=\"highlight\">已入库并建立索引</div>"
    )
    return send_text_card(
        user_id=user_id,
        title="✅ 入库成功",
        description=desc,
        url=source_url or "https://work.weixin.qq.com",
        btn_txt="查看原文",
    )


def notify_ingest_fail(user_id: str, reason: str, url: str = ""):
    """入库失败通知"""
    desc = (
        f"<div class=\"gray\">知识库入库通知</div>"
        f"<div class=\"normal\">链接：{url[:50]}...</div>"
        f"<div class=\"highlight\">失败原因：{reason[:100]}</div>"
    )
    return send_text_card(
        user_id=user_id,
        title="❌ 入库失败",
        description=desc,
        url=url or "https://work.weixin.qq.com",
        btn_txt="重试",
    )
