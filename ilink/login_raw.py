#!/usr/bin/env python3
"""iLink Bot 登录 — 输出二维码 URL 供远程扫码

把二维码原始链接打印出来，可以用任何在线工具生成二维码扫描。
同时也输出一个在线二维码图片链接。
"""

import asyncio
import sys
import json
import time
import requests as req


ILINK_BASE = "https://ilinkai.weixin.qq.com"


def make_headers():
    import random, base64
    uin = random.randint(1, 2**32 - 1)
    uin_b64 = base64.b64encode(str(uin).encode()).decode()
    return {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/json",
        "X-WECHAT-UIN": uin_b64,
    }


def get_qrcode():
    """获取登录二维码"""
    url = f"{ILINK_BASE}/ilink/bot/get_bot_qrcode?bot_type=3"
    resp = req.get(url, headers=make_headers(), timeout=15)
    data = resp.json()
    return data


def poll_status(qrcode_key: str, max_wait=180):
    """轮询扫码状态"""
    url = f"{ILINK_BASE}/ilink/bot/get_qrcode_status?qrcode={qrcode_key}"
    start = time.time()
    while time.time() - start < max_wait:
        time.sleep(3)
        try:
            resp = req.get(url, headers=make_headers(), timeout=30)
            data = resp.json()
        except Exception as e:
            print(f"  轮询出错: {e}, 重试...")
            continue
        if "bot_token" in data:
            return data
        status = data.get("status", "")
        if status == "expired":
            return None
        elapsed = int(time.time() - start)
        print(f"  等待扫码... ({elapsed}s, status={status})")
    return None


def main():
    print("=" * 50)
    print("  BuddyKnow - iLink Bot 登录")
    print("=" * 50)
    print()

    # 1. 获取二维码
    print("正在获取二维码...")
    qr_data = get_qrcode()
    print(f"API 返回: {json.dumps(qr_data, indent=2, ensure_ascii=False)[:500]}")
    print()

    # 尝试提取二维码内容
    qr_content = qr_data.get("qrcode_img_content") or qr_data.get("qrcode_content") or ""
    qr_key = qr_data.get("qrcode") or qr_data.get("qrcode_key") or ""

    if qr_content:
        print(f"二维码内容(URL): {qr_content}")
        # 生成在线二维码图片
        from urllib.parse import quote
        qr_img_url = f"https://api.qrserver.com/v1/create-qr-code/?size=300x300&data={quote(qr_content)}"
        print(f"\n在线二维码图片: {qr_img_url}")
        print("\n请在浏览器中打开上面的图片链接，然后用微信扫码！")
    else:
        print("未获取到二维码内容，完整返回:")
        print(json.dumps(qr_data, indent=2, ensure_ascii=False))

    if not qr_key:
        print("未获取到 qrcode key，无法轮询")
        return

    # 2. 轮询状态
    print("\n等待扫码确认...")
    result = poll_status(qr_key)
    if result and "bot_token" in result:
        token = result["bot_token"]
        baseurl = result.get("baseurl", ILINK_BASE)
        print(f"\n✅ 登录成功!")
        print(f"   bot_token: {token[:20]}...")
        print(f"   baseurl: {baseurl}")

        # 保存 token 供 Bot 使用
        token_file = "/app/.ilink_token.json"
        with open(token_file, "w") as f:
            json.dump({"bot_token": token, "baseurl": baseurl}, f)
        print(f"\n   Token 已保存到 {token_file}")
        print("   重启服务后可自动读取。")
    else:
        print("\n❌ 二维码已过期，请重新运行此脚本。")


if __name__ == "__main__":
    main()
