#!/usr/bin/env python3
"""微信 iLink Bot 登录工具

Usage:
    python ilink/login.py
"""

import asyncio
import json
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent


async def login():
    """执行登录并保存 token"""
    from wechat_bot import Bot

    print("正在初始化...")
    bot = Bot(use_current_user=False)

    print("请扫描二维码登录微信...")
    result = await bot.login()

    print(f"\n✅ 登录成功!")
    print(f"   account_id: {result.account_id}")
    print(f"   user_id: {result.user_id}")

    # 保存到项目目录
    token_file = BASE_DIR / ".ilink_token.json"
    token_data = {
        "bot_token": result.token,
        "account_id": result.account_id,
        "user_id": result.user_id,
        "base_url": result.base_url,
    }
    token_file.write_text(json.dumps(token_data, ensure_ascii=False, indent=2))
    print(f"\n💾 Token 已保存到: {token_file}")

    return result


if __name__ == "__main__":
    asyncio.run(login())
