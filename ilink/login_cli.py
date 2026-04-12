#!/usr/bin/env python3
"""iLink Bot 独立登录脚本

在 Docker 容器内运行此脚本进行扫码登录。
登录成功后 token 会被缓存，重启服务后自动使用。

用法:
    docker exec -it local-rag python /app/ilink/login_cli.py

或者后台运行（通过 docker logs 查看二维码）:
    docker exec local-rag python /app/ilink/login_cli.py
"""

import asyncio
import sys


async def main():
    try:
        from wechat_bot import Bot

        print("=" * 50)
        print("  BuddyKnow - iLink Bot 登录")
        print("=" * 50)
        print()
        print("正在获取登录二维码...")
        print("请用微信扫描下方二维码：")
        print()

        bot = Bot(use_current_user=False)
        result = await bot.login()

        print()
        print("=" * 50)
        print(f"✅ 登录成功!")
        print(f"   Account ID: {result.account_id}")
        print(f"   User ID:    {result.user_id}")
        print("=" * 50)
        print()
        print("Token 已缓存，重启服务后将自动连接。")
        print("现在可以重启容器: docker restart local-rag")

        await bot.stop()

    except ImportError:
        print("❌ wechat-ilink-bot 未安装")
        print("   pip install wechat-ilink-bot")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 登录失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
