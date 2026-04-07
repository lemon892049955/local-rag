"""URL 工具 - 归一化、域名识别、去重检测"""

import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


def normalize_url(url: str) -> str:
    """URL 归一化 - 去除追踪参数，统一格式"""
    url = url.strip()

    # 处理小红书短链 (xhslink.com)
    # 短链需要先解析，这里仅做基础归一化
    parsed = urlparse(url)

    # 去除常见追踪参数
    tracking_params = {
        "utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term",
        "from", "isappinstalled", "scene", "subscene", "clicktime",
        "enterid", "sessionid", "nsukey", "wxshare_count",
        "share_source", "share_medium", "xhsshare",
    }

    query_params = parse_qs(parsed.query, keep_blank_values=False)
    cleaned_params = {
        k: v for k, v in query_params.items()
        if k.lower() not in tracking_params
    }

    cleaned_query = urlencode(cleaned_params, doseq=True)
    normalized = urlunparse((
        parsed.scheme or "https",
        parsed.netloc.lower(),
        parsed.path.rstrip("/"),
        parsed.params,
        cleaned_query,
        "",  # 去掉 fragment
    ))

    return normalized


def detect_source(url: str) -> str:
    """根据 URL 域名识别数据源类型

    Returns:
        "xiaohongshu" | "wechat" | "general"
    """
    domain = urlparse(url).netloc.lower()

    if any(d in domain for d in ["xiaohongshu.com", "xhslink.com", "xhs.cn"]):
        return "xiaohongshu"
    elif "mp.weixin.qq.com" in domain:
        return "wechat"
    else:
        return "general"


def extract_xhs_note_id(url: str):
    """从小红书链接中提取笔记 ID"""
    parsed = urlparse(url)
    # 常见格式: /explore/xxxx 或 /discovery/item/xxxx
    patterns = [
        r"/explore/([a-f0-9]+)",
        r"/discovery/item/([a-f0-9]+)",
        r"/item/([a-f0-9]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, parsed.path)
        if match:
            return match.group(1)
    return None


def check_duplicate(url: str, data_dir):
    """检查 URL 是否已入库

    遍历 data/ 目录下所有 .md 文件的 front-matter 中的 source_url 字段。

    Returns:
        已存在的文件路径 (str) 或 None
    """
    from pathlib import Path
    import yaml

    normalized = normalize_url(url)
    data_path = Path(data_dir)

    if not data_path.exists():
        return None

    for md_file in data_path.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            if not content.startswith("---"):
                continue
            # 提取 YAML front-matter
            parts = content.split("---", 2)
            if len(parts) < 3:
                continue
            meta = yaml.safe_load(parts[1])
            if meta and meta.get("source_url"):
                if normalize_url(meta["source_url"]) == normalized:
                    return str(md_file)
        except Exception:
            continue

    return None
