"""发现模块 — 热榜监控 + 待审核池"""

from .crawler import fetch_all_sources, fetch_hn_ai, fetch_github_ai, fetch_zhihu_hot, fetch_v2ex_hot
from .store import (
    get_pending_items,
    add_items,
    update_status,
    batch_update_status,
    get_item_by_id,
    get_stats,
    clear_old_items,
)

__all__ = [
    "fetch_all_sources",
    "fetch_hn_ai",
    "fetch_github_ai",
    "fetch_zhihu_hot",
    "fetch_v2ex_hot",
    "get_pending_items",
    "add_items",
    "update_status",
    "batch_update_status",
    "get_item_by_id",
    "get_stats",
    "clear_old_items",
]
