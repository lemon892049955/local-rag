"""待审核池存储 — 管理待入库内容

存储结构: data/pending_items.json
{
  "items": [...],
  "last_fetch": "2026-04-19T08:00:00",
  "stats": {"total": 100, "pending": 80, "ingested": 15, "ignored": 5}
}
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

PENDING_FILE = Path(__file__).parent.parent / "data" / "pending_items.json"
MAX_ITEMS = 500  # 最大保留数量


def _load_data() -> dict:
    """加载待审核数据"""
    if not PENDING_FILE.exists():
        return {"items": [], "last_fetch": None, "stats": {"total": 0, "pending": 0, "ingested": 0, "ignored": 0}}

    try:
        with open(PENDING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载待审核数据失败: {e}")
        return {"items": [], "last_fetch": None, "stats": {"total": 0, "pending": 0, "ingested": 0, "ignored": 0}}


def _save_data(data: dict):
    """保存待审核数据"""
    PENDING_FILE.parent.mkdir(exist_ok=True)
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_pending_items(status: str = "pending", page: int = 1, page_size: int = 20,
                      source: Optional[str] = None) -> dict:
    """获取待审核列表

    Args:
        status: pending/ingested/ignored/all
        page: 页码
        page_size: 每页数量
        source: 按来源筛选

    Returns:
        {"items": [...], "total": 100, "page": 1, "page_size": 20}
    """
    data = _load_data()
    items = data.get("items", [])

    # 动态计算质量评分（如果缺失）
    for item in items:
        if "quality_score" not in item:
            item["quality_score"] = _calc_quality_score(item)

    # 筛选
    if status != "all":
        items = [i for i in items if i.get("status") == status]

    if source:
        items = [i for i in items if i.get("source") == source]

    # 排序（质量分为主，热度分为辅）
    def sort_key(x):
        q = x.get("quality_score", 0)
        h = x.get("score", 0)
        # 热度分归一化到 0-100
        h_norm = min(h / 1000, 100) if h else 0
        return q * 0.7 + h_norm * 0.3

    items.sort(key=sort_key, reverse=True)

    # 分页
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = items[start:end]

    return {
        "items": page_items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "sources": list(set(i.get("source") for i in data.get("items", []) if i.get("source"))),
        "categories": list(set(i.get("category", "其他") for i in items if i.get("category"))),
        "last_fetch": data.get("last_fetch"),
    }


def _calc_quality_score(item: dict) -> int:
    """计算内容质量评分"""
    # 分类关键词
    category_keywords = {
        "AI技术": [
            "AI", "人工智能", "GPT", "ChatGPT", "Claude", "DeepSeek", "OpenAI", "Anthropic",
            "大模型", "LLM", "AGI", "机器学习", "深度学习", "神经网络", "Transformer",
            "Agent", "智能体", "RAG", "MCP", "向量数据库", "Embedding", "知识库",
        ],
        "产品设计": ["产品经理", "产品设计", "交互设计", "用户体验", "UX", "UI", "原型"],
        "商业趋势": ["创业", "投资", "融资", "商业模式", "市场分析", "行业趋势", "商业化"],
        "职业发展": ["面试", "简历", "职场", "职业规划", "跳槽", "薪资", "晋升"],
        "开发工具": ["代码", "编程", "开发", "框架", "库", "工具", "Python", "JavaScript"],
        "互联网产品": ["小红书", "抖音", "微信", "B站", "知乎", "社交", "内容", "平台"],
    }
    quality_sources = {
        "机器之心": 10, "量子位": 9, "Hacker News": 8, "GitHub Trending": 8,
        "掘金": 7, "少数派": 7, "36氪": 6,
    }

    title = item.get("title", "").lower()
    summary = (item.get("summary", "") or "").lower()
    text = f"{title} {summary}"
    source = item.get("source", "")

    score = 0
    best_cat = "其他"
    best_matches = 0

    for cat, keywords in category_keywords.items():
        matches = sum(1 for kw in keywords if kw.lower() in text)
        if matches > best_matches:
            best_matches = matches
            best_cat = cat
        score += min(matches * 5, 40)

    item["category"] = best_cat
    score += quality_sources.get(source, 5)
    return score


def add_items(new_items: list[dict]) -> int:
    """添加新内容到待审核池（去重）

    Returns:
        实际新增数量
    """
    data = _load_data()
    existing_ids = {i.get("id") for i in data.get("items", [])}

    added = 0
    for item in new_items:
        if item.get("id") not in existing_ids:
            data["items"].append(item)
            added += 1

    # 更新统计
    data["stats"] = _calc_stats(data["items"])
    data["last_fetch"] = datetime.now().isoformat()

    # 清理超量数据（保留 pending 和最近的 ingested/ignored）
    if len(data["items"]) > MAX_ITEMS:
        pending = [i for i in data["items"] if i.get("status") == "pending"]
        others = sorted(
            [i for i in data["items"] if i.get("status") != "pending"],
            key=lambda x: x.get("fetched_at", ""),
            reverse=True
        )[:100]
        data["items"] = pending + others
        logger.info(f"清理待审核池，保留 {len(data['items'])} 条")

    _save_data(data)
    logger.info(f"添加 {added} 条新内容到待审核池")
    return added


def update_status(item_id: str, status: str) -> bool:
    """更新内容状态

    Args:
        item_id: 内容 ID
        status: ingested/ignored/pending

    Returns:
        是否成功
    """
    data = _load_data()

    for item in data["items"]:
        if item.get("id") == item_id:
            item["status"] = status
            if status == "ingested":
                item["ingested_at"] = datetime.now().isoformat()
            elif status == "ignored":
                item["ignored_at"] = datetime.now().isoformat()
            break
    else:
        return False

    data["stats"] = _calc_stats(data["items"])
    _save_data(data)
    return True


def batch_update_status(item_ids: list[str], status: str) -> int:
    """批量更新状态

    Returns:
        成功更新数量
    """
    data = _load_data()
    updated = 0

    for item in data["items"]:
        if item.get("id") in item_ids:
            item["status"] = status
            if status == "ingested":
                item["ingested_at"] = datetime.now().isoformat()
            elif status == "ignored":
                item["ignored_at"] = datetime.now().isoformat()
            updated += 1

    data["stats"] = _calc_stats(data["items"])
    _save_data(data)
    return updated


def get_item_by_id(item_id: str) -> Optional[dict]:
    """获取单条内容"""
    data = _load_data()
    for item in data["items"]:
        if item.get("id") == item_id:
            return item
    return None


def get_stats() -> dict:
    """获取统计信息"""
    data = _load_data()
    return data.get("stats", {"total": 0, "pending": 0, "ingested": 0, "ignored": 0})


def _calc_stats(items: list[dict]) -> dict:
    """计算统计信息"""
    stats = {"total": len(items), "pending": 0, "ingested": 0, "ignored": 0}
    for item in items:
        status = item.get("status", "pending")
        if status in stats:
            stats[status] += 1
    return stats


def clear_old_items(days: int = 30) -> int:
    """清理过期内容

    Args:
        days: 保留天数

    Returns:
        清理数量
    """
    data = _load_data()
    cutoff = datetime.now().timestamp() - days * 86400

    original_len = len(data["items"])
    data["items"] = [
        i for i in data["items"]
        if i.get("status") == "pending" or
        datetime.fromisoformat(i.get("fetched_at", "2000-01-01")).timestamp() > cutoff
    ]

    removed = original_len - len(data["items"])
    if removed > 0:
        data["stats"] = _calc_stats(data["items"])
        _save_data(data)
        logger.info(f"清理 {removed} 条过期内容")

    return removed
