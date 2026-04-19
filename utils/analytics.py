"""数据埋点模块 — 记录用户行为和系统效果

使用 SQLite 本地存储，支持：
- 用户行为：搜索、点击、入库、停留
- 系统效果：搜索满意度、入库成功率
- 微信机器人：消息处理统计
"""

import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# 埋点数据库路径
ANALYTICS_DB = Path(__file__).parent.parent / "data" / "analytics.db"


def _ensure_db():
    """确保数据库和表存在"""
    ANALYTICS_DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(ANALYTICS_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                action TEXT,
                target TEXT,
                query TEXT,
                result TEXT,
                duration_ms INTEGER,
                extra TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # 创建索引加速查询
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON events(user_id)")
        conn.commit()


@contextmanager
def _get_conn():
    """获取数据库连接"""
    _ensure_db()
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def track(
    event_type: str,
    action: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    target: Optional[str] = None,
    query: Optional[str] = None,
    result: Optional[str] = None,
    duration_ms: Optional[int] = None,
    extra: Optional[dict] = None,
) -> int:
    """记录一个埋点事件

    Args:
        event_type: 事件类型 - search/ingest/click/view/bot/message
        action: 具体动作 - query/click/ingest/success/fail
        user_id: 用户标识（微信 user_id 或浏览器指纹）
        session_id: 会话 ID
        target: 操作对象（文章名、URL 等）
        query: 搜索词或消息内容
        result: 结果状态 - success/fail/timeout
        duration_ms: 耗时毫秒
        extra: 扩展字段（JSON）

    Returns:
        事件 ID
    """
    try:
        with _get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (timestamp, event_type, user_id, session_id, action, target, query, result, duration_ms, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    event_type,
                    user_id,
                    session_id,
                    action,
                    target,
                    query,
                    result,
                    duration_ms,
                    json.dumps(extra, ensure_ascii=False) if extra else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        logger.warning(f"埋点记录失败: {e}")
        return 0


def track_search(
    query: str,
    sources_count: int = 0,
    duration_ms: Optional[int] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> int:
    """记录搜索事件"""
    return track(
        event_type="search",
        action="query",
        query=query,
        result="success" if sources_count > 0 else "no_result",
        duration_ms=duration_ms,
        user_id=user_id,
        session_id=session_id,
        extra={"sources_count": sources_count},
    )


def track_click(
    target: str,
    target_type: str = "article",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> int:
    """记录点击事件"""
    return track(
        event_type="click",
        action="click",
        target=target,
        user_id=user_id,
        session_id=session_id,
        extra={"target_type": target_type},
    )


def track_ingest(
    url: str,
    title: Optional[str] = None,
    success: bool = True,
    duration_ms: Optional[int] = None,
    user_id: Optional[str] = None,
    source: str = "web",
) -> int:
    """记录入库事件"""
    return track(
        event_type="ingest",
        action="ingest",
        target=url,
        result="success" if success else "fail",
        duration_ms=duration_ms,
        user_id=user_id,
        extra={"title": title, "source": source},
    )


def track_bot_message(
    user_id: str,
    message_type: str,
    query: Optional[str] = None,
    result: str = "success",
    duration_ms: Optional[int] = None,
) -> int:
    """记录微信机器人消息"""
    return track(
        event_type="bot",
        action=message_type,  # text/ingest/search/stats
        user_id=user_id,
        query=query,
        result=result,
        duration_ms=duration_ms,
    )


def track_feedback(
    query: str,
    rating: str,
    user_id: Optional[str] = None,
) -> int:
    """记录搜索反馈"""
    return track(
        event_type="feedback",
        action="rate",
        query=query,
        result=rating,  # good/bad
        user_id=user_id,
    )


# ===== 统计查询 =====

def get_stats_summary(days: int = 7) -> dict:
    """获取统计概览"""
    with _get_conn() as conn:
        # 时间范围
        since = datetime.now().replace(hour=0, minute=0, second=0)
        from datetime import timedelta
        since = since - timedelta(days=days)

        # 总体统计
        total_events = conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE timestamp >= ?", (since.isoformat(),)
        ).fetchone()["cnt"]

        # 搜索统计
        search_stats = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN result = 'success' THEN 1 ELSE 0 END) as with_result,
                AVG(duration_ms) as avg_duration
            FROM events
            WHERE event_type = 'search' AND timestamp >= ?
            """,
            (since.isoformat(),),
        ).fetchone()

        # 入库统计
        ingest_stats = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN result = 'success' THEN 1 ELSE 0 END) as success
            FROM events
            WHERE event_type = 'ingest' AND timestamp >= ?
            """,
            (since.isoformat(),),
        ).fetchone()

        # 微信机器人统计
        bot_stats = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(DISTINCT user_id) as unique_users,
                SUM(CASE WHEN result = 'success' THEN 1 ELSE 0 END) as success
            FROM events
            WHERE event_type = 'bot' AND timestamp >= ?
            """,
            (since.isoformat(),),
        ).fetchone()

        # 反馈统计
        feedback_stats = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN result = 'good' THEN 1 ELSE 0 END) as good
            FROM events
            WHERE event_type = 'feedback' AND timestamp >= ?
            """,
            (since.isoformat(),),
        ).fetchone()

        # 活跃用户数
        active_users = conn.execute(
            """
            SELECT COUNT(DISTINCT user_id) as cnt
            FROM events
            WHERE timestamp >= ? AND user_id IS NOT NULL
            """,
            (since.isoformat(),),
        ).fetchone()["cnt"]

        return {
            "period_days": days,
            "total_events": total_events,
            "active_users": active_users,
            "search": {
                "total": search_stats["total"] or 0,
                "with_result": search_stats["with_result"] or 0,
                "no_result_rate": 1 - (search_stats["with_result"] or 0) / max(search_stats["total"] or 1, 1),
                "avg_duration_ms": round(search_stats["avg_duration"] or 0, 0),
            },
            "ingest": {
                "total": ingest_stats["total"] or 0,
                "success": ingest_stats["success"] or 0,
                "success_rate": (ingest_stats["success"] or 0) / max(ingest_stats["total"] or 1, 1),
            },
            "bot": {
                "total": bot_stats["total"] or 0,
                "unique_users": bot_stats["unique_users"] or 0,
                "success_rate": (bot_stats["success"] or 0) / max(bot_stats["total"] or 1, 1),
            },
            "feedback": {
                "total": feedback_stats["total"] or 0,
                "good_rate": (feedback_stats["good"] or 0) / max(feedback_stats["total"] or 1, 1),
            },
        }


def get_top_queries(days: int = 7, limit: int = 20) -> list:
    """获取热门搜索词"""
    with _get_conn() as conn:
        from datetime import timedelta
        since = datetime.now() - timedelta(days=days)

        rows = conn.execute(
            """
            SELECT query, COUNT(*) as cnt,
                   SUM(CASE WHEN result = 'success' THEN 1 ELSE 0 END) as success_cnt
            FROM events
            WHERE event_type = 'search' AND timestamp >= ? AND query IS NOT NULL
            GROUP BY query
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (since.isoformat(), limit),
        ).fetchall()

        return [
            {
                "query": r["query"],
                "count": r["cnt"],
                "success_rate": r["success_cnt"] / max(r["cnt"], 1),
            }
            for r in rows
        ]


def get_no_result_queries(days: int = 7, limit: int = 20) -> list:
    """获取无结果的搜索词（内容缺口）"""
    with _get_conn() as conn:
        from datetime import timedelta
        since = datetime.now() - timedelta(days=days)

        rows = conn.execute(
            """
            SELECT query, COUNT(*) as cnt
            FROM events
            WHERE event_type = 'search' AND result = 'no_result' AND timestamp >= ? AND query IS NOT NULL
            GROUP BY query
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (since.isoformat(), limit),
        ).fetchall()

        return [{"query": r["query"], "count": r["cnt"]} for r in rows]


def get_daily_trend(days: int = 7) -> list:
    """获取每日趋势"""
    with _get_conn() as conn:
        from datetime import timedelta
        since = datetime.now() - timedelta(days=days)

        rows = conn.execute(
            """
            SELECT
                DATE(timestamp) as date,
                COUNT(*) as total,
                SUM(CASE WHEN event_type = 'search' THEN 1 ELSE 0 END) as search_cnt,
                SUM(CASE WHEN event_type = 'ingest' THEN 1 ELSE 0 END) as ingest_cnt,
                SUM(CASE WHEN event_type = 'bot' THEN 1 ELSE 0 END) as bot_cnt,
                COUNT(DISTINCT user_id) as active_users
            FROM events
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
            """,
            (since.isoformat(),),
        ).fetchall()

        return [
            {
                "date": r["date"],
                "total": r["total"],
                "search": r["search_cnt"],
                "ingest": r["ingest_cnt"],
                "bot": r["bot_cnt"],
                "active_users": r["active_users"],
            }
            for r in rows
        ]


def get_recent_events(limit: int = 50, event_type: Optional[str] = None) -> list:
    """获取最近事件列表"""
    with _get_conn() as conn:
        if event_type:
            rows = conn.execute(
                """
                SELECT * FROM events
                WHERE event_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (event_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM events
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [dict(r) for r in rows]


def export_events(days: int = 30) -> list:
    """导出事件数据（用于分析）"""
    with _get_conn() as conn:
        from datetime import timedelta
        since = datetime.now() - timedelta(days=days)

        rows = conn.execute(
            "SELECT * FROM events WHERE timestamp >= ? ORDER BY timestamp",
            (since.isoformat(),),
        ).fetchall()

        return [dict(r) for r in rows]
