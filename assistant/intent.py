"""意图识别模块

将用户输入分类为以下意图:
- search: 搜索知识库
- ingest: 入库新知识 (含URL)
- wiki: Wiki 相关操作
- stats: 查看系统状态
- chat: 闲聊/无明确意图
"""

import re
import logging

logger = logging.getLogger(__name__)

# URL 正则
URL_PATTERN = re.compile(r'https?://[^\s<>"\'，。！？、）\]】}]+')


def detect_intent(text: str) -> dict:
    """规则优先 + 关键词兜底的意图识别

    Returns:
        {"intent": str, "params": dict}
        intent: search | ingest | wiki | stats | chat
    """
    text = text.strip()
    if not text:
        return {"intent": "chat", "params": {}}

    # 1. 含 URL → 入库
    urls = URL_PATTERN.findall(text)
    if urls:
        return {"intent": "ingest", "params": {"urls": urls}}

    # 2. 显式入库指令
    ingest_triggers = ["入库", "收藏", "保存这个", "帮我存"]
    for t in ingest_triggers:
        if t in text:
            return {"intent": "ingest", "params": {"raw_text": text}}

    # 3. Wiki 操作
    wiki_triggers = ["wiki", "编译", "健康检查", "wiki页面", "知识图谱"]
    for t in wiki_triggers:
        if t in text.lower():
            return {"intent": "wiki", "params": {"raw_text": text}}

    # 4. 系统状态
    stats_triggers = ["状态", "统计", "stats", "多少篇", "多少个"]
    for t in stats_triggers:
        if t in text.lower():
            return {"intent": "stats", "params": {}}

    # 5. 默认 → 搜索 (知识库问答)
    # 以问号结尾、或较长文本、或含"什么""怎么""如何"等疑问词
    question_words = ["什么", "怎么", "如何", "为什么", "哪些", "哪个", "多少",
                      "是否", "能不能", "有没有", "对比", "区别", "推荐"]
    for w in question_words:
        if w in text:
            return {"intent": "search", "params": {"query": text}}

    if text.endswith("?") or text.endswith("？"):
        return {"intent": "search", "params": {"query": text}}

    # 短文本视为搜索，超长文本视为闲聊
    if len(text) < 100:
        return {"intent": "search", "params": {"query": text}}

    return {"intent": "chat", "params": {"raw_text": text}}
