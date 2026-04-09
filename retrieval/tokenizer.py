"""统一分词模块 — jieba 精确模式 + 自定义词典 + 停用词

v0.7: 替换 bm25.py 中的字+bigram 分词，提升中文关键词召回精度。
"""

import logging
import re
from pathlib import Path
from typing import List, Set

import jieba

from config import BASE_DIR

logger = logging.getLogger(__name__)

# ===== 停用词表 =====
_stopwords: Set[str] = set()
_initialized = False


def _ensure_init():
    """延迟初始化：加载自定义词典和停用词（只执行一次）"""
    global _initialized
    if _initialized:
        return
    _initialized = True

    # 加载自定义词典
    dict_path = BASE_DIR / "data" / "user_dict.txt"
    if dict_path.exists():
        jieba.load_userdict(str(dict_path))
        logger.info(f"jieba 自定义词典已加载: {dict_path}")
    else:
        logger.info("未找到自定义词典，使用 jieba 默认词典")

    # 加载停用词
    stopwords_path = BASE_DIR / "data" / "stopwords.txt"
    if stopwords_path.exists():
        with open(stopwords_path, encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    _stopwords.add(word)
        logger.info(f"停用词表已加载: {len(_stopwords)} 个")
    else:
        # 内置基础停用词
        _stopwords.update({
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
            "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
            "着", "没有", "看", "好", "自己", "这", "他", "她", "它", "们",
            "那", "些", "什么", "这个", "那个", "这些", "那些", "之", "与",
            "及", "等", "或", "但", "而", "所", "以", "其", "从", "被", "把",
            "对", "中", "为", "能", "可以", "已", "已经", "还", "又", "让",
            "将", "时", "地", "得", "吗", "吧", "呢", "啊", "哦", "嗯",
            "如果", "因为", "所以", "虽然", "但是", "然而", "而且", "并且",
            "或者", "不过", "只是", "这样", "那样", "怎么", "怎样", "如何",
            "为什么", "多少", "哪里", "什么样", "比较", "更加", "非常", "特别",
            "可能", "应该", "需要", "通过", "进行", "使用", "包括", "以及",
            "关于", "根据", "目前", "其中", "主要", "同时", "已经", "之后",
        })
        logger.info(f"使用内置停用词: {len(_stopwords)} 个")


def tokenize(text: str) -> List[str]:
    """中英文混合分词 — jieba 精确模式 + 停用词过滤

    策略：
    1. jieba 精确模式分词（中文）
    2. 额外提取英文单词（jieba 对英文处理不够好）
    3. 过滤停用词 + 单字 + 标点
    4. 全部小写
    """
    _ensure_init()

    # jieba 精确模式分词
    words = jieba.lcut(text)

    # 过滤：停用词 + 单字 + 纯标点/数字
    filtered = []
    for w in words:
        w_stripped = w.strip()
        if not w_stripped:
            continue
        if w_stripped in _stopwords:
            continue
        if len(w_stripped) < 2 and not re.match(r'[a-zA-Z]', w_stripped):
            continue
        # 纯标点/空白跳过
        if re.match(r'^[\s\W]+$', w_stripped) and not re.match(r'[a-zA-Z0-9]', w_stripped):
            continue
        filtered.append(w_stripped.lower())

    # 额外提取英文单词（jieba 可能把 "Vibe Coding" 拆成 "Vibe" "Coding" 或不拆）
    en_words = re.findall(r'[a-zA-Z][a-zA-Z0-9_]{1,}', text)
    en_words = [w.lower() for w in en_words if w.lower() not in _stopwords]

    # 合并去重（保持顺序）
    seen = set()
    result = []
    for w in filtered + en_words:
        if w not in seen:
            seen.add(w)
            result.append(w)

    return result
