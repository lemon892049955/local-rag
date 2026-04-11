"""BM25 关键词索引器

轻量级 BM25 实现，用于关键词搜索路，与向量搜索互补。

v2.1: 标题 3x 加权（title boost）
v0.7: jieba 分词 + 自定义词典 + 停用词过滤
v0.6.2: pickle 持久化
"""

import math
import pickle
import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict

import yaml

from retrieval.tokenizer import tokenize

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 关键词索引 — 支持 pickle 持久化"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = []          # [{id, text, title, metadata}, ...]
        self.doc_freqs = []     # [Counter, ...]  每个文档的词频
        self.idf = {}           # 全局 IDF
        self.avg_dl = 0         # 平均文档长度
        self.doc_lens = []      # 每个文档的 token 数
        self.n_docs = 0
        self.title_token_sets = []  # 每个文档 title 的 token 集合
        self._cache_path = None  # pickle 缓存路径

    def build_from_chunks(self, chunks: List[Dict]):
        """从切片列表构建索引

        Args:
            chunks: [{"chunk_id": "...", "text": "...", "title": "...", ...}, ...]
        """
        self.docs = chunks
        self.n_docs = len(chunks)
        self.doc_freqs = []
        self.doc_lens = []
        self.title_token_sets = []  # 每个文档 title 的 token 集合（用于 title boost）

        # 统计每个文档的词频
        df = Counter()  # 文档频率 (包含某词的文档数)
        for doc in chunks:
            tokens = tokenize(doc.get("text", ""))
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            self.doc_lens.append(len(tokens))
            # title tokens 缓存
            title_tokens = set(tokenize(doc.get("title", "")))
            self.title_token_sets.append(title_tokens)
            # 更新文档频率
            for word in freq:
                df[word] += 1

        self.avg_dl = sum(self.doc_lens) / max(self.n_docs, 1)

        # 计算 IDF
        self.idf = {}
        for word, freq in df.items():
            self.idf[word] = math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)

        # 持久化
        self._save_cache()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """BM25 检索

        Returns:
            [{"chunk_id": "...", "text": "...", "bm25_score": 1.23, ...}, ...]
        """
        if not self.docs:
            return []

        query_tokens = tokenize(query)
        scores = []

        for i, doc in enumerate(self.docs):
            score = 0.0
            dl = self.doc_lens[i]
            freq = self.doc_freqs[i]
            title_tokens = self.title_token_sets[i] if i < len(self.title_token_sets) else set()

            for token in query_tokens:
                idf = self.idf.get(token, 0)

                # 正文 BM25 打分
                if token in freq:
                    tf = freq[token]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
                    score += idf * numerator / denominator

                # 标题命中 boost：额外加 3x IDF（即使正文没命中也能加分）
                if token in title_tokens:
                    score += idf * 3.0

            scores.append((score, i))

        # 排序取 top_k
        scores.sort(key=lambda x: -x[0])
        results = []
        for score, idx in scores[:top_k]:
            if score > 0:
                hit = dict(self.docs[idx])
                hit["bm25_score"] = round(score, 4)
                results.append(hit)

        return results

    def build_from_directory(self, data_dir: Path, cache_name: str = "data_bm25"):
        """从 Markdown 文件目录构建索引（优先从缓存加载）"""
        from retrieval.chunker import SemanticChunker

        cache_path = data_dir / f".{cache_name}.pkl"
        self._cache_path = cache_path

        # 尝试从缓存加载
        if self._load_cache(data_dir):
            return self.n_docs

        # 缓存不存在或过期，全量重建
        chunker = SemanticChunker()
        all_chunks = []

        for md_file in sorted(data_dir.glob("*.md")):
            chunks = chunker.chunk_file(md_file)
            for c in chunks:
                all_chunks.append({
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "title": c.title,
                    "section_title": c.section_title,
                    "chunk_type": c.chunk_type,
                    "source_file": c.source_file,
                    "tags": ", ".join(c.metadata.get("tags", [])),
                    "source_url": c.metadata.get("source_url", ""),
                })

        self.build_from_chunks(all_chunks)
        return len(all_chunks)

    def _save_cache(self):
        """将索引序列化到磁盘"""
        if not self._cache_path:
            return
        try:
            cache_data = {
                "docs": self.docs,
                "doc_freqs": self.doc_freqs,
                "idf": self.idf,
                "avg_dl": self.avg_dl,
                "doc_lens": self.doc_lens,
                "n_docs": self.n_docs,
                "title_token_sets": self.title_token_sets,
            }
            with open(self._cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            logger.info(f"BM25 缓存已保存: {self._cache_path} ({self.n_docs} docs)")
        except Exception as e:
            logger.warning(f"BM25 缓存保存失败: {e}")

    def _load_cache(self, data_dir: Path) -> bool:
        """从磁盘加载缓存（如果缓存比最新文件更新则有效）"""
        if not self._cache_path or not self._cache_path.exists():
            return False

        try:
            # 检查缓存是否比最新的 .md 文件更新
            cache_mtime = self._cache_path.stat().st_mtime
            latest_md = 0
            for md_file in data_dir.glob("*.md"):
                mt = md_file.stat().st_mtime
                if mt > latest_md:
                    latest_md = mt

            if latest_md > cache_mtime:
                logger.info("BM25 缓存已过期，将重建")
                return False

            with open(self._cache_path, "rb") as f:
                cache_data = pickle.load(f)

            self.docs = cache_data["docs"]
            self.doc_freqs = cache_data["doc_freqs"]
            self.idf = cache_data["idf"]
            self.avg_dl = cache_data["avg_dl"]
            self.doc_lens = cache_data["doc_lens"]
            self.n_docs = cache_data["n_docs"]
            self.title_token_sets = cache_data.get("title_token_sets", [set() for _ in range(self.n_docs)])
            logger.info(f"BM25 缓存已加载: {self.n_docs} docs")
            return True

        except Exception as e:
            logger.warning(f"BM25 缓存加载失败: {e}")
            return False
