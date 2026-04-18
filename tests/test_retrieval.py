"""检索模块测试"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestBM25Index:
    """BM25 索引测试"""

    def test_build_from_chunks(self):
        """测试从切片构建索引"""
        from retrieval.bm25 import BM25Index

        bm25 = BM25Index()
        chunks = [
            {"chunk_id": "1", "text": "人工智能是未来", "title": "AI 概述"},
            {"chunk_id": "2", "text": "机器学习是 AI 的子领域", "title": "ML 简介"},
        ]

        bm25.build_from_chunks(chunks)

        assert bm25.n_docs == 2
        assert len(bm25.docs) == 2

    def test_search_returns_results(self):
        """测试搜索返回结果"""
        from retrieval.bm25 import BM25Index

        bm25 = BM25Index()
        chunks = [
            {"chunk_id": "1", "text": "人工智能技术发展迅速", "title": "AI 概述"},
            {"chunk_id": "2", "text": "机器学习算法优化", "title": "ML 简介"},
        ]
        bm25.build_from_chunks(chunks)

        results = bm25.search("人工智能", top_k=2)

        assert len(results) >= 1
        assert results[0]["chunk_id"] == "1"

    def test_search_empty_query(self):
        """测试空查询"""
        from retrieval.bm25 import BM25Index

        bm25 = BM25Index()
        bm25.build_from_chunks([{"chunk_id": "1", "text": "测试内容", "title": "测试"}])

        results = bm25.search("", top_k=5)

        assert results == []


class TestTokenizer:
    """分词器测试"""

    def test_tokenize_chinese(self):
        """测试中文分词"""
        from retrieval.tokenizer import tokenize

        tokens = tokenize("人工智能技术")

        assert len(tokens) > 0
        assert "人工智能" in tokens or "人工" in tokens

    def test_tokenize_mixed(self):
        """测试中英混合"""
        from retrieval.tokenizer import tokenize

        tokens = tokenize("AI 人工智能")

        assert len(tokens) > 0


class TestChunker:
    """切片器测试"""

    def test_chunk_short_text(self):
        """测试短文切片"""
        from retrieval.chunker import SemanticChunker

        chunker = SemanticChunker()

        # 创建临时测试文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# 测试标题\n\n这是测试内容。")
            temp_path = Path(f.name)

        try:
            chunks = chunker.chunk_file(temp_path)
            assert len(chunks) >= 1
            # title 来自文件名或第一个标题
            assert chunks[0].title is not None
        finally:
            temp_path.unlink()
