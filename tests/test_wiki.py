"""Wiki 编译测试"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path


class TestWikiCompiler:
    """Wiki 编译器测试"""

    def test_compile_queue_singleton(self):
        """测试编译队列单例"""
        from wiki.compile_queue import get_queue

        q1 = get_queue()
        q2 = get_queue()

        assert q1 is q2

    @pytest.mark.skip(reason="异步入队测试在 CI 环境中不稳定")
    def test_enqueue_compile(self):
        """测试编译入队"""
        from wiki.compile_queue import enqueue_compile, get_queue

        # 清空队列
        q = get_queue()
        while not q.empty():
            q.get()

        import asyncio
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Test\n\nContent")
            temp_path = Path(f.name)

        try:
            asyncio.run(enqueue_compile(temp_path))
            assert q.qsize() >= 1
        finally:
            temp_path.unlink()


class TestPageStore:
    """Wiki 页面存储测试"""

    def test_list_wiki_pages_empty(self):
        """测试空 Wiki 列表"""
        from wiki.page_store import list_wiki_pages
        from config import WIKI_DIR

        # 如果 Wiki 目录存在但为空
        pages = list_wiki_pages()

        assert isinstance(pages, list)
