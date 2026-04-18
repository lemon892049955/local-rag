"""入库管线测试"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path


class TestIngestPipeline:
    """入库管线测试"""

    @pytest.mark.asyncio
    async def test_ingest_url_duplicate_skip(self):
        """测试重复 URL 跳过"""
        from services.ingest_pipeline import ingest_url

        with patch("services.ingest_pipeline.check_duplicate") as mock_dup:
            mock_dup.return_value = "existing_file.md"

            result = await ingest_url("https://example.com/test", skip_duplicate=True)

            assert result["success"] is True
            assert result["duplicate"] is True

    @pytest.mark.asyncio
    async def test_ingest_url_force_mode_deletes_old_file(self):
        """测试强制重入库模式删除旧文件"""
        import tempfile
        from services.ingest_pipeline import ingest_url

        # 创建临时文件模拟旧文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Old content")
            old_file = Path(f.name)

        try:
            # 模拟 check_duplicate 返回旧文件路径
            with patch("services.ingest_pipeline.check_duplicate") as mock_dup:
                mock_dup.return_value = str(old_file)

                # 模拟后续流程
                with patch("services.ingest_pipeline._get_router") as mock_router, \
                     patch("services.ingest_pipeline._get_cleaner") as mock_cleaner, \
                     patch("services.ingest_pipeline._get_engine") as mock_engine, \
                     patch("services.ingest_pipeline._get_indexer") as mock_indexer:

                    mock_router.return_value.fetch = AsyncMock(return_value=Mock(
                        content="test", title="Test", images=[], source_platform="web", author=None, original_tags=[]
                    ))
                    mock_cleaner.return_value.clean = AsyncMock(return_value=Mock(
                        title="Test Title", content="Cleaned", summary="Summary", tags=["test"]
                    ))
                    mock_engine.return_value.save = Mock(return_value=Path("data/test.md"))
                    mock_indexer.return_value.index_file = Mock(return_value=5)

                    result = await ingest_url("https://example.com/test", force=True)

                    # 验证旧文件被删除
                    assert not old_file.exists()
                    assert result["success"] is True
        finally:
            if old_file.exists():
                old_file.unlink()


class TestURLUtils:
    """URL 工具测试"""

    def test_normalize_url(self):
        """测试 URL 归一化"""
        from utils.url_utils import normalize_url

        url1 = normalize_url("https://example.com/path?utm_source=test")
        url2 = normalize_url("https://example.com/path")

        # 去除追踪参数后应该一致
        assert "utm_source" not in url1

    def test_check_duplicate_not_found(self):
        """测试去重检查未找到"""
        from utils.url_utils import check_duplicate
        from config import DATA_DIR

        result = check_duplicate("https://nonexistent-unique-url-12345.com", DATA_DIR)

        assert result is None
