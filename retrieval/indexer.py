"""ChromaDB 向量索引器

使用本地 SentenceTransformers 模型生成 Embedding，
存储在 ChromaDB 中实现语义检索。
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings

from config import VECTORDB_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL, DATA_DIR
from retrieval.chunker import SemanticChunker, Chunk


class VectorIndexer:
    """向量索引管理器"""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(VECTORDB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._embedding_fn = None
        self.chunker = SemanticChunker()

    @property
    def embedding_fn(self):
        """延迟加载 Embedding 函数（首次调用时加载模型）"""
        if self._embedding_fn is None:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL,
            )
        return self._embedding_fn

    @property
    def collection(self):
        """获取或创建 ChromaDB collection"""
        return self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def index_file(self, filepath: Path) -> int:
        """为单个文件建立向量索引

        Args:
            filepath: Markdown 文件路径

        Returns:
            索引的切片数量
        """
        chunks = self.chunker.chunk_file(filepath)
        if not chunks:
            return 0

        # 先删除该文件的旧索引
        self._remove_file_chunks(filepath)

        # 批量添加
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source_file": c.source_file,
                    "title": c.title,
                    "section_title": c.section_title,
                    "chunk_type": c.chunk_type,
                    "tags": ", ".join(c.metadata.get("tags", [])),
                    "source_url": c.metadata.get("source_url", ""),
                }
                for c in chunks
            ],
        )

        return len(chunks)

    def reindex_all(self, data_dir=None) -> int:
        """重建全部索引（从 Markdown 文件恢复）

        Returns:
            总索引切片数
        """
        data_dir = data_dir or DATA_DIR

        # 清空 collection
        try:
            self.client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass

        # 重新索引所有文件
        total = 0
        for md_file in sorted(data_dir.glob("*.md")):
            count = self.index_file(md_file)
            total += count

        return total

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """语义搜索

        Args:
            query: 查询文本
            top_k: 返回前 K 个结果

        Returns:
            [{"text": "...", "title": "...", "distance": 0.12, ...}, ...]
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        if not results["ids"][0]:
            return []

        hits = []
        for i, chunk_id in enumerate(results["ids"][0]):
            hits.append({
                "chunk_id": chunk_id,
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
                **results["metadatas"][0][i],
            })

        return hits

    def _remove_file_chunks(self, filepath: Path):
        """删除某文件的所有切片索引"""
        filename = filepath.name
        try:
            existing = self.collection.get(
                where={"source_file": str(filepath)},
            )
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
        except Exception:
            pass

    def get_stats(self) -> dict:
        """获取索引统计信息"""
        try:
            count = self.collection.count()
        except Exception:
            count = 0
        return {"total_chunks": count}
