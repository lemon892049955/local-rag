"""ChromaDB 向量索引器

使用本地 SentenceTransformers 模型生成 Embedding，
存储在 ChromaDB 中实现语义检索。

v0.7: 支持 bge-small-zh-v1.5 + query instruction
"""

import logging
from pathlib import Path
from typing import List

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from chromadb.config import Settings

from config import VECTORDB_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_QUERY_INSTRUCTION, DATA_DIR
from retrieval.chunker import SemanticChunker, Chunk

logger = logging.getLogger(__name__)


class BGEEmbeddingFunction(EmbeddingFunction):
    """自定义 Embedding 函数 — 支持 bge 系列的 query instruction

    bge 模型在查询时需要加 instruction 前缀，文档侧不需要。
    ChromaDB 的 add() 走 document 模式（无前缀），query() 走 query 模式（加前缀）。
    """

    def __init__(self, model_name: str, query_instruction: str = ""):
        self._model = None
        self._model_name = model_name
        self._query_instruction = query_instruction
        self._is_query_mode = False

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载 Embedding 模型: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Embedding 模型加载完成: {self._model_name} ({self._model.get_sentence_embedding_dimension()}维)")
        return self._model

    def __call__(self, input: Documents) -> Embeddings:
        if self._is_query_mode and self._query_instruction:
            input = [self._query_instruction + text for text in input]
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()

    def set_query_mode(self, is_query: bool):
        """切换查询/文档模式"""
        self._is_query_mode = is_query


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
            self._embedding_fn = BGEEmbeddingFunction(
                model_name=EMBEDDING_MODEL,
                query_instruction=EMBEDDING_QUERY_INSTRUCTION,
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
        """为单个文件建立向量索引"""
        chunks = self.chunker.chunk_file(filepath)
        if not chunks:
            return 0

        # 文档模式（不加 query instruction）
        self.embedding_fn.set_query_mode(False)

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
        except Exception as e:
            logger.debug(f"删除旧 collection 失败（可能不存在）: {e}")

        # 重新索引所有文件
        total = 0
        for md_file in sorted(data_dir.glob("*.md")):
            count = self.index_file(md_file)
            total += count

        return total

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """语义搜索（query 模式：自动加 instruction 前缀）"""
        # 切换到查询模式
        self.embedding_fn.set_query_mode(True)
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"向量搜索失败: {e}")
            return []
        finally:
            self.embedding_fn.set_query_mode(False)

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
        except Exception as e:
            logger.debug(f"删除文件切片失败: {filepath} - {e}")

    def get_stats(self) -> dict:
        """获取索引统计信息"""
        try:
            count = self.collection.count()
        except Exception as e:
            logger.warning(f"获取索引统计失败: {e}")
            count = 0
        return {"total_chunks": count}
