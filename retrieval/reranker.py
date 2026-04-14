"""Cross-Encoder Re-ranking 模块

v0.7: RRF 粗排后，用 Cross-Encoder 对 (query, chunk) pair 做精排。
延迟加载模型（首次搜索时），常驻内存。

模型: BAAI/bge-reranker-v2-m3 (~570MB, 多语言)
"""

import logging
import os
from typing import List, Dict

logger = logging.getLogger(__name__)

# 默认 Reranker 模型
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
# 可能的本地缓存路径列表
RERANKER_LOCAL_PATHS = [
    os.path.join(os.environ.get("SENTENCE_TRANSFORMERS_HOME", "vectordb/models"), "BAAI/bge-reranker-v2-m3"),
    # huggingface_hub snapshot_download 的默认缓存位置
    os.path.join(os.environ.get("HF_HOME", "/app/.cache/huggingface"), "models--BAAI--bge-reranker-v2-m3"),
]


def _find_local_model() -> str | None:
    """在已知缓存路径中查找模型，支持 snapshot 格式"""
    for path in RERANKER_LOCAL_PATHS:
        # 直接目录（含 config.json）
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
        # huggingface_hub snapshot 格式：models--xx/snapshots/<hash>/
        snapshots_dir = os.path.join(path, "snapshots")
        if os.path.isdir(snapshots_dir):
            for entry in sorted(os.listdir(snapshots_dir), reverse=True):
                candidate = os.path.join(snapshots_dir, entry)
                if os.path.isfile(os.path.join(candidate, "config.json")):
                    return candidate
    return None


class Reranker:
    """Cross-Encoder Re-ranking — 延迟加载"""

    def __init__(self, model_name: str = RERANKER_MODEL):
        self._model = None
        self._model_name = model_name

    @property
    def model(self):
        """延迟加载 Cross-Encoder 模型（优先本地缓存）"""
        if self._model is None:
            try:
                # 确保 HuggingFace 镜像设置生效（国内服务器必须）
                if not os.environ.get("HF_ENDPOINT"):
                    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                os.environ.setdefault("HF_HUB_OFFLINE", "0")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

                from sentence_transformers import CrossEncoder

                # 优先从本地缓存加载
                local_path = _find_local_model()
                if local_path:
                    logger.info(f"从本地加载 Reranker: {local_path}")
                    self._model = CrossEncoder(local_path)
                else:
                    logger.info(f"从网络加载 Reranker: {self._model_name}")
                    self._model = CrossEncoder(self._model_name)
                logger.info(f"Reranker 模型加载完成")
            except Exception as e:
                logger.error(f"Reranker 模型加载失败: {e}")
                self._model = None
        return self._model

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """对候选 chunks 做精排

        Args:
            query: 用户查询
            chunks: RRF 融合后的候选切片列表
            top_k: 返回前 K 个

        Returns:
            按 rerank_score 降序排列的 chunks
        """
        if not chunks or len(chunks) <= 1:
            return chunks[:top_k]

        if self.model is None:
            logger.warning("Reranker 不可用，跳过精排")
            return chunks[:top_k]

        try:
            pairs = [(query, c.get("text", "")) for c in chunks]
            scores = self.model.predict(pairs)

            ranked = sorted(
                zip(chunks, scores),
                key=lambda x: -float(x[1])
            )

            results = []
            for chunk, score in ranked[:top_k]:
                hit = dict(chunk)
                hit["rerank_score"] = round(float(score), 4)
                results.append(hit)

            logger.info(f"Rerank 完成: {len(chunks)} 候选 → Top-{top_k}, "
                        f"最高分={results[0]['rerank_score'] if results else 'N/A'}")
            return results

        except Exception as e:
            logger.error(f"Rerank 失败，返回原序: {e}")
            return chunks[:top_k]
