"""混合检索 + RAG 答案生成 (v0.3 Wiki 并行多路召回)

核心改进：
1. 查询改写：LLM 将模糊问题分解为检索关键词
2. 并行多路召回：Wiki(Top-2 宏观结构) + Data(Top-3 微观细节)
3. RRF 融合排序：Reciprocal Rank Fusion 合并结果
4. 摘要切片加权：summary chunk 在排序中获得额外加成
5. 异步回填：高价值答案后台生成 Wiki 洞察页
"""

import asyncio
import logging
from typing import List, Dict

from openai import OpenAI
from config import get_llm_config, DATA_DIR, WIKI_DIR
from retrieval.indexer import VectorIndexer
from retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)


# ===== Prompt =====

REWRITE_PROMPT = """你是一个搜索查询优化器。用户会提出一个自然语言问题，你需要将它改写为 2-3 个精确的检索查询词/短语，用于在知识库中检索。

规则：
1. 提取问题中的核心实体和关键概念
2. 生成 2-3 个互补的检索 query（一个偏语义，一个偏关键词）
3. 只输出 query，每行一个，不要编号和解释

例子：
用户问题：之前那篇写如何用 AI 优化大批量版权投诉工作流的文章提到了哪三个工具？
输出：
AI 优化版权投诉工作流
版权投诉 AI 工具
投诉工作流自动化"""

RAG_SYSTEM_PROMPT = """你是用户的个人知识库助手。下面提供了两类检索结果:
1. Wiki 页面: 经过编译整理的结构化知识（提供宏观上下文和交叉引用）
2. 原始文章片段: 原始来源的具体内容（提供微观事实和细节）

请综合两类信息回答用户的问题。Wiki 提供框架，原始片段补充细节。

规则：
1. 优先使用检索结果中的信息回答，不要编造
2. 如果检索结果不足以回答，坦诚说明"知识库中暂未找到相关信息"
3. 回答中标注信息来源，格式为 [来源: 文档标题]
4. 如果多个文档涉及同一主题，综合整理后回答
5. 回答简洁直接，使用中文
6. 如果检索到的内容与问题关联度低，不要强行关联"""

RAG_USER_TEMPLATE = """用户问题：{query}

---以下是知识 Wiki 的相关页面（提供结构化上下文）---

{wiki_context}

---以下是原始文章的相关片段（提供具体事实和细节）---

{data_context}

请综合以上两类信息回答用户的问题。"""


class HybridSearcher:
    """混合检索 + RAG 答案生成 (v0.3 并行多路召回)"""

    def __init__(self, indexer=None):
        self.vector_indexer = indexer or VectorIndexer()
        self.data_bm25 = BM25Index()
        self.wiki_bm25 = BM25Index()
        self._data_bm25_built = False
        self._wiki_bm25_built = False

        config = get_llm_config()
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
        self.model = config["model"]

    def _ensure_bm25(self):
        """确保 BM25 索引已构建"""
        if not self._data_bm25_built:
            self.data_bm25.build_from_directory(DATA_DIR)
            self._data_bm25_built = True
        if not self._wiki_bm25_built:
            if WIKI_DIR.exists():
                self._rebuild_wiki_bm25()

    def rebuild_bm25(self):
        """重建 BM25 索引（入库新文件后调用）"""
        self.data_bm25.build_from_directory(DATA_DIR)
        self._data_bm25_built = True

    def _rebuild_wiki_bm25(self):
        """重建 Wiki BM25 索引"""
        from retrieval.chunker import SemanticChunker
        chunker = SemanticChunker()
        all_chunks = []
        for subdir in ["topics", "entities", "insights"]:
            wiki_subdir = WIKI_DIR / subdir
            if not wiki_subdir.exists():
                continue
            for md_file in sorted(wiki_subdir.glob("*.md")):
                chunks = chunker.chunk_file(md_file)
                for c in chunks:
                    all_chunks.append({
                        "chunk_id": f"wiki:{c.chunk_id}",
                        "text": c.text,
                        "title": c.title,
                        "section_title": c.section_title,
                        "chunk_type": c.chunk_type,
                        "source_file": c.source_file,
                        "tags": ", ".join(c.metadata.get("tags", [])),
                        "source_url": c.metadata.get("source_url", ""),
                    })
        if all_chunks:
            self.wiki_bm25.build_from_chunks(all_chunks)
        self._wiki_bm25_built = True

    async def search(self, query: str, top_k: int = 5) -> dict:
        """完整的混合检索 + RAG 流程 (v0.3 并行多路召回)

        Wiki(Top-2 宏观结构) + Data(Top-3 微观细节) 一起喂 LLM。

        Args:
            query: 用户自然语言查询
            top_k: Data 路返回的切片数

        Returns:
            {"answer": "...", "sources": [...], "debug": {...}}
        """
        self._ensure_bm25()

        # 1. 查询改写
        queries = await self._rewrite_query(query)
        logger.info(f"查询改写: {query} -> {queries}")

        # 2. Data 路召回 (微观细节)
        data_vector_hits = []
        for q in queries:
            hits = self.vector_indexer.search(q, top_k=top_k)
            data_vector_hits.extend(hits)

        data_bm25_hits = []
        for q in queries:
            hits = self.data_bm25.search(q, top_k=top_k)
            data_bm25_hits.extend(hits)

        data_merged = self._rrf_merge(data_vector_hits, data_bm25_hits, top_k=3)

        # 3. Wiki 路召回 (宏观结构)
        wiki_merged = []
        if WIKI_DIR.exists() and self._wiki_bm25_built:
            wiki_vector_hits = []
            for q in queries:
                hits = self.vector_indexer.search(q, top_k=2)
                # 筛选 wiki 来源的结果
                wiki_vector_hits.extend([h for h in hits if "wiki" in h.get("source_file", "")])

            wiki_bm25_hits = []
            for q in queries:
                hits = self.wiki_bm25.search(q, top_k=2)
                wiki_bm25_hits.extend(hits)

            wiki_merged = self._rrf_merge(wiki_vector_hits, wiki_bm25_hits, top_k=2)

        if not data_merged and not wiki_merged:
            return {
                "answer": "知识库中暂未找到与您问题相关的内容。请尝试换个关键词，或先录入相关内容。",
                "sources": [],
                "debug": {"queries": queries, "data_hits": 0, "wiki_hits": 0},
            }

        # 4. 组装双层 Context
        wiki_context = "_(Wiki 暂无相关页面)_"
        if wiki_merged:
            wiki_parts = []
            for i, hit in enumerate(wiki_merged, 1):
                wiki_parts.append(
                    f"【Wiki 页面 {i}】\n"
                    f"标题: {hit.get('title', '未知')}\n"
                    f"内容:\n{hit['text']}\n"
                )
            wiki_context = "\n---\n".join(wiki_parts)

        data_context = "_(原始文章暂无相关片段)_"
        if data_merged:
            data_parts = []
            for i, hit in enumerate(data_merged, 1):
                data_parts.append(
                    f"【原始片段 {i}】\n"
                    f"来源: {hit.get('title', '未知')}\n"
                    f"章节: {hit.get('section_title', '未知')}\n"
                    f"内容:\n{hit['text']}\n"
                )
            data_context = "\n---\n".join(data_parts)

        # 5. LLM 生成答案
        prompt = RAG_USER_TEMPLATE.format(
            query=query,
            wiki_context=wiki_context,
            data_context=data_context,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"LLM 调用失败: {e}\n\n检索到的内容:\n{data_context[:1000]}"

        # 6. 整理来源（去重）
        all_hits = wiki_merged + data_merged
        seen_titles = set()
        sources = []
        for hit in all_hits:
            title = hit.get("title", "未知")
            if title not in seen_titles:
                seen_titles.add(title)
                sources.append({
                    "title": title,
                    "section": hit.get("section_title", ""),
                    "distance": round(hit.get("distance", 0), 4),
                    "rrf_score": round(hit.get("rrf_score", 0), 4),
                    "source_url": hit.get("source_url", ""),
                    "match_type": hit.get("match_type", ""),
                })

        # 7. 异步回填判定（不阻塞返回）
        if len(sources) >= 3 and len(answer) > 500:
            try:
                asyncio.get_event_loop().create_task(
                    self._maybe_backfill(query, answer, sources)
                )
            except Exception:
                pass

        return {
            "answer": answer,
            "sources": sources,
            "debug": {
                "original_query": query,
                "rewritten_queries": queries,
                "wiki_candidates": len(wiki_merged),
                "data_candidates": len(data_merged),
            },
        }

    async def _maybe_backfill(self, query: str, answer: str, sources: list):
        """异步回填：高价值答案生成 Wiki 洞察页（后台执行，不阻塞查询）"""
        try:
            from wiki.page_store import list_wiki_pages
            # 检查是否已有近似洞察页
            existing = list_wiki_pages()
            existing_titles = [p.get("title", "").lower() for p in existing]
            # 简单去重：查询关键词是否已在某个洞察页标题中
            query_lower = query.lower()
            for t in existing_titles:
                if query_lower in t or t in query_lower:
                    return  # 已有近似页，跳过

            from wiki.compile_queue import get_queue
            # 回填走编译队列，此处只记录日志
            from wiki.index_builder import append_log
            append_log("QUERY_BACKFILL_CANDIDATE", query[:50], [f"答案长度: {len(answer)}", f"来源数: {len(sources)}"])
        except Exception:
            pass

    async def _rewrite_query(self, query: str) -> List[str]:
        """用 LLM 改写查询，生成多个检索 query"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            text = response.choices[0].message.content.strip()
            queries = [q.strip() for q in text.split("\n") if q.strip()]
            # 始终保留原始查询
            if query not in queries:
                queries.insert(0, query)
            return queries[:3]  # 最多 3 个
        except Exception as e:
            logger.warning(f"查询改写失败: {e}")
            return [query]

    def _rrf_merge(self, vector_hits: List[Dict], bm25_hits: List[Dict],
                   top_k: int = 5, k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion 融合排序

        RRF Score = Σ 1/(k + rank_i)

        额外加权：
        - 摘要切片 (chunk_type=summary) 获得 1.3x 加成
        - 同时命中两路的切片获得额外加成
        """
        scores = {}     # chunk_id -> rrf_score
        chunk_map = {}  # chunk_id -> chunk_data
        hit_sources = {}  # chunk_id -> set of sources

        # 向量路排名
        seen_ids = set()
        rank = 0
        for hit in vector_hits:
            cid = hit.get("chunk_id", "")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            rank += 1
            rrf = 1.0 / (k + rank)
            scores[cid] = scores.get(cid, 0) + rrf
            chunk_map[cid] = hit
            hit_sources.setdefault(cid, set()).add("vector")

        # BM25 路排名
        seen_ids = set()
        rank = 0
        for hit in bm25_hits:
            cid = hit.get("chunk_id", "")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            rank += 1
            rrf = 1.0 / (k + rank)
            scores[cid] = scores.get(cid, 0) + rrf
            if cid not in chunk_map:
                chunk_map[cid] = hit
            hit_sources.setdefault(cid, set()).add("bm25")

        # 加权调整
        for cid, score in scores.items():
            chunk = chunk_map[cid]

            # 摘要切片加成 1.3x
            if chunk.get("chunk_type") == "summary":
                score *= 1.3

            # 双路命中加成 1.2x
            if len(hit_sources.get(cid, set())) >= 2:
                score *= 1.2

            scores[cid] = score

        # 排序 + 取 Top-K
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

        results = []
        for cid, score in ranked:
            hit = dict(chunk_map[cid])
            hit["rrf_score"] = score
            sources = hit_sources.get(cid, set())
            if len(sources) >= 2:
                hit["match_type"] = "向量+关键词"
            elif "vector" in sources:
                hit["match_type"] = "向量语义"
            else:
                hit["match_type"] = "关键词"
            results.append(hit)

        return results
