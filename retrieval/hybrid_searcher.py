"""混合检索 + RAG 答案生成 (v2.2 最优组合版)

核心改进：
1. 查询改写与检索并行：原始 query 先检索，改写完毕后合并
2. SSE 流式输出：search_stream() 分阶段推送 sources → answer tokens
3. 意图路由 + 分路加权 + Reranker 精排 + Top3 裁剪
4. BM25 标题 3x 加权（在 bm25.py 中实现）
5. v2.2: DeepSeek + Prompt 6条铁律（含矛盾标记）+ SSE 前端流式
"""

import asyncio
import json
import logging
from typing import List, Dict, AsyncGenerator

from openai import OpenAI
from config import get_llm_config, DATA_DIR, WIKI_DIR
from retrieval.indexer import VectorIndexer
from retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)


# ===== Prompt =====

# 意图分类（规则优先，LLM fallback）
INTENT_TYPES = {
    "detail": "细节查找",      # 具体事实、数据、步骤
    "precise": "精确查找",     # 明确指定某篇文章/概念
    "overview": "主题概览",    # 宏观了解某个主题
    "entity": "实体查询",      # 查人物/公司/产品
    "fuzzy": "模糊查询",       # 之前看过一篇...、有没有关于...
    "negative": "否定查询",    # 知识库不可能有的内容
}

# 分路加权配置：不同意图下各类切片的权重
INTENT_WEIGHTS = {
    "detail":  {"data": 2.0, "topics": 1.5, "concepts": 0.5, "entities": 0.5, "moc": 0.2},
    "precise": {"data": 2.0, "topics": 1.5, "concepts": 0.5, "entities": 0.5, "moc": 0.2},
    "overview": {"data": 0.8, "topics": 1.5, "concepts": 2.0, "entities": 0.8, "moc": 1.5},
    "entity":  {"data": 1.0, "topics": 0.5, "concepts": 0.5, "entities": 3.0, "moc": 0.2},
    "fuzzy":   {"data": 1.5, "topics": 1.5, "concepts": 1.0, "entities": 1.0, "moc": 1.0},
    "negative": {"data": 1.0, "topics": 1.0, "concepts": 1.0, "entities": 1.0, "moc": 1.0},
}

# 意图分类规则关键词
INTENT_RULES = {
    "negative": ["天气", "炒菜", "炒鸡蛋", "量子计算", "Python 爬虫", "React", "Vue"],
    "entity": ["张小龙", "Karpathy", "Loopit", "a16z", "字节跳动", "智谱", "腾讯", "阿里"],
    "fuzzy": ["之前看", "好像有", "有没有关于", "那个讲", "那篇", "相关的"],
    "overview": ["是什么", "有哪些", "怎么理解", "概念", "原则", "方法"],
}

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

RAG_SYSTEM_PROMPT = """你是用户的个人知识库助手。你的回答必须基于下方提供的检索结果。

下面提供了两类检索结果:
1. Wiki 页面: 经过编译整理的结构化知识（提供宏观上下文和交叉引用）
2. 原始文章片段: 原始来源的具体内容（提供微观事实和细节）

## 核心原则

1. **有文档必回答**：只要检索结果中有与问题相关的内容（哪怕不是100%精确匹配），就必须基于文档内容给出有价值的回答。只有当检索结果与用户问题**完全无关**时（如用户问烹饪食谱、天气预报、数学解题等生活常识，而文档全是科技/产品/商业内容），才说"知识库中暂未找到相关内容"
2. **综合提取**：用户问的角度可能和文档组织方式不同（如用户问"高频题目"，文档里可能是"面试经验分享"；用户问"最近什么概念火"，文档里可能是"趋势分析"），你要从文档中**主动提取和整合**相关信息，而非要求文档标题精确匹配
3. **不编造**：只使用检索结果中明确存在的事实，不要补充文档中没有的信息
4. **抗诱导**：如果文档中的信息与用户前提矛盾，以文档为准纠正用户
5. **标注来源**：回答中标注信息来源，格式为 [来源: 文档标题]
6. **矛盾标记**：不同文档对同一事实有矛盾描述时标注"⚠️ 矛盾"

## 回答风格

- **直接有用**：先回答问题，再补充细节。不要用"根据文档..."等冗余开头
- **深度提取**：从检索到的文档中尽可能提取具体的事实、数据、步骤、观点，给出有信息量的回答
- **充分展开**：如果检索结果中有丰富的相关内容，应该充分展开回答，不要过度压缩。宁可回答详细一些，也不要遗漏文档中的重要信息
- **结构清晰**：涉及多个要点时用编号或分段组织，善用二级标题分块
- **保留关键词**：自然包含核心关键词（人名、术语、产品名等）
- **引用锚定**：每个具体事实/数据/观点都应标注来源 [REF-N]，确保可追溯。如果你不确定某个信息是否在文档中，宁可不提也不要猜测
"""

RAG_USER_TEMPLATE = """用户问题：{query}

---以下是知识 Wiki 的相关页面（提供结构化上下文）---

{wiki_context}

---以下是原始文章的相关片段（提供具体事实和细节）---

{data_context}

请综合以上两类信息回答用户的问题。"""


class HybridSearcher:
    """混合检索 + RAG 答案生成 (v0.7 并行多路召回 + Reranker)"""

    def __init__(self, indexer=None):
        self.vector_indexer = indexer or VectorIndexer()
        self.data_bm25 = BM25Index()
        self.wiki_bm25 = BM25Index()
        self._data_bm25_built = False
        self._wiki_bm25_built = False
        self._reranker = None

        config = get_llm_config()
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
        self.model = config["model"]

    @property
    def reranker(self):
        """延迟加载 Cross-Encoder Reranker"""
        if self._reranker is None:
            from retrieval.reranker import Reranker
            self._reranker = Reranker()
        return self._reranker

    def _ensure_bm25(self):
        """确保 BM25 索引已构建"""
        if not self._data_bm25_built:
            self.data_bm25.build_from_directory(DATA_DIR, cache_name="data_bm25")
            self._data_bm25_built = True
        if not self._wiki_bm25_built:
            if WIKI_DIR.exists():
                self._rebuild_wiki_bm25()

    def rebuild_bm25(self):
        """重建 BM25 索引（入库新文件后调用）"""
        self.data_bm25.build_from_directory(DATA_DIR, cache_name="data_bm25")
        self._data_bm25_built = True

    def _rebuild_wiki_bm25(self):
        """重建 Wiki BM25 索引"""
        from retrieval.chunker import SemanticChunker
        chunker = SemanticChunker()
        all_chunks = []
        for subdir in ["topics", "entities", "concepts", "moc"]:
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
        """v2.1 混合检索 + 意图路由 + 并行改写+检索 + Reranker 精排

        流程：意图分类 → 原始query检索‖查询改写 → 改写query补充检索 → 分路加权 → Reranker → Top3 → LLM
        """
        self._ensure_bm25()

        # 0. 意图分类
        intent = self._classify_intent(query)
        weights = INTENT_WEIGHTS.get(intent, INTENT_WEIGHTS["fuzzy"])
        logger.info(f"意图分类: {query[:30]} -> {intent}")

        # 1. 并行：原始 query 检索 + 查询改写（不再串行等待改写完成）
        original_results_future = asyncio.get_event_loop().run_in_executor(
            None, self._retrieve_all, query
        )
        rewrite_future = self._rewrite_query(query)

        # 等待两者完成
        original_results, rewritten_queries = await asyncio.gather(
            original_results_future, rewrite_future
        )
        logger.info(f"查询改写: {query} -> {rewritten_queries}")

        # 2. 用改写后的 query 做补充检索（排除原始 query，避免重复）
        extra_queries = [q for q in rewritten_queries if q != query]
        if extra_queries:
            extra_results = await asyncio.get_event_loop().run_in_executor(
                None, self._retrieve_extra, extra_queries
            )
        else:
            extra_results = {"data_vector": [], "data_bm25": [], "wiki_vector": [], "wiki_bm25": []}

        # 合并原始 + 改写的检索结果
        all_data_vector = original_results["data_vector"] + extra_results["data_vector"]
        all_data_bm25 = original_results["data_bm25"] + extra_results["data_bm25"]
        all_wiki_vector = original_results["wiki_vector"] + extra_results["wiki_vector"]
        all_wiki_bm25 = original_results["wiki_bm25"] + extra_results["wiki_bm25"]

        data_candidates = self._rrf_merge(all_data_vector, all_data_bm25, top_k=10)
        wiki_candidates = self._rrf_merge(all_wiki_vector, all_wiki_bm25, top_k=10)

        if not data_candidates and not wiki_candidates:
            return {
                "answer": "知识库中暂未找到与您问题相关的内容。请尝试换个关键词，或先录入相关内容。",
                "sources": [],
                "debug": {"rewritten_queries": rewritten_queries, "intent": intent, "data_hits": 0, "wiki_hits": 0},
            }

        # 3. 分路加权 + Reranker + Top3（复用辅助方法）
        all_candidates = self._apply_intent_weights(data_candidates, wiki_candidates, weights)
        all_candidates.sort(key=lambda x: -x.get("rrf_score", 0))
        candidates_for_rerank = all_candidates[:20]

        if len(candidates_for_rerank) > 2:
            try:
                reranked = self.reranker.rerank(query, candidates_for_rerank, top_k=5)
                candidates_for_rerank = reranked
            except Exception as e:
                logger.warning(f"Reranker 失败，使用加权排序: {e}")
                candidates_for_rerank = candidates_for_rerank[:5]
        else:
            candidates_for_rerank = candidates_for_rerank[:5]

        # 相关性阈值判断：如果最高分太低，认为没有相关内容
        if candidates_for_rerank:
            top_score = candidates_for_rerank[0].get("rerank_score", candidates_for_rerank[0].get("rrf_score", 0))
            if top_score < 0.005:  # 相关性过低阈值
                logger.info(f"检索结果相关性过低 ({top_score:.4f})，返回暂未找到")
                return {
                    "answer": "知识库中暂未找到与您问题相关的内容。请尝试换个关键词，或先录入相关内容。",
                    "sources": [],
                    "debug": {"rewritten_queries": rewritten_queries, "intent": intent, "top_score": top_score, "reason": "low_relevance"},
                }

        context_chunks = self._select_context_chunks(candidates_for_rerank, all_candidates)

        # 4. 构建 Context + LLM 生成答案
        wiki_context, data_context = self._build_context(context_chunks)
        prompt = RAG_USER_TEMPLATE.format(query=query, wiki_context=wiki_context, data_context=data_context)
        # 动态 max_tokens：context 越丰富，允许回答越长
        context_len = sum(len(h.get("text", "")) for h in context_chunks)
        max_tokens = min(4000, max(2000, context_len // 2))
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
                timeout=120,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"RAG LLM 调用失败: {type(e).__name__}: {e}")
            fallback_parts = []
            for hit in context_chunks[:3]:
                fallback_parts.append(f"**{hit.get('title', '未知')}**\n{hit['text'][:300]}")
            answer = "LLM 暂时不可用，以下是检索到的相关内容：\n\n" + "\n\n---\n\n".join(fallback_parts) if fallback_parts else f"搜索失败: {e}"

        # 5. 整理来源
        sources = self._build_sources(candidates_for_rerank)

        # 6. 提取高亮关键词
        highlight_keywords = self._extract_highlight_keywords(query, rewritten_queries)

        # 6. 异步回填判定（不阻塞返回）
        if len(sources) >= 3 and len(answer) > 500:
            try:
                asyncio.get_event_loop().create_task(
                    self._maybe_backfill(query, answer, sources)
                )
            except Exception as e:
                logger.debug(f"回填任务创建失败: {e}")

        # 9. 输出缓存 — 高质量回答落文件
        try:
            self._cache_answer(query, answer, sources)
        except Exception as e:
            logger.debug(f"答案缓存失败: {e}")

        return {
            "answer": answer,
            "sources": sources,
            "highlight_keywords": highlight_keywords,
            "debug": {
                "original_query": query,
                "intent": intent,
                "rewritten_queries": rewritten_queries,
                "total_candidates": len(all_candidates),
                "reranked_top5": len(candidates_for_rerank),
                "context_chunks": len(context_chunks),
            },
        }

    def _extract_highlight_keywords(self, query: str, rewritten_queries: List[str]) -> list[str]:
        """从查询和改写结果中提取高亮关键词"""
        from retrieval.tokenizer import tokenize
        all_keywords = set()
        for q in [query] + list(rewritten_queries):
            tokens = tokenize(q)
            # 过滤停用词和短词
            for t in tokens:
                if len(t) >= 2:
                    all_keywords.add(t)
        return list(all_keywords)[:15]

    def _classify_intent(self, query: str) -> str:
        """轻量级意图分类（规则优先，零 LLM 调用）

        Returns:
            intent type: detail/precise/overview/entity/fuzzy/negative
        """
        q = query.lower()

        # 规则匹配（优先级从高到低）
        for intent, keywords in INTENT_RULES.items():
            for kw in keywords:
                if kw.lower() in q:
                    return intent

        # 精确查找：包含具体文章/版本/文件名
        if any(kw in q for kw in ["v0.", "v1.", "版本", "那篇", "文章说", "文章提到"]):
            return "precise"

        # 细节查找：问具体事实/步骤/数据
        if any(kw in q for kw in ["怎么做", "具体", "步骤", "流程", "几个", "几层", "几分", "多少", "区别", "对比"]):
            return "detail"

        # 主题概览
        if any(kw in q for kw in ["是什么", "有哪些", "怎么理解", "什么意思", "概念", "原则", "核心"]):
            return "overview"

        # 默认：细节查找（保守策略，优先正文）
        return "detail"

    def _cache_answer(self, query: str, answer: str, sources: list):
        """高质量回答缓存到 outputs/qa/ — 复利积累"""
        # 只缓存有来源、有实质内容的回答
        if not sources or len(answer) < 100:
            return
        if "知识库中暂未找到" in answer or "LLM 暂时不可用" in answer:
            return

        import hashlib
        from config import BASE_DIR
        from datetime import datetime

        qa_dir = BASE_DIR / "outputs" / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)

        # 用 query hash 做文件名
        qhash = hashlib.md5(query.encode()).hexdigest()[:8]
        date = datetime.now().strftime("%Y%m%d")
        filepath = qa_dir / f"{date}_{qhash}.md"

        # 不覆盖已有缓存
        if filepath.exists():
            return

        source_list = "\n".join(f"  - {s.get('title', '未知')}" for s in sources[:5])
        content = f"""---
query: '{query}'
cached_at: '{datetime.now().strftime("%Y-%m-%d %H:%M")}'
sources_count: {len(sources)}
---

# Q: {query}

{answer}

## 来源
{source_list}
"""
        filepath.write_text(content, encoding="utf-8")
        logger.info(f"答案缓存: {filepath.name}")

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

    def _retrieve_all(self, query: str) -> dict:
        """对单个 query 执行全部检索路（同步，供线程池调用）"""
        data_vector = [h for h in self.vector_indexer.search(query, top_k=10)
                       if "wiki" not in h.get("source_file", "")]
        data_bm25 = self.data_bm25.search(query, top_k=10)
        wiki_vector = [h for h in self.vector_indexer.search(query, top_k=10)
                       if "wiki" in h.get("source_file", "")]
        wiki_bm25 = self.wiki_bm25.search(query, top_k=10) if self._wiki_bm25_built else []
        return {"data_vector": data_vector, "data_bm25": data_bm25,
                "wiki_vector": wiki_vector, "wiki_bm25": wiki_bm25}

    def _retrieve_extra(self, queries: List[str]) -> dict:
        """对额外改写 queries 执行检索（同步，供线程池调用）"""
        result = {"data_vector": [], "data_bm25": [], "wiki_vector": [], "wiki_bm25": []}
        for q in queries:
            r = self._retrieve_all(q)
            for k in result:
                result[k].extend(r[k])
        return result

    def _build_context(self, context_chunks: list) -> tuple:
        """从 Top5 切片构建 wiki_context 和 data_context

        保证原文至少 3 个 chunk（原文有具体细节，比 wiki 摘要信息量更大）
        """
        # 分离 wiki 和 data chunks
        data_hits = [h for h in context_chunks if h.get("source_type") == "data"]
        wiki_hits = [h for h in context_chunks if h.get("source_type") != "data"]

        # 如果原文不够 3 个，从全部候选中补充
        # （context_chunks 已经是 reranked top5，这里只做分配调整）

        wiki_parts = []
        data_parts = []
        ref_idx = 0
        for hit in data_hits[:5]:
            ref_idx += 1
            data_parts.append(
                f"[REF-{ref_idx}]【原始片段】\n"
                f"来源: {hit.get('title', '未知')}\n"
                f"章节: {hit.get('section_title', '未知')}\n"
                f"内容:\n{hit['text']}\n"
            )
        for hit in wiki_hits[:3]:
            ref_idx += 1
            type_label = {"topics": "主题页", "concepts": "概念卡", "entities": "实体页", "moc": "导航页"}.get(
                hit.get("source_type", ""), "Wiki")
            wiki_parts.append(
                f"[REF-{ref_idx}]【{type_label}】\n"
                f"标题: {hit.get('title', '未知')}\n"
                f"内容:\n{hit['text']}\n"
            )
        wiki_context = "\n---\n".join(wiki_parts) if wiki_parts else "_(Wiki 暂无相关页面)_"
        data_context = "\n---\n".join(data_parts) if data_parts else "_(原始文章暂无相关片段)_"
        return wiki_context, data_context

    def _select_context_chunks(self, reranked: list, all_candidates: list, min_data: int = 3, total: int = 7) -> list:
        """选择送入 LLM 的 context chunks，保证原文占比 + 质量门控

        策略：
        1. 保底保留 reranked top5 全部进入 context
        2. 仅超出 top5 的低分 chunk (rerank_score < 0.01) 才丢弃
        3. 先从 reranked 中取，原文不够从 all_candidates 补充
        """
        data_chunks = []
        wiki_chunks = []
        seen_ids = set()
        kept_count = 0

        for h in reranked:
            cid = h.get("chunk_id", id(h))
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            # 质量门控：保底保留 reranked top5，多余的低分才丢弃
            rerank_score = h.get("rerank_score", h.get("rrf_score", 1.0))
            if kept_count >= 5 and rerank_score < 0.01:
                logger.debug(f"丢弃低相关 chunk: {h.get('title', '')[:30]} score={rerank_score:.3f}")
                continue
            kept_count += 1
            if h.get("source_type") == "data":
                data_chunks.append(h)
            else:
                wiki_chunks.append(h)

        # 如果原文不够，从全部候选中补充
        if len(data_chunks) < min_data:
            for h in all_candidates:
                if len(data_chunks) >= min_data:
                    break
                cid = h.get("chunk_id", id(h))
                if cid in seen_ids:
                    continue
                if h.get("source_type") == "data":
                    data_chunks.append(h)
                    seen_ids.add(cid)

        # 组合：优先原文，补充 wiki，保证 top5 都进入 context
        selected_data = data_chunks[:min_data + 2]  # 最多 5 个原文
        remaining = total - len(selected_data)
        selected_wiki = wiki_chunks[:max(remaining, 2)]  # 至少 2 个 wiki

        result = selected_data + selected_wiki
        return result[:total]

    def _build_sources(self, candidates: list) -> list:
        """从候选结果构建去重来源列表"""
        seen_titles = set()
        sources = []
        for hit in candidates[:5]:
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
                    "source_type": hit.get("source_type", ""),
                })
        return sources

    async def search_stream(self, query: str, top_k: int = 5) -> AsyncGenerator[str, None]:
        """SSE 流式搜索：分阶段推送 sources → answer tokens

        Yields:
            SSE 格式的 data 行：
            - data: {"type": "sources", "sources": [...], "debug": {...}}
            - data: {"type": "token", "content": "..."}
            - data: {"type": "done"}
        """
        self._ensure_bm25()

        # 0. 意图分类
        intent = self._classify_intent(query)
        weights = INTENT_WEIGHTS.get(intent, INTENT_WEIGHTS["fuzzy"])

        # 1. 并行：原始 query 检索 + 查询改写
        original_results_future = asyncio.get_event_loop().run_in_executor(
            None, self._retrieve_all, query
        )
        rewrite_future = self._rewrite_query(query)
        original_results, rewritten_queries = await asyncio.gather(
            original_results_future, rewrite_future
        )

        # 2. 改写 query 补充检索
        extra_queries = [q for q in rewritten_queries if q != query]
        if extra_queries:
            extra_results = await asyncio.get_event_loop().run_in_executor(
                None, self._retrieve_extra, extra_queries
            )
        else:
            extra_results = {"data_vector": [], "data_bm25": [], "wiki_vector": [], "wiki_bm25": []}

        # 合并 + RRF
        data_candidates = self._rrf_merge(
            original_results["data_vector"] + extra_results["data_vector"],
            original_results["data_bm25"] + extra_results["data_bm25"], top_k=10)
        wiki_candidates = self._rrf_merge(
            original_results["wiki_vector"] + extra_results["wiki_vector"],
            original_results["wiki_bm25"] + extra_results["wiki_bm25"], top_k=10)

        if not data_candidates and not wiki_candidates:
            yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'debug': {'intent': intent}}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': '知识库中暂未找到与您问题相关的内容。'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # 3. 分路加权 + Reranker + Top3
        all_candidates = self._apply_intent_weights(data_candidates, wiki_candidates, weights)
        all_candidates.sort(key=lambda x: -x.get("rrf_score", 0))
        candidates_for_rerank = all_candidates[:20]

        if len(candidates_for_rerank) > 2:
            try:
                candidates_for_rerank = self.reranker.rerank(query, candidates_for_rerank, top_k=5)
            except Exception:
                candidates_for_rerank = candidates_for_rerank[:5]
        else:
            candidates_for_rerank = candidates_for_rerank[:5]

        context_chunks = self._select_context_chunks(candidates_for_rerank, all_candidates)
        sources = self._build_sources(candidates_for_rerank)

        # 推送 sources（用户立即看到来源）
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'debug': {'intent': intent, 'rewritten_queries': rewritten_queries}}, ensure_ascii=False)}\n\n"

        # 4. LLM 流式生成答案
        wiki_context, data_context = self._build_context(context_chunks)
        prompt = RAG_USER_TEMPLATE.format(query=query, wiki_context=wiki_context, data_context=data_context)
        # 动态 max_tokens
        context_len = sum(len(h.get("text", "")) for h in context_chunks)
        max_tokens = min(4000, max(2000, context_len // 2))

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
                stream=True,
                timeout=120,
            )
            full_answer = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_answer += delta.content
                    yield f"data: {json.dumps({'type': 'token', 'content': delta.content}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"SSE LLM 流式失败: {e}")
            yield f"data: {json.dumps({'type': 'token', 'content': f'LLM 暂时不可用: {e}'}, ensure_ascii=False)}\n\n"
            full_answer = ""

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        # 后台缓存
        if full_answer:
            try:
                self._cache_answer(query, full_answer, sources)
            except Exception:
                pass

    def _apply_intent_weights(self, data_candidates: list, wiki_candidates: list, weights: dict) -> list:
        """对候选列表施加意图权重"""
        all_candidates = []
        for c in data_candidates:
            c = dict(c)
            c["rrf_score"] = c.get("rrf_score", 0) * weights.get("data", 1.0)
            c["source_type"] = "data"
            all_candidates.append(c)
        for c in wiki_candidates:
            c = dict(c)
            source_file = c.get("source_file", "")
            if "concepts" in source_file:
                stype = "concepts"
            elif "entities" in source_file:
                stype = "entities"
            elif "moc" in source_file:
                stype = "moc"
            else:
                stype = "topics"
            c["rrf_score"] = c.get("rrf_score", 0) * weights.get(stype, 1.0)
            c["source_type"] = stype
            all_candidates.append(c)
        return all_candidates

    async def _rewrite_query(self, query: str) -> List[str]:
        """用 LLM 改写查询，生成多个检索 query

        优化：短 query（<8字）跳过改写，节省 2-3s 延迟
        """
        # 短 query 直接返回，不浪费 LLM 调用
        if len(query.strip()) < 8:
            return [query]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=200,
                timeout=8,
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
