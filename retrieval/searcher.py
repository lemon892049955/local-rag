"""RAG 检索 + 答案生成

流程：
1. 用户查询 -> 向量检索 Top-K 切片
2. 组装 Context + Query 成 Prompt
3. 调用 LLM 生成带引用来源 (Citation) 的答案
"""

from openai import OpenAI
from config import get_llm_config
from retrieval.indexer import VectorIndexer


RAG_SYSTEM_PROMPT = """你是用户的个人知识库助手。根据用户的问题和下面提供的知识库检索结果，给出准确、有条理的回答。

规则：
1. 只使用提供的检索结果来回答问题，不要编造信息
2. 如果检索结果不足以回答问题，坦诚说明"知识库中暂未找到相关信息"
3. 回答中需要标注信息来源，格式为 [来源: 文档标题]
4. 回答简洁直接，使用中文"""

RAG_USER_TEMPLATE = """用户问题：{query}

---以下是知识库检索结果---

{context}

请根据以上检索结果回答用户的问题。"""


class RAGSearcher:
    """RAG 检索 + 答案生成"""

    def __init__(self, indexer=None):
        self.indexer = indexer or VectorIndexer()
        config = get_llm_config()
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
        self.model = config["model"]

    async def search(self, query: str, top_k: int = 3) -> dict:
        """完整的 RAG 搜索流程

        Args:
            query: 用户的自然语言查询
            top_k: 检索的切片数量

        Returns:
            {"answer": "...", "sources": [{"title": "...", "distance": 0.12}, ...]}
        """
        # 1. 向量检索
        hits = self.indexer.search(query, top_k=top_k)

        if not hits:
            return {
                "answer": "知识库中暂未找到与您问题相关的内容。请尝试换个关键词，或先通过 /ingest 接口录入相关内容。",
                "sources": [],
            }

        # 2. 组装 Context
        context_parts = []
        for i, hit in enumerate(hits, 1):
            context_parts.append(
                f"【检索结果 {i}】\n"
                f"来源文档: {hit.get('title', '未知')}\n"
                f"章节: {hit.get('section_title', '未知')}\n"
                f"相似度距离: {hit.get('distance', 'N/A')}\n"
                f"内容:\n{hit['text']}\n"
            )
        context = "\n---\n".join(context_parts)

        # 3. 调用 LLM 生成答案
        prompt = RAG_USER_TEMPLATE.format(query=query, context=context)

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
            answer = f"LLM 调用失败: {e}\n\n以下是检索到的原始内容片段:\n\n{context}"

        # 4. 整理来源信息
        sources = [
            {
                "title": hit.get("title", "未知"),
                "section": hit.get("section_title", ""),
                "distance": round(hit.get("distance", 0), 4),
                "source_url": hit.get("source_url", ""),
            }
            for hit in hits
        ]

        return {"answer": answer, "sources": sources}
