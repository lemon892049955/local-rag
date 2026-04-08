"""LLM 结构化清洗 - 可插拔 Provider 架构

将原始抓取内容通过 LLM "脱水"为标准化知识资产：
- title: 精炼标题
- summary: 20-50 字核心总结
- tags: 3-5 个业务标签
- cleaned_content: 纯干货正文

v0.6.2: 滑动窗口清洗 — 长文分段清洗后合并，不再粗暴截断。
"""

import json
import logging
import re
from dataclasses import dataclass
from openai import OpenAI

from config import get_llm_config

logger = logging.getLogger(__name__)

# 单段最大字符数（留 buffer 给 Prompt + 输出）
CHUNK_SIZE = 6000
# 超过此长度启用分段清洗
LONG_THRESHOLD = 6500


@dataclass
class CleanedKnowledge:
    """LLM 清洗后的结构化知识"""
    title: str
    summary: str
    tags: list[str]
    cleaned_content: str


SYSTEM_PROMPT = """你是一个专业的知识资产提取助手。你的任务是将用户提供的原始网页/文章内容进行"脱水处理"，提取核心知识。

你必须严格按以下 JSON 格式输出（不要输出任何其他内容）：

{
    "title": "精炼标题（如果原标题已很好可直接使用，否则重新拟一个更准确的）",
    "summary": "20-50字的核心一句话总结，概括这篇内容最有价值的信息",
    "tags": ["标签1", "标签2", "标签3"],
    "cleaned_content": "清洗后的纯干货正文，使用 Markdown 格式排版"
}

清洗规则：
1. **降噪**：剥离所有营销话术、过多Emoji、无意义超链接、"关注公众号"等引导语
2. **标签**：提取 3-5 个高维业务标签（如：产品方法论、电商SOP、技术架构），不要使用过于宽泛的标签
3. **正文**：保留核心步骤、方法论、数据、案例等有价值内容，使用 Markdown 标题层级(##/###)组织
4. **摘要**：一句话总结必须精准概括核心价值，让人看到摘要就知道这篇文章解决什么问题

注意：只输出 JSON，不要有任何前缀或后缀文字。"""


# 分段清洗用的精简 Prompt（只提取正文，不生成标题/摘要/标签）
SEGMENT_SYSTEM_PROMPT = """你是一个专业的知识资产提取助手。你的任务是对文章的一个片段进行"脱水处理"。

你必须严格按以下 JSON 格式输出：
{
    "cleaned_content": "清洗后的纯干货正文，使用 Markdown 格式排版"
}

规则：
1. 剥离营销话术、Emoji、无意义超链接
2. 保留核心步骤、方法论、数据、案例
3. 使用 Markdown 标题层级(##/###)组织
4. 这只是长文的一个片段，不要写开头总结或结尾总结

只输出 JSON，不要其他文字。"""


# 合并后提取摘要/标签的轻量 Prompt
MERGE_SYSTEM_PROMPT = """你是一个专业的知识资产提取助手。用户会提供一篇已清洗的长文干货正文，请为它提取标题、摘要和标签。

你必须严格按以下 JSON 格式输出：
{
    "title": "精炼标题",
    "summary": "20-50字核心一句话总结",
    "tags": ["标签1", "标签2", "标签3"]
}

规则：
- 标签 3-5 个，高维业务标签，不要过于宽泛
- 摘要必须精准概括全文核心价值

只输出 JSON，不要其他文字。"""


USER_PROMPT_TEMPLATE = """请对以下内容进行脱水处理：

【原标题】{title}
【来源】{source}
【作者】{author}
【原始标签】{original_tags}

---以下是正文---

{content}"""


SEGMENT_USER_TEMPLATE = """请对以下长文片段进行脱水处理（这是第 {part_num}/{total_parts} 段）：

【原标题】{title}
【来源】{source}

---以下是片段内容---

{content}"""


class LLMCleaner:
    """LLM 结构化清洗器 - 支持多 Provider + 长文分段清洗"""

    def __init__(self):
        config = get_llm_config()
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
        self.model = config["model"]

    async def clean(self, title: str, content: str, source: str = "",
                    author: str = "", original_tags=None) -> CleanedKnowledge:
        """调用 LLM 清洗原始内容

        短文（≤6000字）：单次完整清洗
        长文（>6000字）：分段清洗 → 合并正文 → 轻量提取标题/摘要/标签
        """
        if len(content) <= LONG_THRESHOLD:
            # 短文：单次完整清洗
            return await self._clean_single(title, content, source, author, original_tags)
        else:
            # 长文：分段清洗
            logger.info(f"长文清洗模式: {len(content)} 字, 将分段处理")
            return await self._clean_long(title, content, source, author, original_tags)

    async def _clean_single(self, title, content, source, author, original_tags):
        """短文单次清洗（≤6000字）"""
        user_prompt = USER_PROMPT_TEMPLATE.format(
            title=title,
            source=source,
            author=author,
            original_tags=", ".join(original_tags) if original_tags else "无",
            content=content,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=4000,
            )
            raw_output = response.choices[0].message.content.strip()
            return self._parse_response(raw_output, title)
        except Exception as e:
            logger.warning(f"LLM 清洗失败: {e}")
            return CleanedKnowledge(
                title=title,
                summary=content[:80] + "...",
                tags=["待分类"],
                cleaned_content=content,
            )

    async def _clean_long(self, title, content, source, author, original_tags):
        """长文分段清洗（>6000字）

        策略：
        1. 按段落边界切成多段（每段≤6000字）
        2. 每段独立送 LLM 提取干货
        3. 拼合所有段的 cleaned_content
        4. 再做一次轻量调用提取全文标题/摘要/标签
        """
        # Step 1: 按段落边界分段
        segments = self._split_content(content)
        total = len(segments)
        logger.info(f"长文分为 {total} 段清洗")

        # Step 2: 逐段清洗
        cleaned_parts = []
        for i, seg in enumerate(segments):
            try:
                user_prompt = SEGMENT_USER_TEMPLATE.format(
                    part_num=i + 1,
                    total_parts=total,
                    title=title,
                    source=source,
                    content=seg,
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=4000,
                )
                raw = response.choices[0].message.content.strip()
                parsed = self._parse_json(raw)
                part_content = parsed.get("cleaned_content", seg) if parsed else seg
                if part_content:
                    cleaned_parts.append(part_content)
                logger.info(f"段 {i+1}/{total} 清洗完成: {len(part_content)} 字")
            except Exception as e:
                logger.warning(f"段 {i+1} 清洗失败: {e}，保留原文")
                cleaned_parts.append(seg)

        # Step 3: 合并正文
        merged_content = "\n\n".join(cleaned_parts)

        # Step 4: 轻量提取标题/摘要/标签（只看前3000字即可）
        try:
            merge_prompt = f"【原标题】{title}\n【来源】{source}\n【原始标签】{', '.join(original_tags) if original_tags else '无'}\n\n---以下是清洗后的全文（截取前部分）---\n\n{merged_content[:3000]}"
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": MERGE_SYSTEM_PROMPT},
                    {"role": "user", "content": merge_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            meta_raw = response.choices[0].message.content.strip()
            meta = self._parse_json(meta_raw) or {}
        except Exception as e:
            logger.warning(f"长文元数据提取失败: {e}")
            meta = {}

        return CleanedKnowledge(
            title=meta.get("title", title),
            summary=meta.get("summary", merged_content[:80] + "..."),
            tags=meta.get("tags", original_tags or ["待分类"]),
            cleaned_content=merged_content,
        )

    def _split_content(self, content: str) -> list:
        """按段落边界切分长文，每段不超过 CHUNK_SIZE 字符"""
        paragraphs = content.split("\n\n")
        segments = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 > CHUNK_SIZE:
                if current:
                    segments.append(current.strip())
                # 单段超长则硬切
                if len(para) > CHUNK_SIZE:
                    for i in range(0, len(para), CHUNK_SIZE):
                        segments.append(para[i:i + CHUNK_SIZE])
                    current = ""
                else:
                    current = para
            else:
                current = f"{current}\n\n{para}" if current else para

        if current.strip():
            segments.append(current.strip())

        return segments if segments else [content[:CHUNK_SIZE]]

    def _parse_response(self, raw: str, fallback_title: str) -> CleanedKnowledge:
        """解析 LLM 的 JSON 响应"""
        data = self._parse_json(raw)
        if not data:
            return CleanedKnowledge(
                title=fallback_title,
                summary="LLM 输出解析失败，请检查原始内容",
                tags=["待分类"],
                cleaned_content=raw,
            )

        return CleanedKnowledge(
            title=data.get("title", fallback_title),
            summary=data.get("summary", ""),
            tags=data.get("tags", ["待分类"]),
            cleaned_content=data.get("cleaned_content", ""),
        )

    def _parse_json(self, raw: str) -> dict:
        """解析 LLM 的 JSON 输出（带容错）"""
        # 尝试从 markdown 代码块中提取 JSON
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*}", "}", raw)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None
