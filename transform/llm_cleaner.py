"""LLM 结构化清洗 - 可插拔 Provider 架构

将原始抓取内容通过 LLM "脱水"为标准化知识资产：
- title: 精炼标题
- summary: 20-50 字核心总结
- tags: 3-5 个业务标签
- cleaned_content: 纯干货正文
"""

import json
import re
from dataclasses import dataclass
from openai import OpenAI

from config import get_llm_config


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


USER_PROMPT_TEMPLATE = """请对以下内容进行脱水处理：

【原标题】{title}
【来源】{source}
【作者】{author}
【原始标签】{original_tags}

---以下是正文---

{content}"""


class LLMCleaner:
    """LLM 结构化清洗器 - 支持多 Provider"""

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

        Args:
            title: 原始标题
            content: 原始正文
            source: 来源平台
            author: 作者
            original_tags: 平台原始标签

        Returns:
            CleanedKnowledge 结构化数据
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(
            title=title,
            source=source,
            author=author,
            original_tags=", ".join(original_tags) if original_tags else "无",
            content=content[:8000],  # 截断防止超 token 限制
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
            # LLM 调用失败时，返回一个基础版本（仅做最小清洗）
            return CleanedKnowledge(
                title=title,
                summary=content[:80] + "...",
                tags=["待分类"],
                cleaned_content=content,
            )

    def _parse_response(self, raw: str, fallback_title: str) -> CleanedKnowledge:
        """解析 LLM 的 JSON 响应"""
        # 尝试从 markdown 代码块中提取 JSON
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1)

        # 尝试直接解析
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # 尝试修复常见问题（如多余逗号）
            cleaned = re.sub(r",\s*}", "}", raw)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
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
