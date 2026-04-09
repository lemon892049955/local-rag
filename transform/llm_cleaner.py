"""LLM 元数据提取 - 可插拔 Provider 架构

v0.6.4: 策略转变 — LLM 只负责提取元数据（标题/摘要/标签），正文保留原文。
原始全文经过基础降噪后直接落库，确保知识零丢失。
LLM 提取的标题/摘要/标签用于索引和 Wiki 编译。
"""

import json
import logging
import re
from dataclasses import dataclass
from openai import OpenAI

from config import get_llm_config

logger = logging.getLogger(__name__)

# 元数据提取只需看前 4000 字即可判断全文主题
META_PREVIEW_SIZE = 4000


@dataclass
class CleanedKnowledge:
    """结构化知识（元数据 + 原文正文）"""
    title: str
    summary: str
    tags: list[str]
    cleaned_content: str  # 现在是降噪后的原文，不再是 LLM 脱水版


EXTRACT_SYSTEM_PROMPT = """你是一个专业的知识资产提取助手。你的任务是为用户提供的文章提取**元数据**（标题、摘要、标签）。

你必须严格按以下 JSON 格式输出（不要输出任何其他内容）：

{
    "title": "精炼标题（如果原标题已很好可直接使用，否则重新拟一个更准确的）",
    "summary": "20-50字的核心一句话总结，概括这篇内容最有价值的信息",
    "tags": ["标签1", "标签2", "标签3"]
}

规则：
1. **标题**：准确、简洁，概括全文主题
2. **摘要**：一句话总结核心价值，让人看到摘要就知道这篇文章解决什么问题
3. **标签**：3-5 个高维业务标签（如：产品方法论、电商SOP、技术架构），不要过于宽泛

注意：只输出 JSON，不要有任何前缀或后缀文字。不需要输出正文内容。"""


EXTRACT_USER_TEMPLATE = """请为以下文章提取标题、摘要和标签：

【原标题】{title}
【来源】{source}
【作者】{author}
【原始标签】{original_tags}

---以下是正文（节选前部分）---

{content}"""


class LLMCleaner:
    """LLM 元数据提取器

    职责变更（v0.6.4）：
    - 之前：LLM 负责"脱水"（提取标题/摘要/标签 + 重写正文）
    - 现在：LLM 只提取元数据（标题/摘要/标签），正文保留原文

    好处：
    - 知识零丢失：原文完整保留，检索时能命中完整上下文
    - Token 成本降低：只需一次轻量调用（原来长文要多次）
    - 落库内容有血有肉：保留案例、论述、数据等完整信息
    """

    def __init__(self):
        config = get_llm_config()
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
        self.model = config["model"]

    async def clean(self, title: str, content: str, source: str = "",
                    author: str = "", original_tags=None) -> CleanedKnowledge:
        """提取元数据 + 基础降噪原文

        1. LLM 提取标题/摘要/标签（只看前 4000 字）
        2. 原文做基础降噪（去广告引导语、多余空行等）
        3. 返回 CleanedKnowledge（元数据 + 降噪原文）
        """
        # Step 1: 基础降噪（纯规则，不用 LLM）
        cleaned_text = self._basic_denoise(content)

        # Step 2: LLM 提取元数据
        meta = await self._extract_metadata(
            title=title,
            content=cleaned_text[:META_PREVIEW_SIZE],
            source=source,
            author=author,
            original_tags=original_tags,
        )

        return CleanedKnowledge(
            title=meta.get("title", title),
            summary=meta.get("summary", cleaned_text[:80] + "..."),
            tags=meta.get("tags", original_tags or ["待分类"]),
            cleaned_content=cleaned_text,
        )

    async def _extract_metadata(self, title, content, source, author, original_tags) -> dict:
        """LLM 轻量调用：只提取标题/摘要/标签"""
        user_prompt = EXTRACT_USER_TEMPLATE.format(
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
                    {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=500,  # 只输出元数据，500 token 足够
            )
            raw_output = response.choices[0].message.content.strip()
            parsed = self._parse_json(raw_output)
            if parsed:
                return parsed
        except Exception as e:
            logger.warning(f"LLM 元数据提取失败: {e}")

        return {}

    def _basic_denoise(self, content: str) -> str:
        """基础降噪（纯规则，不用 LLM）

        去除：
        - 公众号/小红书/知乎常见引导语和尾部广告
        - 账号信息（微信号、公众号名、抖音号等）
        - 招聘/商务合作/投稿信息
        - 版权声明/免责声明
        - 重复段落（微信抓取时常见）
        - 多余空行和分隔线堆叠
        保留：
        - 所有正文内容、案例、论述、数据
        - 图片 OCR 占位符/内容
        - Markdown 格式
        """
        text = content

        # === 1. 公众号/平台引导语 ===
        noise_patterns = [
            # 关注/转发引导
            r'点击.*?关注.*?\n',
            r'长按.*?识别.*?二维码.*?\n',
            r'扫码.*?关注.*?\n',
            r'▼.*?推荐阅读.*?▼',
            r'—+\s*END\s*—+',
            r'点击.*?阅读原文.*',
            r'(?:欢迎|记得)(?:点赞|转发|分享|收藏|在看).*?\n',
            r'↓+\s*点击.*?↓+',
            r'戳.*?"阅读原文".*',
            r'(?:喜欢|觉得有用).*?(?:点赞|在看|转发|分享).*?\n',
            r'点击下方.*?(?:名片|卡片|关注).*?\n',
            r'(?:星标|置顶).*?公众号.*?\n',
            # 小红书引导
            r'#小红书.*?\n',
            r'(?:关注我|关注作者).*?(?:获取|了解|更多).*?\n',
            r'(?:点赞|收藏|评论).*?(?:三连|支持).*?\n',
        ]
        for pat in noise_patterns:
            text = re.sub(pat, '\n', text, flags=re.IGNORECASE)

        # === 2. 账号/联系方式信息 ===
        account_patterns = [
            r'(?:微信号|微信|wx|wechat)\s*[:：]\s*\S+.*?\n',
            r'(?:公众号|订阅号|服务号)\s*[:：]\s*\S+.*?\n',
            r'(?:抖音号|抖音|dy)\s*[:：]\s*\S+.*?\n',
            r'(?:小红书号|小红书ID|xhs)\s*[:：]\s*\S+.*?\n',
            r'(?:微博|weibo)\s*[:：]\s*\S+.*?\n',
            r'(?:QQ|qq群?|QQ群?)\s*[:：]\s*\d+.*?\n',
            r'(?:电话|手机|tel|phone)\s*[:：]\s*[\d\-+]+.*?\n',
            r'(?:邮箱|email|mail)\s*[:：]\s*\S+@\S+.*?\n',
            r'添加.*?(?:微信|好友|助手).*?(?:备注|了解|咨询).*?\n',
            r'加.*?(?:群|微信|好友).*?(?:交流|学习|探讨).*?\n',
        ]
        for pat in account_patterns:
            text = re.sub(pat, '\n', text, flags=re.IGNORECASE)

        # === 3. 招聘/商务/投稿 ===
        biz_patterns = [
            r'(?:商务合作|广告合作|品牌合作).*?(?:请联系|详情|咨询).*?\n',
            r'(?:投稿|来稿).*?(?:请发|邮箱|联系).*?\n',
            r'(?:我们在招人|我们正在招|招聘|岗位).*?(?:欢迎|详情|见).*?\n',
            r'(?:转载|授权).*?(?:请联系|后台|注明).*?\n',
        ]
        for pat in biz_patterns:
            text = re.sub(pat, '\n', text, flags=re.IGNORECASE)

        # === 4. 版权/免责声明（通常在末尾，整段去除） ===
        tail_noise = [
            r'\n(?:免责声明|版权声明|原创声明|©|Copyright).*',
            r'\n(?:本文由|本文转自|本文来源).*?(?:授权|转载|发布).*',
            r'\n(?:侵权|侵删).*?(?:联系|删除).*',
        ]
        for pat in tail_noise:
            text = re.sub(pat, '', text, flags=re.IGNORECASE | re.DOTALL)

        # === 5. 公众号后台操作提示 ===
        text = re.sub(r'(?:公众号|后台).*?(?:回复|发送).*?(?:获取|领取|下载).*?\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'(?:公众号|后台).*?(?:点击|菜单).*?(?:咨询|客服|联系).*?\n', '\n', text, flags=re.IGNORECASE)

        # === 6. 重复段落去除（微信抓取常见：同一段出现两次） ===
        text = self._remove_duplicate_paragraphs(text)

        # === 7. 末尾重复链接去除（URL 和标题各出现一次） ===
        text = re.sub(r'\n(https?://\S+)\n\n\1\b', r'\n\1', text)  # 连续相同URL
        lines = text.split('\n')
        cleaned_lines = []
        prev_line = ''
        for line in lines:
            stripped = line.strip()
            # 跳过和上一行内容完全相同的行
            if stripped and stripped == prev_line:
                continue
            cleaned_lines.append(line)
            if stripped:
                prev_line = stripped
        text = '\n'.join(cleaned_lines)

        # === 8. 清理多余空行（保留最多2个） ===
        text = re.sub(r'\n{4,}', '\n\n\n', text)

        # === 9. 去首尾空白 ===
        text = text.strip()

        return text

    def _remove_duplicate_paragraphs(self, text: str) -> str:
        """去除重复段落（微信公众号抓取时，列表项和详情常重复）"""
        paragraphs = text.split('\n\n')
        seen = set()
        result = []
        for para in paragraphs:
            # 标准化：去空白、去 Markdown 符号后比较
            normalized = re.sub(r'[\s\-\*#>]+', '', para.strip())
            if len(normalized) < 10:
                result.append(para)  # 短段落保留（可能是分隔线等）
                continue
            if normalized not in seen:
                seen.add(normalized)
                result.append(para)
        return '\n\n'.join(result)

    def _parse_json(self, raw: str) -> dict:
        """解析 LLM 的 JSON 输出（带容错）"""
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
