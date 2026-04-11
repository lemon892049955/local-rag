# Wiki 编译质量优化方案 — P0 + P1

> 版本: v1.0 | 日期: 2026-04-10 | 状态: 待实施

---

## 一、现状诊断

### 1.1 系统架构概览

```
URL → Fetcher → OCR → LLM元数据提取 → Markdown落库 → 向量索引
                                                          ↓
                                              编译队列（串行）
                                                          ↓
                                    LLM编译计划 → 创建/追加页面 → 重建索引
```

当前编译链路核心模块：
- `services/ingest_pipeline.py` — 入库管线，最终调用 `enqueue_compile()`
- `wiki/compile_queue.py` — 串行编译队列，逐篇文章触发编译
- `wiki/compiler.py` — 编译引擎核心，包含 4 个 Prompt 模板
- `wiki/index_builder.py` — `build_lightweight_summary()` 为 LLM 提供 Wiki 现状
- `wiki/page_store.py` — 页面 CRUD 操作

### 1.2 量化问题数据

| 指标 | 当前值 | 问题 |
|------|--------|------|
| Topic 页面总数 | 48 | 过多，存在大量重复 |
| Entity 页面总数 | 1 | 严重偏少，很多实体被错建为 topic |
| Insight 页面总数 | 0 | 完全空白 |
| 可识别的重复页面组 | ≥8 组 | 见下方详细列表 |
| 洞察追加与核心内容重复率 | ~60% | 追加内容大量复述已有观点 |

### 1.3 已确认的重复页面清单

| # | 重复组 | 涉及页面 | 重复类型 |
|---|--------|---------|----------|
| 1 | AI产品经理焦虑 | `AI产品经理的焦虑与挑战.md` ↔ `AI产品经理的职场焦虑与挑战.md` | 同主题两个页面，内容几乎一致 |
| 2 | AI短剧角色设计 | `AI仿真人短剧角色设计.md` / `角色设计指南.md` / `角色设计易翻车点.md` / `角色设计补三视图.md` | 同一篇文章被拆成 4 个页面 |
| 3 | AI时代产品经理 | `新实践.md` / `新角色与工作方式.md` / `核心能力重塑.md` | 同主题碎片化 |
| 4 | AI时代文科生 | `就业前景.md` ↔ `就业前景：机遇还是挑战？.md` | 完全重复 |
| 5 | Karpathy知识库 | `搭建教程.md` ↔ `构建指南.md` | 重复 |
| 6 | Vibe Coding | `工作流程.md` ↔ `工作流程详解.md` | 重复 |
| 7 | 梁宁商业洞察 | `商业洞察与个人成长.md` ↔ `五年商业思考.md` | 重复 |
| 8 | AI技术梳理 | `AI技术梳理.md` ↔ `李开复对人工智能技术的梳理.md` | 重复 |
| 9 | AI真人漫剧 | `崛起与题材趋势.md` ↔ `流行趋势.md` | 重复 |

### 1.4 洞察追加质量问题样例

**典型案例：`AI产品经理的职场焦虑与挑战.md`**

核心内容已包含：
> "许多AI功能的添加更多是为了安抚管理层的焦虑，而非真正的市场需求"

追加洞察中来源 `260409_VkZZQTkp` 写的：
> "许多AI功能的添加更多是为了安抚老板的焦虑，而非真正的市场需求，这可能导致资源的浪费和产品方向的偏离"

**问题**：完全是已有内容的换一种说法，零增量信息。7 条追加洞察中，至少 5 条与核心内容近似重复。

**典型案例 2**：来源 `梁宁：五年商业洞察与个人成长` 被追加到 `AI产品经理焦虑` 页面，内容是泛泛而谈的"个人成长与商业闭环"，与页面主题关联度极低。

---

## 二、根因分析

### 根因 1：LLM 编译决策时信息不充分

**现状**：`build_lightweight_summary()` 只输出每个页面的「标题 + 60字摘要」一行摘要

```python
# index_builder.py:112
lines.append(f"- [{subdir}] {title}: {summary}")
```

**后果**：当 Wiki 已有 `AI产品经理的职场焦虑与挑战` 时，LLM 只能看到：
```
- [topics] AI产品经理的职场焦虑与挑战: AI产品经理面临技术落地难、商业化挑战和职场焦虑。
```

一行文字不足以让 LLM 判断新文章「AI产品经理的焦虑与挑战」是否与之重复。

### 根因 2：缺乏量化语义信号

编译计划完全依赖 LLM 文本理解，没有任何 Embedding 距离、相似度分数等**定量信号**辅助判断。

### 根因 3：分类边界模糊

`COMPILE_PLAN_PROMPT` 中：
```
5. 页面路径格式: topics/概念名.md 或 entities/实体名.md
```

只有一句话的指引，LLM 在 topic vs entity 的边界上没有明确标准，默认全部建 topic。

### 根因 4：追加洞察无去重机制

`APPEND_INSIGHT_PROMPT` 只说"不要重复已有内容"，但：
1. 页面内容被截断到 4000 字（`page_content[:4000]`），LLM 可能看不到全部已有洞察
2. 没有量化的重复检测，纯靠 LLM 自觉
3. 没有"关联度阈值"概念，低关联内容也被追加

---

## 三、优化方案

### 方案总览

```
                    ┌──────────────────────────────────────────────┐
                    │           改造后的编译链路                     │
                    └──────────────────────────────────────────────┘

新文章入库
    │
    ▼
[Phase 0] Embedding 预计算 ← 复用入库时已有的向量
    │
    ▼
[Phase 1 · P0] 语义去重门控 ─── 找相似页面，注入编译计划 Prompt
    │
    ▼
[Phase 2 · P1] 丰富摘要 + Taxonomy ─── 让 LLM 看到聚类结构 + 分类树
    │
    ▼
[Phase 3] LLM 编译计划 ─── 基于充分信息做决策
    │
    ├─ 新建页面 → [Phase 4 · P1] 分类决策（topic/entity/insight + 归属分类）
    │
    └─ 追加洞察 → [Phase 5 · P0] 去重门控 ─── Embedding 相似度检查 + 强化 Prompt
```

---

### 方案 A [P0]：编译前语义去重门控

**目标**：在 LLM 制定编译计划前，用 Embedding 相似度找到已有的高度相似页面，注入 Prompt 中作为定量依据。

**改造文件**：`wiki/compiler.py`

#### A.1 新增方法：`_find_similar_pages()`

```python
async def _find_similar_pages(self, article_text: str, threshold: float = 0.82) -> list[dict]:
    """用 Embedding 检索与新文章高度相似的 Wiki 页面

    Args:
        article_text: 文章正文（取前 2000 字做 embedding 检索）
        threshold: 相似度阈值（cosine similarity），默认 0.82

    Returns:
        [{"path": "topics/xxx.md", "title": "...", "similarity": 0.89, "preview": "前200字..."}]
    """
    from retrieval.indexer import VectorIndexer
    indexer = VectorIndexer()

    # 用文章前 2000 字做语义检索，在 Wiki 页面中找近似
    hits = indexer.search(article_text[:2000], top_k=8)

    # 只保留 Wiki 来源的命中
    wiki_hits = [h for h in hits if "wiki" in h.get("source_file", "")]

    similar = []
    seen_pages = set()  # 同一页面去重
    for hit in wiki_hits:
        # ChromaDB 返回的 distance 是 cosine distance = 1 - cosine_similarity
        cosine_sim = 1 - hit.get("distance", 1.0)
        source_file = hit.get("source_file", "")

        # 提取页面路径（从 source_file 中）
        page_path = self._extract_wiki_page_path(source_file)
        if not page_path or page_path in seen_pages:
            continue
        seen_pages.add(page_path)

        if cosine_sim >= threshold:
            similar.append({
                "path": page_path,
                "title": hit.get("title", ""),
                "similarity": round(cosine_sim, 3),
                "preview": hit.get("text", "")[:200],
            })

    # 按相似度降序
    similar.sort(key=lambda x: -x["similarity"])
    return similar[:5]  # 最多返回 5 个
```

#### A.2 改造 `compile()` 主流程

在 Step 2（获取 Wiki 摘要）和 Step 3（LLM 制定计划）之间插入去重检查：

```python
# Step 2.5: 语义去重检查
similar_pages = await self._find_similar_pages(body)
similar_pages_info = ""
if similar_pages:
    parts = []
    for sp in similar_pages:
        parts.append(
            f"  - [{sp['path']}] {sp['title']} (相似度: {sp['similarity']:.0%})\n"
            f"    预览: {sp['preview'][:100]}..."
        )
    similar_pages_info = (
        "\n⚠️ 以下 Wiki 页面与本文高度相似，请优先考虑 update 而非 new：\n"
        + "\n".join(parts)
    )
```

#### A.3 改造 `COMPILE_PLAN_PROMPT`

在 Prompt 中注入相似页面信息：

```python
COMPILE_PLAN_PROMPT = """你是一个知识 Wiki 编译器。用户刚刚添加了一篇新文章到知识库。
你的任务是分析这篇文章，并决定如何将其知识编译到 Wiki 中。

{wiki_summary}

{similar_pages_section}

---以下是新文章的内容---
...（后续不变）

请输出一个 JSON 编译计划。规则：
1. 一篇文章最多创建 1-2 个新页面，聚焦文章最核心的主题，不要过度拆分
2. ⚠️ 如果上方列出了相似度 ≥ 85% 的已有页面，你必须优先使用 update_pages 追加到该页面，而非创建新页面
3. 如果相似度在 70%-85% 之间，需在 reason 中说明为什么选择新建而非更新
4. 不要为过于宽泛的概念建页（如"技术""互联网""AI""工作"）
...
"""
```

#### A.4 改造 `_make_plan()` 传参

```python
plan = await self._make_plan(
    wiki_summary=wiki_summary,
    similar_pages_section=similar_pages_info,  # 新增
    filename=filename,
    ...
)
```

---

### 方案 B [P0]：洞察追加去重

**目标**：消除追加洞察与已有内容的重复，引入关联度检查。

**改造文件**：`wiki/compiler.py`

#### B.1 强化 `APPEND_INSIGHT_PROMPT`

```python
APPEND_INSIGHT_PROMPT = """你是一个知识 Wiki 编译器。以下是一篇新文章的内容，请提取与 Wiki 页面「{page_title}」相关的新增洞察。

新文章文件名: {filename}
新文章内容:
{article_content}

已有 Wiki 页面「{page_title}」的当前内容:
{page_content}

规则:
1. 只提取与该页面主题**直接相关**的新信息
2. **严格去重**：在输出前，逐条检查你要追加的每个观点：
   a) 如果已有核心内容或历史洞察中已有相同/近似表述 → 必须跳过
   b) 如果新信息只是已有观点的换一种说法、换一个角度复述 → 必须跳过
   c) 只有**真正的增量信息**（新事实、新案例、新数据、新观点）才值得追加
3. **关联度要求**：只追加与页面核心主题直接相关的信息
   - 如果新文章的主题与本页面的核心主题关联度低（如"梁宁商业思考"与"AI产品经理焦虑"），应该跳过
   - 不要强行建立关联
4. 如果经过筛选后没有任何增量信息，直接输出空字符串 ""
5. 输出纯文本段落（可用 Markdown 格式），不要输出标题和 front-matter
6. 如果新文章与已有内容有矛盾，用 "> ⚠️ 矛盾:" 标记
7. 来源引用: [来源: {filename}]
8. 每条洞察尽量精简，只写增量部分，不要重复铺垫已有上下文

只输出要追加的内容文本，不要任何前缀后缀。"""
```

#### B.2 新增代码层去重检查

在 `_generate_insight()` 返回后、实际追加前，增加 Embedding 相似度校验：

```python
async def _check_insight_novelty(self, insight: str, page_content: str) -> bool:
    """检查追加洞察是否与已有内容有足够差异

    Returns:
        True = 有增量价值，可以追加
        False = 与已有内容过于相似，应跳过
    """
    from retrieval.indexer import VectorIndexer
    indexer = VectorIndexer()

    # 将洞察做 embedding
    insight_embedding = indexer.embedding_fn([insight])[0]

    # 将已有内容分段，逐段比对
    paragraphs = [p.strip() for p in page_content.split("\n\n") if len(p.strip()) > 50]
    if not paragraphs:
        return True

    page_embeddings = indexer.embedding_fn(paragraphs[:20])  # 最多取20段

    # 计算最大相似度
    import numpy as np
    max_sim = 0.0
    for pe in page_embeddings:
        sim = np.dot(insight_embedding, pe)  # 已归一化，dot = cosine sim
        max_sim = max(max_sim, sim)

    logger.info(f"洞察与已有内容最大相似度: {max_sim:.3f}")
    return max_sim < 0.80  # 相似度 < 0.80 才认为有增量价值
```

#### B.3 改造更新流程（在 `compile()` 的 4b 步骤中）

```python
# 4b: 更新已有页面 (Append-only)
for page_info in plan.get("update_pages", []):
    ...
    insight = await self._generate_insight(...)
    if insight and insight.strip():
        # [新增] 代码层去重校验
        is_novel = await self._check_insight_novelty(insight, existing["body"])
        if not is_novel:
            logger.info(f"洞察与已有内容过于相似，跳过追加: {page_path}")
            log_details.append(f"跳过: {page_path} (洞察无增量)")
            continue
        page_store.append_insight(page_path, today, filename, insight)
        ...
```

---

### 方案 C [P1]：丰富摘要上下文

**目标**：让 LLM 在编译决策时看到带聚类分组的 Wiki 全貌，而非扁平列表。

**改造文件**：`wiki/index_builder.py`

#### C.1 改造 `build_lightweight_summary()`

```python
def build_lightweight_summary() -> str:
    """为 LLM 编译决策生成带聚类的丰富摘要

    改造：从扁平列表升级为按关键词聚类的分组摘要，
    让 LLM 能看到"同类页面已有哪些"，做出更好的归并决策。
    """
    from utils.frontmatter import read_frontmatter as _read_frontmatter
    from collections import defaultdict
    import re

    # 收集所有页面信息
    all_pages = []
    for subdir in ["topics", "entities", "insights"]:
        dir_path = WIKI_DIR / subdir
        if not dir_path.exists():
            continue
        for md_file in sorted(dir_path.glob("*.md")):
            meta = _read_frontmatter(md_file)
            if meta:
                all_pages.append({
                    "path": f"{subdir}/{md_file.name}",
                    "type": subdir,
                    "title": meta.get("title", md_file.stem),
                    "summary": (meta.get("summary", "") or "")[:80],
                    "tags": meta.get("tags", []),
                    "sources_count": len(meta.get("sources", [])),
                })

    if not all_pages:
        return "当前 Wiki 为空，没有任何页面。"

    # 按标签聚类
    tag_clusters = defaultdict(list)
    untagged = []
    for page in all_pages:
        tags = page.get("tags", [])
        if tags:
            # 取第一个标签作为主分类
            primary_tag = tags[0] if isinstance(tags, list) else str(tags)
            tag_clusters[primary_tag].append(page)
        else:
            # 无标签的，按标题关键词简单聚类
            untagged.append(page)

    # 对无标签页面做简单的关键词聚类
    keyword_clusters = defaultdict(list)
    for page in untagged:
        title = page["title"]
        # 提取关键词做聚类（简单规则）
        matched = False
        for keyword in ["AI", "产品经理", "2026", "短剧", "知识库", "Vibe", "Claude"]:
            if keyword.lower() in title.lower():
                keyword_clusters[keyword].append(page)
                matched = True
                break
        if not matched:
            keyword_clusters["其他"].append(page)

    # 合并聚类结果
    all_clusters = {}
    for tag, pages in tag_clusters.items():
        all_clusters[tag] = pages
    for kw, pages in keyword_clusters.items():
        key = f"[关键词]{kw}"
        if key in all_clusters:
            all_clusters[key].extend(pages)
        else:
            all_clusters[key] = pages

    # 生成输出
    lines = [f"当前 Wiki 共 {len(all_pages)} 个页面，按主题分组如下：\n"]

    for cluster_name, pages in sorted(all_clusters.items(), key=lambda x: -len(x[1])):
        lines.append(f"【{cluster_name}】({len(pages)} 个页面)")
        for p in pages:
            lines.append(
                f"  - [{p['type']}] {p['title']}: {p['summary']} (来源:{p['sources_count']}篇)"
            )
        lines.append("")

    return "\n".join(lines)
```

#### C.2 效果对比

**改造前**（扁平列表）：
```
当前 Wiki 已有页面：
- [topics] AI产品经理的焦虑与挑战: 年薪60W的AI产品经理面临技术落地难、产品创新难...
- [topics] AI产品经理的职场焦虑与挑战: AI产品经理面临技术落地难...
- [topics] AI时代产品经理的新实践: Anthropic产品经理Cat Wu分享...
...（48行平铺）
```

**改造后**（聚类分组）：
```
当前 Wiki 共 48 个页面，按主题分组如下：

【产品经理】(5 个页面)
  - [topics] AI产品经理的焦虑与挑战: 年薪60W... (来源:2篇)
  - [topics] AI产品经理的职场焦虑与挑战: 面临技术落地难... (来源:5篇)
  - [topics] AI时代产品经理的新实践: Anthropic PM... (来源:9篇)
  - [topics] AI时代产品经理的新角色: ... (来源:5篇)
  - [topics] 腾讯产品经理晋升影响力半径: ... (来源:1篇)

【AI短剧/AIGC视频】(5 个页面)
  - [topics] AI仿真人短剧角色设计: ... (来源:1篇)
  - [topics] AI仿真人短剧角色设计指南: ... (来源:1篇)
  ...
```

LLM 一眼就能看到"产品经理"分组下已有 5 个页面，自然会倾向于合并而非新建。

---

### 方案 D [P1]：自演化 Taxonomy（分类树）

**目标**：维护一棵 LLM 自动演化的分类树，解决分类扁平化和 topic/entity/insight 类型失衡问题。

**新增文件**：`wiki/_taxonomy.yaml` + `wiki/taxonomy.py`

#### D.1 Taxonomy 数据结构

`wiki/_taxonomy.yaml`：

```yaml
# 由编译引擎自动维护，请勿手动编辑
# 上次更新: 2026-04-10
version: 1
categories:
  AI趋势与产业:
    description: AI技术和产业发展趋势
    pages:
      - topics/2026年AI领域三大关键趋势.md
      - topics/2026年AI与数据科学五大趋势.md
      - topics/2026年AI产业趋势聚合化自动化普惠化.md
    children:
      AI硬件与投资:
        pages:
          - topics/2026年AI硬件市场趋势.md
          - topics/2026年低空经济与商业航天市场趋势.md
      AI工具与工作流:
        pages:
          - topics/Claude_Code终端编程助手.md
          - topics/2026年AI工作流革命.md

  产品方法论:
    description: 产品管理方法、实践和职业发展
    pages:
      - topics/AI时代产品经理的新实践.md
    children:
      产品经理职业:
        pages:
          - topics/AI产品经理的职场焦虑与挑战.md
          - topics/腾讯产品经理晋升影响力半径.md

  AIGC内容创作:
    description: AI辅助的内容创作和短剧制作
    pages:
      - topics/AI仿真人短剧角色设计.md
      - topics/AI真人漫剧在抖音的崛起与题材趋势.md
```

#### D.2 新增 `wiki/taxonomy.py`

```python
"""Wiki 分类树维护器

维护 wiki/_taxonomy.yaml，为编译决策提供分类上下文。
分类树由 LLM 自动演化，每次编译时更新。
"""

import logging
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI
from config import get_llm_config, WIKI_DIR

logger = logging.getLogger(__name__)

TAXONOMY_FILE = WIKI_DIR / "_taxonomy.yaml"

CLASSIFY_PROMPT = """你是一个知识分类专家。以下是当前的分类树和一个新创建的 Wiki 页面信息。
请决定这个页面应该归入哪个分类节点。

当前分类树:
{taxonomy_tree}

新页面信息:
- 路径: {page_path}
- 标题: {page_title}
- 摘要: {page_summary}
- 类型: {page_type}

规则:
1. 如果有合适的已有分类，直接归入
2. 如果需要新建分类，给出分类名和描述
3. 页面类型建议：
   - topic: 围绕概念/方法论/技术的知识主题（如"Vibe Coding工作流"）
   - entity: 围绕具体人物/公司/产品的事实（如"霸王茶姬""Karpathy"）
   - insight: 跨主题的对比分析
4. 只输出 JSON，格式:
{{"category": "分类名", "subcategory": "子分类名(可选)", "suggested_type": "topic|entity|insight", "reason": "原因"}}"""

REORG_PROMPT = """你是一个知识分类专家。以下是当前的分类树。请审视并优化分类结构。

当前分类树:
{taxonomy_tree}

规则:
1. 合并重复或高度重叠的分类
2. 如果某分类下只有1个页面且可以归入其他分类，建议合并
3. 如果某分类下超过8个页面，建议拆分子分类
4. 输出优化后的完整 YAML 格式分类树
"""


def load_taxonomy() -> dict:
    """加载分类树"""
    if TAXONOMY_FILE.exists():
        return yaml.safe_load(TAXONOMY_FILE.read_text(encoding="utf-8")) or {}
    return {"version": 1, "categories": {}}


def save_taxonomy(taxonomy: dict):
    """保存分类树"""
    from datetime import datetime
    yaml_str = yaml.dump(
        taxonomy, default_flow_style=False,
        allow_unicode=True, sort_keys=False,
    )
    header = f"# 由编译引擎自动维护，请勿手动编辑\n# 上次更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    TAXONOMY_FILE.write_text(header + yaml_str, encoding="utf-8")


def get_taxonomy_summary() -> str:
    """生成分类树的文本摘要（供 Prompt 使用）"""
    taxonomy = load_taxonomy()
    categories = taxonomy.get("categories", {})
    if not categories:
        return "当前暂无分类体系。"

    lines = ["当前分类体系："]
    for cat_name, cat_data in categories.items():
        if isinstance(cat_data, dict):
            pages = cat_data.get("pages", [])
            children = cat_data.get("children", {})
            lines.append(f"├─ {cat_name} ({len(pages)} 个页面)")
            for child_name, child_data in (children or {}).items():
                child_pages = child_data.get("pages", []) if isinstance(child_data, dict) else []
                lines.append(f"│  ├─ {child_name} ({len(child_pages)} 个页面)")

    return "\n".join(lines)


def add_page_to_taxonomy(page_path: str, category: str, subcategory: str = ""):
    """将页面添加到分类树"""
    taxonomy = load_taxonomy()
    categories = taxonomy.setdefault("categories", {})

    if category not in categories:
        categories[category] = {"description": "", "pages": [], "children": {}}

    cat = categories[category]
    if subcategory:
        children = cat.setdefault("children", {})
        if subcategory not in children:
            children[subcategory] = {"pages": []}
        sub = children[subcategory]
        if page_path not in sub.get("pages", []):
            sub.setdefault("pages", []).append(page_path)
    else:
        if page_path not in cat.get("pages", []):
            cat.setdefault("pages", []).append(page_path)

    save_taxonomy(taxonomy)
```

#### D.3 编译时集成

在 `COMPILE_PLAN_PROMPT` 中注入分类树：

```python
taxonomy_info = taxonomy.get_taxonomy_summary()
```

在新页面创建后，调用 LLM 分类并更新 taxonomy：

```python
# 创建新页面后
for page_info in plan.get("new_pages", []):
    ...
    # 分类归档
    classify_result = await self._classify_page(page_path, page_title, page_summary, page_type)
    if classify_result:
        taxonomy.add_page_to_taxonomy(
            page_path,
            classify_result.get("category", "未分类"),
            classify_result.get("subcategory", ""),
        )
```

---

## 四、实施计划

### Phase 1: P0 改造（预计 2-3 小时）

| 步骤 | 改造内容 | 涉及文件 | 风险 |
|------|---------|---------|------|
| 1.1 | 新增 `_find_similar_pages()` 方法 | `wiki/compiler.py` | 低 - 纯新增 |
| 1.2 | 改造 `compile()` 主流程，插入去重检查 | `wiki/compiler.py` | 低 |
| 1.3 | 改造 `COMPILE_PLAN_PROMPT`，注入相似页面信息 | `wiki/compiler.py` | 低 |
| 1.4 | 改造 `_make_plan()` 签名和调用 | `wiki/compiler.py` | 低 |
| 1.5 | 强化 `APPEND_INSIGHT_PROMPT` | `wiki/compiler.py` | 低 |
| 1.6 | 新增 `_check_insight_novelty()` 方法 | `wiki/compiler.py` | 中 - 依赖 numpy |
| 1.7 | 改造 4b 步骤，加入去重校验 | `wiki/compiler.py` | 低 |

### Phase 2: P1 改造（预计 2-3 小时）

| 步骤 | 改造内容 | 涉及文件 | 风险 |
|------|---------|---------|------|
| 2.1 | 重写 `build_lightweight_summary()` | `wiki/index_builder.py` | 中 - 聚类逻辑需调试 |
| 2.2 | 新增 `wiki/taxonomy.py` | 新增文件 | 低 |
| 2.3 | 新增 `wiki/_taxonomy.yaml` 初始化 | 新增文件 | 低 |
| 2.4 | 编译计划 Prompt 注入 taxonomy | `wiki/compiler.py` | 低 |
| 2.5 | 新页面创建后自动分类 | `wiki/compiler.py` | 中 - 额外 LLM 调用 |
| 2.6 | `COMPILE_PLAN_PROMPT` 增加 type 决策指引 | `wiki/compiler.py` | 低 |

### Phase 3: 存量修复（手动/半自动，可选）

| 步骤 | 内容 | 说明 |
|------|------|------|
| 3.1 | 合并已确认的 8 组重复页面 | 手动选择保留页，合并 sources |
| 3.2 | 初始化 taxonomy | 基于现有 48 个页面生成初始分类树 |
| 3.3 | 重新分类错误的 topic → entity | 如"霸王茶姬"应为 entity |

---

## 五、验证标准

### P0 验收标准

- [ ] 同一篇文章的不同版本（如微信、小红书双发）入库后，不再重复建页
- [ ] 追加洞察与已有内容的重复率降至 < 20%
- [ ] 低关联度文章不再被强行追加到无关页面

### P1 验收标准

- [ ] 编译计划中 LLM 能看到聚类后的 Wiki 全貌
- [ ] 新建页面能被自动归入分类树
- [ ] entity 类型的页面占比从 2% 提升到 15%+

---

## 六、附录：改造影响范围

| 文件 | 改动类型 | 影响 |
|------|---------|------|
| `wiki/compiler.py` | 修改 | 核心 - 新增方法 + 改造 Prompt + 改造流程 |
| `wiki/index_builder.py` | 修改 | 中等 - 重写 `build_lightweight_summary()` |
| `wiki/taxonomy.py` | 新增 | 低 - 独立模块 |
| `wiki/_taxonomy.yaml` | 新增 | 低 - 数据文件 |
| `requirements.txt` | 检查 | 确认 numpy 已在依赖中 |

**不受影响的模块**：`ingestion/*`, `transform/*`, `storage/*`, `retrieval/*`, `services/*`, `wecom/*`
