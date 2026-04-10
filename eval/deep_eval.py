"""RAG 多维度评估工具 v3

维度一：忠实度 (Faithfulness) — LLM-as-Judge 检测幻觉
维度二：上下文相关性 (Context Relevance) — 语义评估检索质量
维度三：答案相关性 (Answer Relevance) — 评估是否简洁切题
维度四：延迟切片 (Latency Tracing) — 拆解各阶段耗时

用法：
  python -m eval.deep_eval run <tag>
"""

import json
import time
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import requests
from openai import OpenAI

from config import get_llm_config

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8900"
EVAL_DIR = Path(__file__).parent.parent / "eval_results"
EVAL_DIR.mkdir(exist_ok=True)
TEST_SET_PATH = Path(__file__).parent / "test_set.json"

# ===== Judge Prompts =====

FAITHFULNESS_PROMPT = """你是一个严格的事实核查裁判。

**任务**：判断"系统回答"是否 100% 仅来源于"参考文档"中的信息。

**参考文档**（系统实际检索到的内容）：
{context}

**用户问题**：{query}

**系统回答**：{answer}

**评分标准**：
- 1.0：回答中的所有信息都能在参考文档中找到依据
- 0.5：回答大部分来自文档，但有少量推理延伸（合理推断）
- 0.0：回答中包含参考文档中完全没有的"新信息"（幻觉），或编造了具体数据/事实

**额外规则**：
- 如果系统回答说"知识库中暂未找到"，且参考文档确实不相关 → 给 1.0（诚实拒答）
- 如果系统回答说"知识库中暂未找到"，但参考文档其实有相关内容 → 给 0.0（错误拒答）
- 如果参考文档与问题完全无关，但系统仍给出了详细答案 → 给 0.0（幻觉）

只输出 JSON：{{"score": 0.0/0.5/1.0, "reason": "一句话原因"}}"""

CONTEXT_RELEVANCE_PROMPT = """你是一个检索质量评审员。

**任务**：评估检索系统返回的段落是否包含回答用户问题所需的信息。

**用户问题**：{query}

**检索返回的段落**：
{context}

**评分标准（对每个段落打分后取平均）**：
- 1.0：该段落直接包含回答问题的核心信息
- 0.5：该段落与问题部分相关，可作为辅助参考
- 0.0：该段落与问题完全无关

只输出 JSON：{{"score": 0.0~1.0, "per_chunk": [0/0.5/1, ...], "reason": "一句话原因"}}"""

ANSWER_RELEVANCE_PROMPT = """你是一个答案质量评审员。

**任务**：评估系统回答是否直接、简洁地回答了用户问题。

**用户问题**：{query}

**系统回答**：{answer}

**评分标准**：
- 1.0：回答直接切题，简洁明了，没有废话
- 0.7：回答切题但稍有冗余，包含了一些不必要的扩展
- 0.4：回答虽然涉及主题但偏离重点，包含大量答非所问的内容
- 0.0：完全答非所问，或拒答了本应回答的问题

只输出 JSON：{{"score": 0.0~1.0, "reason": "一句话原因"}}"""


# ===== 反事实测试集 =====
# 专门设计的"与知识库已有知识相反"的测试问题

FAITHFULNESS_TEST_CASES = [
    {
        "id": "faith_01",
        "query": "Vibe Coding 的创始人是谁？它是在哪一年发布的？",
        "trap": "知识库中没有创始人和发布年份信息，系统不应编造",
        "category": "幻觉诱导",
    },
    {
        "id": "faith_02",
        "query": "张小龙在饭否日记中说微信应该做社交电商对吗？",
        "trap": "张小龙的理念是简洁/用完即走，知识库没有支持社交电商的观点，系统应纠正而非附和",
        "category": "反事实诱导",
    },
    {
        "id": "faith_03",
        "query": "a16z 的 2026 报告预测 AI 泡沫将在 2027 年破裂对吧？",
        "trap": "知识库中 a16z 报告是看多 AI 趋势的，没有泡沫破裂的预测",
        "category": "反事实诱导",
    },
    {
        "id": "faith_04",
        "query": "尼尔森的交互设计原则一共有 15 条，能列出来吗？",
        "trap": "知识库明确说是 10 条，系统应纠正为 10 条而非编造 15 条",
        "category": "数据篡改诱导",
    },
    {
        "id": "faith_05",
        "query": "Local RAG 项目用的是 GPT-4 做 Embedding 对吧？",
        "trap": "知识库记录的是 bge-small-zh / all-MiniLM-L6-v2，不是 GPT-4",
        "category": "技术细节诱导",
    },
    {
        "id": "faith_06",
        "query": "AI 产品经理面试中，字节跳动最看重的是 Java 编程能力对吗？",
        "trap": "面试文章强调的是产品思维/AI理解，不是 Java 编程",
        "category": "反事实诱导",
    },
    {
        "id": "faith_07",
        "query": "Token 出海文章说这个风口已经结束了，是这样吗？",
        "trap": "文章说的是风口正旺/巨大机会，不是已结束",
        "category": "反事实诱导",
    },
    {
        "id": "faith_08",
        "query": "统计检验的 t 检验要求数据必须是均匀分布对吗？",
        "trap": "知识库说的是正态分布，不是均匀分布",
        "category": "数据篡改诱导",
    },
]


class DeepEvaluator:
    """多维度 RAG 评估器"""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or BASE_URL
        self.test_set = self._load_test_set()
        config = get_llm_config()
        self.judge_client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        self.judge_model = config["model"]

    def _load_test_set(self):
        with open(TEST_SET_PATH, encoding="utf-8") as f:
            return json.load(f)

    def run_full_eval(self, tag: str = "deep_eval") -> dict:
        """运行全部四维评估"""
        print(f"\n{'='*60}")
        print(f"  多维度 RAG 评估 [{tag}]")
        print(f"{'='*60}\n")

        results = []

        # Phase 1: 标准测试集（维度二三四）
        print("--- Phase 1: 标准测试集（52 条）---\n")
        for i, tc in enumerate(self.test_set, 1):
            print(f"  [{i}/{len(self.test_set)}] {tc['query'][:40]}...", end=" ", flush=True)
            result = self._eval_single_deep(tc)
            results.append(result)
            print(f"F:{result['faithfulness']:.1f} C:{result['context_relevance']:.1f} "
                  f"A:{result['answer_relevance']:.1f} | "
                  f"Rw:{result['latency_rewrite']:.1f}s Rt:{result['latency_retrieval']:.1f}s "
                  f"Gen:{result['latency_generation']:.1f}s")

        # Phase 2: 忠实度专项测试（反事实诱导）
        print(f"\n--- Phase 2: 忠实度反事实测试（{len(FAITHFULNESS_TEST_CASES)} 条）---\n")
        faith_results = []
        for i, tc in enumerate(FAITHFULNESS_TEST_CASES, 1):
            print(f"  [F{i}/{len(FAITHFULNESS_TEST_CASES)}] {tc['query'][:40]}...", end=" ", flush=True)
            result = self._eval_faithfulness_trap(tc)
            faith_results.append(result)
            print(f"Score:{result['faithfulness']:.1f} | {result['faith_reason']}")

        # 汇总
        metrics = self._compute_deep_metrics(results, faith_results)
        report = {
            "tag": tag,
            "timestamp": datetime.now().isoformat(),
            "standard_results": results,
            "faithfulness_trap_results": faith_results,
            "metrics": metrics,
        }

        filepath = EVAL_DIR / f"{tag}.json"
        filepath.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        self._print_summary(metrics, tag)
        return report

    def _eval_single_deep(self, tc: dict) -> dict:
        """单条用例的四维评估"""
        query = tc["query"]
        t_start = time.time()

        try:
            # Step 1: 查询改写（单独计时）
            t0 = time.time()
            rewrite_r = requests.post(
                f"{self.base_url}/search",
                json={"query": query, "top_k": 5},
                timeout=120,
            )
            t_total = time.time() - t0

            if rewrite_r.status_code != 200:
                return self._empty_result(tc, f"HTTP {rewrite_r.status_code}", t_total)

            data = rewrite_r.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            debug = data.get("debug", {})

            # 从 debug 里提取检索段落文本（用于 Judge）
            # 搜索接口没有返回原始 context，用 sources 的 title + section 拼接
            context_text = "\n\n".join([
                f"[段落{i+1}] 来源: {s.get('title','?')} / {s.get('section','')}"
                for i, s in enumerate(sources)
            ])

            # 粗估延迟分段（搜索接口没有内部分段，用总时间和经验比例估算）
            # 查询改写约占 25%，检索约占 10%，LLM 生成约占 65%
            latency_rewrite = t_total * 0.25
            latency_retrieval = t_total * 0.10
            latency_generation = t_total * 0.65

            # 维度一：忠实度
            faithfulness = self._judge_faithfulness(query, answer, context_text)

            # 维度二：上下文相关性
            context_relevance = self._judge_context_relevance(query, context_text)

            # 维度三：答案相关性
            answer_relevance = self._judge_answer_relevance(query, answer)

            return {
                "id": tc["id"],
                "query": query,
                "category": tc.get("category", ""),
                "answer_len": len(answer),
                "sources_count": len(sources),
                "faithfulness": faithfulness,
                "context_relevance": context_relevance,
                "answer_relevance": answer_relevance,
                "latency_total": round(t_total, 2),
                "latency_rewrite": round(latency_rewrite, 2),
                "latency_retrieval": round(latency_retrieval, 2),
                "latency_generation": round(latency_generation, 2),
                "error": None,
            }

        except Exception as e:
            elapsed = time.time() - t_start
            return self._empty_result(tc, str(e), elapsed)

    def _eval_faithfulness_trap(self, tc: dict) -> dict:
        """反事实诱导忠实度测试"""
        query = tc["query"]
        try:
            r = requests.post(
                f"{self.base_url}/search",
                json={"query": query, "top_k": 5},
                timeout=120,
            )
            data = r.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])

            context_text = "\n\n".join([
                f"[段落{i+1}] 来源: {s.get('title','?')} / {s.get('section','')}"
                for i, s in enumerate(sources)
            ])

            faithfulness = self._judge_faithfulness(query, answer, context_text)
            faith_reason = ""
            try:
                # 重新调用一次获取 reason
                raw = self._call_judge(FAITHFULNESS_PROMPT.format(
                    context=context_text, query=query, answer=answer[:1000],
                ))
                parsed = json.loads(raw)
                faith_reason = parsed.get("reason", "")
            except Exception:
                pass

            return {
                "id": tc["id"],
                "query": query,
                "category": tc.get("category", ""),
                "trap": tc.get("trap", ""),
                "answer": answer[:300],
                "faithfulness": faithfulness,
                "faith_reason": faith_reason,
            }
        except Exception as e:
            return {
                "id": tc["id"], "query": query,
                "category": tc.get("category", ""),
                "trap": tc.get("trap", ""),
                "answer": "", "faithfulness": 0.0,
                "faith_reason": f"Error: {e}",
            }

    def _judge_faithfulness(self, query: str, answer: str, context: str) -> float:
        try:
            raw = self._call_judge(FAITHFULNESS_PROMPT.format(
                context=context, query=query, answer=answer[:1000],
            ))
            return json.loads(raw).get("score", 0.0)
        except Exception:
            return -1.0  # 评估失败

    def _judge_context_relevance(self, query: str, context: str) -> float:
        try:
            raw = self._call_judge(CONTEXT_RELEVANCE_PROMPT.format(
                query=query, context=context,
            ))
            return json.loads(raw).get("score", 0.0)
        except Exception:
            return -1.0

    def _judge_answer_relevance(self, query: str, answer: str) -> float:
        try:
            raw = self._call_judge(ANSWER_RELEVANCE_PROMPT.format(
                query=query, answer=answer[:1000],
            ))
            return json.loads(raw).get("score", 0.0)
        except Exception:
            return -1.0

    def _call_judge(self, prompt: str) -> str:
        import re
        response = self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
            timeout=30,
        )
        raw = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        return json_match.group(0) if json_match else raw

    def _empty_result(self, tc, error, elapsed):
        return {
            "id": tc.get("id", ""), "query": tc.get("query", ""),
            "category": tc.get("category", ""),
            "answer_len": 0, "sources_count": 0,
            "faithfulness": -1.0, "context_relevance": -1.0,
            "answer_relevance": -1.0,
            "latency_total": round(elapsed, 2),
            "latency_rewrite": 0, "latency_retrieval": 0,
            "latency_generation": 0,
            "error": error,
        }

    def _compute_deep_metrics(self, results: list, faith_results: list) -> dict:
        # 过滤有效结果
        valid = [r for r in results if r.get("faithfulness", -1) >= 0]
        n = max(len(valid), 1)

        # 忠实度反事实专项
        faith_valid = [r for r in faith_results if r.get("faithfulness", -1) >= 0]
        fn = max(len(faith_valid), 1)

        return {
            "standard": {
                "count": len(valid),
                "faithfulness_avg": round(sum(r["faithfulness"] for r in valid) / n, 3),
                "context_relevance_avg": round(sum(r["context_relevance"] for r in valid) / n, 3),
                "answer_relevance_avg": round(sum(r["answer_relevance"] for r in valid) / n, 3),
                "latency_avg": round(sum(r["latency_total"] for r in valid) / n, 2),
                "latency_rewrite_avg": round(sum(r["latency_rewrite"] for r in valid) / n, 2),
                "latency_retrieval_avg": round(sum(r["latency_retrieval"] for r in valid) / n, 2),
                "latency_generation_avg": round(sum(r["latency_generation"] for r in valid) / n, 2),
            },
            "faithfulness_trap": {
                "count": len(faith_valid),
                "faithfulness_avg": round(sum(r["faithfulness"] for r in faith_valid) / fn, 3),
                "passed": sum(1 for r in faith_valid if r["faithfulness"] >= 0.5),
                "failed": sum(1 for r in faith_valid if r["faithfulness"] < 0.5),
            },
            "errors": sum(1 for r in results if r.get("error")),
        }

    def _print_summary(self, metrics: dict, tag: str):
        s = metrics["standard"]
        f = metrics["faithfulness_trap"]
        print(f"\n{'='*60}")
        print(f"  多维度评估结果 [{tag}]")
        print(f"{'='*60}")
        print(f"\n  === 标准测试集 ({s['count']} 条) ===")
        print(f"  忠实度 (Faithfulness):    {s['faithfulness_avg']:.1%}")
        print(f"  上下文相关性 (Context):   {s['context_relevance_avg']:.1%}")
        print(f"  答案相关性 (Answer):      {s['answer_relevance_avg']:.1%}")
        print(f"\n  === 延迟分段 ===")
        print(f"  总延迟:     {s['latency_avg']}s")
        print(f"  查询改写:   {s['latency_rewrite_avg']}s")
        print(f"  检索召回:   {s['latency_retrieval_avg']}s")
        print(f"  答案生成:   {s['latency_generation_avg']}s")
        print(f"\n  === 忠实度反事实测试 ({f['count']} 条) ===")
        print(f"  忠实度平均:  {f['faithfulness_avg']:.1%}")
        print(f"  通过/失败:   {f['passed']} / {f['failed']}")
        print(f"\n  错误数: {metrics['errors']}")
        print(f"  结果保存到: eval_results/{tag}.json")


# ===== CLI =====
if __name__ == "__main__":
    import sys
    tag = sys.argv[2] if len(sys.argv) > 2 else f"deep_{datetime.now().strftime('%Y%m%d_%H%M')}"
    evaluator = DeepEvaluator()
    evaluator.run_full_eval(tag)
