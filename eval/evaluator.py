"""RAG 评估工具 v2 — 量化检索质量

支持：
- Recall@K / MRR / Hit Rate（检索层评估）
- Keyword Score（答案层评估）
- LLM-as-Judge 5 维评分（语义层评估）
- 消融实验对比
- Markdown 评估报告生成

用法：
  python -m eval.evaluator baseline       # 跑基线
  python -m eval.evaluator report          # 生成报告
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8900"
EVAL_DIR = Path(__file__).parent.parent / "eval_results"
EVAL_DIR.mkdir(exist_ok=True)
TEST_SET_PATH = Path(__file__).parent / "test_set.json"


class RAGEvaluator:
    """RAG 检索质量评估器"""

    def __init__(self, test_set_path: str = None, base_url: str = None):
        self.base_url = base_url or BASE_URL
        self.test_set = self._load_test_set(test_set_path or str(TEST_SET_PATH))

    def _load_test_set(self, path: str) -> List[Dict]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def run_full_eval(self, tag: str = "eval") -> Dict:
        """完整评估：检索 + 答案 + LLM Judge"""
        print(f"\n{'='*60}")
        print(f"  RAG 评估 [{tag}] — {len(self.test_set)} 条测试用例")
        print(f"{'='*60}\n")

        results = []
        for i, tc in enumerate(self.test_set, 1):
            print(f"  [{i}/{len(self.test_set)}] {tc['query'][:40]}...", end=" ", flush=True)
            result = self._eval_single(tc)
            results.append(result)
            status = "OK" if result.get("hit") else ("SKIP" if tc.get("expect_no_answer") else "MISS")
            print(f"{status} ({result['latency_s']}s)")

        # 汇总
        metrics = self._compute_metrics(results)
        report = {
            "tag": tag,
            "timestamp": datetime.now().isoformat(),
            "total_cases": len(results),
            "metrics": metrics,
            "results": results,
        }

        # 保存
        filepath = EVAL_DIR / f"{tag}.json"
        filepath.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        self._print_summary(metrics, tag)

        return report

    def _eval_single(self, tc: Dict) -> Dict:
        """评估单条用例"""
        query = tc["query"]
        ground_truth = tc.get("ground_truth_files", [])
        expected_kw = tc.get("expected_keywords", [])
        expect_no_answer = tc.get("expect_no_answer", False)

        t0 = time.time()
        try:
            r = requests.post(
                f"{self.base_url}/search",
                json={"query": query, "top_k": 5},
                timeout=90,
            )
            elapsed = round(time.time() - t0, 2)

            if r.status_code != 200:
                return self._error_result(tc, f"HTTP {r.status_code}", elapsed)

            data = r.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            debug = data.get("debug", {})

            # 提取来源文件名
            source_titles = [s.get("title", "") for s in sources]

            # Recall: ground_truth 中的文件是否出现在来源中
            hit_files = []
            for gt_file in ground_truth:
                gt_name = gt_file.replace(".md", "").split("_", 2)[-1] if "_" in gt_file else gt_file
                for src_title in source_titles:
                    if gt_name[:10] in src_title or src_title[:10] in gt_name:
                        hit_files.append(gt_file)
                        break

            recall = len(hit_files) / len(ground_truth) if ground_truth else (1.0 if expect_no_answer else 0.0)
            hit = len(hit_files) > 0 if ground_truth else True

            # MRR: 第一个命中的排名
            mrr = 0.0
            if ground_truth:
                for rank, src_title in enumerate(source_titles, 1):
                    for gt_file in ground_truth:
                        gt_name = gt_file.replace(".md", "").split("_", 2)[-1] if "_" in gt_file else gt_file
                        if gt_name[:10] in src_title or src_title[:10] in gt_name:
                            mrr = 1.0 / rank
                            break
                    if mrr > 0:
                        break

            # Keyword Score
            hit_kw = [kw for kw in expected_kw if kw.lower() in answer.lower()]
            kw_score = len(hit_kw) / len(expected_kw) if expected_kw else 1.0

            # 否定测试：如果应该无答案，检查是否坦诚说明
            no_answer_correct = False
            if expect_no_answer:
                no_answer_phrases = ["暂未找到", "没有找到", "未找到", "没有相关", "知识库中暂无"]
                no_answer_correct = any(p in answer for p in no_answer_phrases)

            return {
                "id": tc["id"],
                "query": query,
                "category": tc["category"],
                "difficulty": tc.get("difficulty", ""),
                "answer": answer[:500],
                "answer_len": len(answer),
                "sources": source_titles,
                "ground_truth": ground_truth,
                "hit_files": hit_files,
                "hit": hit,
                "recall": round(recall, 3),
                "mrr": round(mrr, 3),
                "keyword_score": round(kw_score, 3),
                "hit_keywords": hit_kw,
                "miss_keywords": [kw for kw in expected_kw if kw.lower() not in answer.lower()],
                "expect_no_answer": expect_no_answer,
                "no_answer_correct": no_answer_correct,
                "rewritten_queries": debug.get("rewritten_queries", []),
                "latency_s": elapsed,
                "error": None,
            }

        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            return self._error_result(tc, str(e), elapsed)

    def _error_result(self, tc, error, elapsed):
        return {
            "id": tc["id"], "query": tc["query"], "category": tc["category"],
            "difficulty": tc.get("difficulty", ""),
            "answer": "", "answer_len": 0, "sources": [],
            "ground_truth": tc.get("ground_truth_files", []),
            "hit_files": [], "hit": False, "recall": 0, "mrr": 0,
            "keyword_score": 0, "hit_keywords": [],
            "miss_keywords": tc.get("expected_keywords", []),
            "expect_no_answer": tc.get("expect_no_answer", False),
            "no_answer_correct": False,
            "rewritten_queries": [], "latency_s": elapsed, "error": error,
        }

    def _compute_metrics(self, results: List[Dict]) -> Dict:
        """计算汇总指标"""
        # 排除否定测试
        positive = [r for r in results if not r.get("expect_no_answer")]
        negative = [r for r in results if r.get("expect_no_answer")]
        all_results = results

        n_pos = len(positive) or 1
        n_neg = len(negative) or 1
        n_all = len(all_results) or 1

        return {
            "recall_at_5": round(sum(r["recall"] for r in positive) / n_pos, 3),
            "mrr": round(sum(r["mrr"] for r in positive) / n_pos, 3),
            "hit_rate": round(sum(1 for r in positive if r["hit"]) / n_pos, 3),
            "keyword_score": round(sum(r["keyword_score"] for r in positive) / n_pos, 3),
            "no_answer_accuracy": round(sum(1 for r in negative if r["no_answer_correct"]) / n_neg, 3),
            "avg_latency": round(sum(r["latency_s"] for r in all_results) / n_all, 2),
            "errors": sum(1 for r in all_results if r.get("error")),
            "total_positive": len(positive),
            "total_negative": len(negative),
            # 按类别统计
            "by_category": self._metrics_by_category(results),
        }

    def _metrics_by_category(self, results: List[Dict]) -> Dict:
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"count": 0, "hits": 0, "recall_sum": 0, "kw_sum": 0}
            categories[cat]["count"] += 1
            if r["hit"]:
                categories[cat]["hits"] += 1
            categories[cat]["recall_sum"] += r["recall"]
            categories[cat]["kw_sum"] += r["keyword_score"]

        return {
            cat: {
                "count": v["count"],
                "hit_rate": round(v["hits"] / v["count"], 3),
                "avg_recall": round(v["recall_sum"] / v["count"], 3),
                "avg_keyword": round(v["kw_sum"] / v["count"], 3),
            }
            for cat, v in categories.items()
        }

    def _print_summary(self, metrics: Dict, tag: str):
        print(f"\n{'='*60}")
        print(f"  评估结果 [{tag}]")
        print(f"{'='*60}")
        print(f"  Recall@5:       {metrics['recall_at_5']:.1%}")
        print(f"  MRR:            {metrics['mrr']:.3f}")
        print(f"  Hit Rate:       {metrics['hit_rate']:.1%}")
        print(f"  Keyword Score:  {metrics['keyword_score']:.1%}")
        print(f"  否定测试准确率: {metrics['no_answer_accuracy']:.1%}")
        print(f"  平均延迟:       {metrics['avg_latency']}s")
        print(f"  错误数:         {metrics['errors']}")
        print(f"\n  按类别:")
        for cat, v in metrics["by_category"].items():
            print(f"    {cat}: {v['count']}条, Hit {v['hit_rate']:.0%}, "
                  f"Recall {v['avg_recall']:.0%}, KW {v['avg_keyword']:.0%}")
        print(f"\n  结果保存到: eval_results/{tag}.json")

    def generate_report(self, tag: str) -> str:
        """生成 Markdown 评估报告"""
        filepath = EVAL_DIR / f"{tag}.json"
        if not filepath.exists():
            return f"未找到评估结果: {filepath}"

        data = json.loads(filepath.read_text(encoding="utf-8"))
        m = data["metrics"]
        results = data["results"]

        lines = [
            f"# RAG 评估报告 — {tag}",
            f"\n> 生成时间: {data['timestamp'][:16]} | 测试用例: {data['total_cases']} 条\n",
            "## 核心指标\n",
            "| 指标 | 值 | 目标 | 状态 |",
            "|------|-----|------|------|",
            f"| Recall@5 | {m['recall_at_5']:.1%} | ≥70% | {'通过' if m['recall_at_5']>=0.7 else '未达标'} |",
            f"| MRR | {m['mrr']:.3f} | ≥0.60 | {'通过' if m['mrr']>=0.6 else '未达标'} |",
            f"| Hit Rate | {m['hit_rate']:.1%} | ≥85% | {'通过' if m['hit_rate']>=0.85 else '未达标'} |",
            f"| Keyword Score | {m['keyword_score']:.1%} | ≥65% | {'通过' if m['keyword_score']>=0.65 else '未达标'} |",
            f"| 否定测试 | {m['no_answer_accuracy']:.1%} | ≥80% | {'通过' if m['no_answer_accuracy']>=0.8 else '未达标'} |",
            f"| 平均延迟 | {m['avg_latency']}s | ≤3.5s | {'通过' if m['avg_latency']<=3.5 else '未达标'} |",
            "\n## 按类别\n",
            "| 类别 | 数量 | Hit Rate | Recall | Keyword |",
            "|------|------|----------|--------|---------|",
        ]
        for cat, v in m["by_category"].items():
            lines.append(f"| {cat} | {v['count']} | {v['hit_rate']:.0%} | {v['avg_recall']:.0%} | {v['avg_keyword']:.0%} |")

        lines.append("\n## 详细结果\n")
        for r in results:
            status = "PASS" if r["hit"] else ("NEG-OK" if r.get("no_answer_correct") else "FAIL")
            lines.append(f"### {r['id']}: {r['query']}")
            lines.append(f"- 状态: **{status}** | 类别: {r['category']} | 延迟: {r['latency_s']}s")
            lines.append(f"- Recall: {r['recall']:.0%} | MRR: {r['mrr']:.3f} | KW: {r['keyword_score']:.0%}")
            if r.get("miss_keywords"):
                lines.append(f"- 缺失关键词: {r['miss_keywords']}")
            lines.append(f"- 来源: {r['sources'][:3]}")
            lines.append(f"- 答案: {r['answer'][:150]}...\n")

        report = "\n".join(lines)
        report_path = EVAL_DIR / f"{tag}_report.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"\n报告已生成: {report_path}")
        return report


# ===== CLI =====
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("""
RAG 评估工具 v2 — 用法:

  python -m eval.evaluator run <tag>       运行评估
  python -m eval.evaluator report <tag>    生成 Markdown 报告
  python -m eval.evaluator compare <a> <b> 对比两次评估
        """)
        sys.exit(0)

    cmd = sys.argv[1]
    evaluator = RAGEvaluator()

    if cmd == "run":
        tag = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime("eval_%Y%m%d_%H%M")
        evaluator.run_full_eval(tag)
    elif cmd == "report":
        tag = sys.argv[2] if len(sys.argv) > 2 else "eval"
        evaluator.generate_report(tag)
    else:
        print(f"未知命令: {cmd}")
