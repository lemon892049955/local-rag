"""RAG 效果评测工具

用法:
  1. 修改前: python eval_rag.py save before
  2. 改完代码重启服务后: python eval_rag.py save after
  3. 对比: python eval_rag.py diff

结果保存在 eval_results/ 目录，可反复对比。
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_URL = "http://localhost:8900"
EVAL_DIR = Path(__file__).parent / "eval_results"
EVAL_DIR.mkdir(exist_ok=True)

# ===== 测试用例 =====
# 每条包含: 问题 + 期望关键词(用于自动判断答案质量)
TEST_CASES = [
    # ===== 精确查找类 =====
    {
        "query": "Vibe Coding 的总体成功率是多少？",
        "expect_keywords": ["89.8%", "90%", "成功率"],
        "category": "精确查找",
    },
    {
        "query": "张小龙认为好的工具应该是什么样的？",
        "expect_keywords": ["简单高效", "用完即走"],
        "category": "精确查找",
    },
    {
        "query": "Karpathy 的知识库有哪三层架构？",
        "expect_keywords": ["只读区", "写作区", "规则区"],
        "category": "精确查找",
    },
    {
        "query": "Vibe Coding 有哪三种失败原型？",
        "expect_keywords": ["沙箱墙", "上下文窗口", "安全过滤器"],
        "category": "精确查找",
    },
    # ===== 细节提取类 =====
    {
        "query": "作者用 Cursor 和 Claude Code 完成了哪三个项目？",
        "expect_keywords": ["AgenticaSoC", "Blinds", "up-cli"],
        "category": "细节提取",
    },
    {
        "query": "Vibe Coding 的七宗罪中，第一宗罪是什么？",
        "expect_keywords": ["宣布胜利", "Quick Win", "永远修"],
        "category": "细节提取",
    },
    {
        "query": "张小龙认为社交产品的定位是什么？",
        "expect_keywords": ["孤独", "去处", "消除孤独"],
        "category": "细节提取",
    },
    # ===== 主题概览类 =====
    {
        "query": "AI 时代产品经理需要哪些核心能力转变？",
        "expect_keywords": ["数据", "技术", "产品"],
        "category": "主题概览",
    },
    {
        "query": "如何搭建一个 AI 驱动的个人知识库？",
        "expect_keywords": ["RAG", "知识库", "检索"],
        "category": "主题概览",
    },
    # ===== 跨文档关联类 =====
    {
        "query": "准备 AI 产品经理面试需要了解哪些内容？",
        "expect_keywords": ["大模型", "面试", "AI"],
        "category": "跨文档关联",
    },
    {
        "query": "好的产品设计应该遵循哪些原则？",
        "expect_keywords": ["简单", "用户", "效率"],
        "category": "跨文档关联",
    },
    # ===== 否定测试类 =====
    {
        "query": "如何做一道好吃的红烧肉？",
        "expect_keywords": ["知识库中暂未找到", "暂未找到"],
        "category": "否定测试",
    },
    {
        "query": "React Hooks 的使用方法是什么？",
        "expect_keywords": ["知识库中暂未找到", "暂未找到"],
        "category": "否定测试",
    },
    # ===== 模糊查询类 =====
    {
        "query": "之前那篇讲 AI 编程失败经验的文章提到了哪些问题？",
        "expect_keywords": ["沙箱墙", "上下文", "失败"],
        "category": "模糊查询",
    },
    # ===== 补充测试 =====
    {
        "query": "张小龙对产品商业化有什么看法？",
        "expect_keywords": ["商业化", "自然", "不急"],
        "category": "商业化",
    },
    {
        "query": "AI 产品成功的关键是什么？",
        "expect_keywords": ["数据", "决策", "产品经理"],
        "category": "AI产品",
    },
    {
        "query": "Loopit 是如何在短时间内登顶美区娱乐榜首的？",
        "expect_keywords": ["Loopit", "登顶", "AI社区"],
        "category": "AI产品",
    },
]


TIMEOUT = 180  # DeepSeek API 较慢，给足时间


def _run_single(tc: dict, idx: int, total: int) -> dict:
    """执行单条测试用例"""
    query = tc["query"]
    print(f"  [{idx}/{total}] {query}...", end=" ", flush=True)
    t0 = time.time()

    try:
        r = requests.post(
            f"{BASE_URL}/search",
            json={"query": query, "top_k": 3},
            timeout=TIMEOUT,
        )
        elapsed = round(time.time() - t0, 2)

        if r.status_code == 200:
            data = r.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            debug = data.get("debug", {})

            hit_keywords = [kw for kw in tc["expect_keywords"] if kw in answer]
            keyword_score = len(hit_keywords) / len(tc["expect_keywords"]) if tc["expect_keywords"] else 0

            result = {
                "query": query,
                "category": tc["category"],
                "answer": answer,
                "answer_len": len(answer),
                "sources": [s.get("title", "") for s in sources],
                "source_count": len(sources),
                "rewritten_queries": debug.get("rewritten_queries", []),
                "wiki_candidates": debug.get("wiki_candidates", 0),
                "data_candidates": debug.get("data_candidates", 0),
                "keyword_score": round(keyword_score, 2),
                "hit_keywords": hit_keywords,
                "miss_keywords": [kw for kw in tc["expect_keywords"] if kw not in answer],
                "latency_s": elapsed,
                "error": None,
            }
            print(f"✅ {elapsed}s, 关键词 {len(hit_keywords)}/{len(tc['expect_keywords'])}")
        else:
            result = {
                "query": query, "category": tc["category"],
                "error": f"HTTP {r.status_code}: {r.text[:200]}",
                "answer": "", "answer_len": 0, "sources": [],
                "source_count": 0, "keyword_score": 0,
                "hit_keywords": [], "miss_keywords": tc["expect_keywords"],
                "latency_s": elapsed, "rewritten_queries": [],
                "wiki_candidates": 0, "data_candidates": 0,
            }
            print(f"❌ HTTP {r.status_code}")

    except Exception as e:
        result = {
            "query": query, "category": tc["category"],
            "error": str(e), "answer": "", "answer_len": 0,
            "sources": [], "source_count": 0, "keyword_score": 0,
            "hit_keywords": [], "miss_keywords": tc["expect_keywords"],
            "latency_s": 0, "rewritten_queries": [],
            "wiki_candidates": 0, "data_candidates": 0,
        }
        print(f"❌ {e}")

    return result


def run_eval() -> list:
    """对所有测试用例执行搜索，先跑第 1 条验证成功再跑剩余"""
    total = len(TEST_CASES)

    # 先跑第 1 条探测
    print("  ⏳ 先跑第 1 条验证服务可用性...\n")
    first = _run_single(TEST_CASES[0], 1, total)
    if first.get("error"):
        print(f"\n  🛑 第 1 条失败（{first['error'][:80]}），中止评测。请检查服务状态。")
        return [first]

    print(f"\n  ✅ 第 1 条成功（{first['latency_s']}s），继续跑剩余 {total - 1} 条...\n")
    results = [first]

    for i, tc in enumerate(TEST_CASES[1:], 2):
        result = _run_single(tc, i, total)
        results.append(result)

    return results


def save_results(tag: str):
    """运行评测并保存结果"""
    print(f"\n🔍 运行 RAG 评测 [{tag}]...\n")
    results = run_eval()

    # 汇总统计
    avg_keyword = sum(r["keyword_score"] for r in results) / len(results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    avg_answer_len = sum(r["answer_len"] for r in results) / len(results)
    avg_sources = sum(r["source_count"] for r in results) / len(results)

    report = {
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_cases": len(results),
            "avg_keyword_score": round(avg_keyword, 3),
            "avg_latency_s": round(avg_latency, 2),
            "avg_answer_len": round(avg_answer_len),
            "avg_source_count": round(avg_sources, 1),
            "errors": sum(1 for r in results if r["error"]),
        },
        "results": results,
    }

    filepath = EVAL_DIR / f"{tag}.json"
    filepath.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"📊 评测结果 [{tag}]")
    print(f"{'='*60}")
    print(f"  关键词命中率:  {avg_keyword:.1%}")
    print(f"  平均延迟:      {avg_latency:.2f}s")
    print(f"  平均答案长度:  {avg_answer_len:.0f} 字")
    print(f"  平均来源数:    {avg_sources:.1f}")
    print(f"  错误数:        {report['summary']['errors']}")
    print(f"\n  已保存到: {filepath}")


def diff_results():
    """对比 before 和 after"""
    before_path = EVAL_DIR / "before.json"
    after_path = EVAL_DIR / "after.json"

    if not before_path.exists():
        print("❌ 未找到 before.json，请先运行: python eval_rag.py save before")
        return
    if not after_path.exists():
        print("❌ 未找到 after.json，请先运行: python eval_rag.py save after")
        return

    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))

    bs = before["summary"]
    af = after["summary"]

    def arrow(old, new, higher_better=True):
        if new > old:
            return f"{'🟢' if higher_better else '🔴'} {old} → {new} (+{new-old:.3f})" if isinstance(new, float) else f"{'🟢' if higher_better else '🔴'} {old} → {new}"
        elif new < old:
            return f"{'🔴' if higher_better else '🟢'} {old} → {new} ({new-old:.3f})" if isinstance(new, float) else f"{'🔴' if higher_better else '🟢'} {old} → {new}"
        return f"⚪ {old} (无变化)"

    print(f"\n{'='*60}")
    print(f"📊 RAG 效果对比: {before['tag']} vs {after['tag']}")
    print(f"   {before['timestamp'][:16]} vs {after['timestamp'][:16]}")
    print(f"{'='*60}")
    print(f"  关键词命中率:  {arrow(bs['avg_keyword_score'], af['avg_keyword_score'])}")
    print(f"  平均延迟:      {arrow(bs['avg_latency_s'], af['avg_latency_s'], higher_better=False)}")
    print(f"  平均答案长度:  {arrow(bs['avg_answer_len'], af['avg_answer_len'])}")
    print(f"  平均来源数:    {arrow(bs['avg_source_count'], af['avg_source_count'])}")
    print(f"  错误数:        {arrow(bs['errors'], af['errors'], higher_better=False)}")

    # 逐题对比
    print(f"\n{'─'*60}")
    print(f"  逐题对比:")
    print(f"{'─'*60}")

    for i, (br, ar) in enumerate(zip(before["results"], after["results"])):
        query = br["query"]
        bk = br["keyword_score"]
        ak = ar["keyword_score"]
        bl = br["latency_s"]
        al = ar["latency_s"]
        ba = br["answer_len"]
        aa = ar["answer_len"]

        status = "🟢" if ak > bk else ("🔴" if ak < bk else "⚪")
        print(f"\n  {status} Q{i+1}: {query}")
        print(f"     关键词: {bk:.0%} → {ak:.0%}  |  长度: {ba} → {aa}  |  延迟: {bl}s → {al}s")

        if br["miss_keywords"] != ar["miss_keywords"]:
            if br["miss_keywords"] and not ar["miss_keywords"]:
                print(f"     ✅ 修复: 之前缺失 {br['miss_keywords']}")
            elif ar["miss_keywords"]:
                print(f"     缺失关键词: {ar['miss_keywords']}")

        # 答案变化（取前100字对比）
        b_short = br["answer"][:100].replace("\n", " ")
        a_short = ar["answer"][:100].replace("\n", " ")
        if b_short != a_short:
            print(f"     Before: {b_short}...")
            print(f"     After:  {a_short}...")

    print(f"\n{'='*60}")


def show_last(tag: str = None):
    """查看某次评测的详细结果"""
    if tag:
        filepath = EVAL_DIR / f"{tag}.json"
    else:
        files = sorted(EVAL_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            print("❌ 没有评测结果")
            return
        filepath = files[0]

    if not filepath.exists():
        print(f"❌ 未找到 {filepath}")
        return

    data = json.loads(filepath.read_text(encoding="utf-8"))
    print(f"\n📊 评测详情 [{data['tag']}] — {data['timestamp'][:16]}")
    print(f"{'='*60}")

    for i, r in enumerate(data["results"], 1):
        status = "✅" if r["keyword_score"] >= 0.5 else "⚠️" if r["keyword_score"] > 0 else "❌"
        print(f"\n{status} Q{i} [{r['category']}]: {r['query']}")
        print(f"   关键词: {r['keyword_score']:.0%} (命中: {r['hit_keywords']}, 缺失: {r['miss_keywords']})")
        print(f"   来源: {r['sources']}")
        print(f"   改写: {r['rewritten_queries']}")
        print(f"   Wiki: {r['wiki_candidates']} | Data: {r['data_candidates']} | 延迟: {r['latency_s']}s")
        print(f"   答案: {r['answer'][:200]}...")


# ===== CLI =====
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
RAG 评测工具 — 用法:

  python eval_rag.py save <tag>    运行评测并保存 (tag=before/after/v1/...)
  python eval_rag.py diff          对比 before vs after
  python eval_rag.py show [tag]    查看某次评测详情
  python eval_rag.py run           只运行不保存，快速看结果

示例流程:
  1. python eval_rag.py save before     # 修改前基线
  2. (修改代码，重启服务)
  3. python eval_rag.py save after      # 修改后
  4. python eval_rag.py diff            # 对比
        """)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "save":
        tag = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime("%Y%m%d_%H%M")
        save_results(tag)
    elif cmd == "diff":
        diff_results()
    elif cmd == "show":
        tag = sys.argv[2] if len(sys.argv) > 2 else None
        show_last(tag)
    elif cmd == "run":
        results = run_eval()
        avg_kw = sum(r["keyword_score"] for r in results) / len(results)
        print(f"\n关键词命中率: {avg_kw:.1%}")
    else:
        print(f"未知命令: {cmd}")
