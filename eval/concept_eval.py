"""概念/实体抽取质量评测

对比 Wiki 编译引擎抽取的概念和实体 vs Ground Truth 标注，
计算 Precision / Recall / F1。

用法：
    python -m eval.concept_eval [--ground-truth eval/concept_ground_truth.yaml]
"""

import re
import yaml
import logging
from pathlib import Path
from typing import List, Set, Tuple
from collections import defaultdict

from config import WIKI_DIR

logger = logging.getLogger(__name__)


def load_ground_truth(path: str = "eval/concept_ground_truth.yaml") -> dict:
    """加载人工标注的 Ground Truth"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_system_concepts() -> dict:
    """从 Wiki 目录中收集系统抽取的概念和实体"""
    concepts = set()
    entities = {"persons": set(), "products": set(), "companies": set()}

    # 概念 = topics 目录下的页面标题 + concepts 目录（如果存在）
    for subdir in ["topics", "concepts"]:
        dir_path = WIKI_DIR / subdir
        if not dir_path.exists():
            continue
        for md_file in dir_path.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            # 从 frontmatter 提取 title
            title_match = re.search(r"^title:\s*(.+)$", content, re.MULTILINE)
            if title_match:
                concepts.add(title_match.group(1).strip().strip("'\""))

    # 实体 = entities 目录下的页面
    entities_dir = WIKI_DIR / "entities"
    if entities_dir.exists():
        for md_file in entities_dir.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            title_match = re.search(r"^title:\s*(.+)$", content, re.MULTILINE)
            etype_match = re.search(r"^entity_type:\s*(.+)$", content, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip().strip("'\"")
                etype = etype_match.group(1).strip() if etype_match else "other"
                if etype == "person":
                    entities["persons"].add(title)
                elif etype == "product":
                    entities["products"].add(title)
                elif etype == "company":
                    entities["companies"].add(title)
                else:
                    # 猜测类型
                    entities["products"].add(title)

    return {"concepts": concepts, "entities": entities}


def fuzzy_match(a: str, b: str) -> bool:
    """模糊匹配两个概念名（中英文、大小写、空格不敏感）"""
    def normalize(s):
        s = s.lower().strip()
        s = re.sub(r"[\s_\-·]", "", s)
        return s

    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    # 子串匹配（一方包含另一方）
    if len(na) >= 3 and len(nb) >= 3:
        if na in nb or nb in na:
            return True
    return False


def match_sets(ground_truth: Set[str], system_output: Set[str]) -> Tuple[int, int, int]:
    """计算匹配数、GT总数、系统总数"""
    matched = 0
    gt_matched = set()
    sys_matched = set()

    for gt_item in ground_truth:
        for sys_item in system_output:
            if sys_item not in sys_matched and fuzzy_match(gt_item, sys_item):
                matched += 1
                gt_matched.add(gt_item)
                sys_matched.add(sys_item)
                break

    return matched, len(ground_truth), len(system_output)


def evaluate(gt_path: str = "eval/concept_ground_truth.yaml") -> dict:
    """运行完整评测"""
    gt = load_ground_truth(gt_path)
    system = load_system_concepts()

    # ===== 1. 概念评测 =====
    # 收集 GT 中所有概念（去重）
    gt_all_concepts = set()
    for article in gt.get("articles", []):
        for c in article.get("concepts", []):
            gt_all_concepts.add(c)

    sys_concepts = system["concepts"]
    c_matched, c_gt_total, c_sys_total = match_sets(gt_all_concepts, sys_concepts)

    concept_precision = c_matched / c_sys_total if c_sys_total > 0 else 0
    concept_recall = c_matched / c_gt_total if c_gt_total > 0 else 0
    concept_f1 = (2 * concept_precision * concept_recall / (concept_precision + concept_recall)
                  if (concept_precision + concept_recall) > 0 else 0)

    # ===== 2. 实体评测 =====
    gt_entities = gt.get("entities_global", {})
    entity_results = {}
    total_e_matched = 0
    total_e_gt = 0
    total_e_sys = 0

    for etype in ["persons", "products", "companies"]:
        gt_set = set(gt_entities.get(etype, []))
        sys_set = system["entities"].get(etype, set())
        e_matched, e_gt, e_sys = match_sets(gt_set, sys_set)
        total_e_matched += e_matched
        total_e_gt += e_gt
        total_e_sys += e_sys
        entity_results[etype] = {
            "gt_count": e_gt,
            "sys_count": e_sys,
            "matched": e_matched,
            "precision": e_matched / e_sys if e_sys > 0 else 0,
            "recall": e_matched / e_gt if e_gt > 0 else 0,
        }

    entity_precision = total_e_matched / total_e_sys if total_e_sys > 0 else 0
    entity_recall = total_e_matched / total_e_gt if total_e_gt > 0 else 0
    entity_f1 = (2 * entity_precision * entity_recall / (entity_precision + entity_recall)
                 if (entity_precision + entity_recall) > 0 else 0)

    # ===== 3. 分类评测（LLM-as-Judge）=====
    # 读取 taxonomy 分类结果
    taxonomy_path = WIKI_DIR / "_taxonomy.yaml"
    category_count = 0
    uncategorized = 0
    total_pages = 0
    if taxonomy_path.exists():
        tax = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
        cats = tax.get("categories", {}) if tax else {}
        category_count = len(cats)
        for cat_data in cats.values():
            if isinstance(cat_data, dict):
                total_pages += len(cat_data.get("pages", []))
    # 计算覆盖率
    all_pages = list((WIKI_DIR / "topics").glob("*.md")) + list((WIKI_DIR / "entities").glob("*.md"))
    actual_total = len(all_pages)
    coverage = total_pages / actual_total if actual_total > 0 else 0

    results = {
        "concepts": {
            "gt_count": c_gt_total,
            "sys_count": c_sys_total,
            "matched": c_matched,
            "precision": round(concept_precision, 3),
            "recall": round(concept_recall, 3),
            "f1": round(concept_f1, 3),
        },
        "entities": {
            "total_gt": total_e_gt,
            "total_sys": total_e_sys,
            "total_matched": total_e_matched,
            "precision": round(entity_precision, 3),
            "recall": round(entity_recall, 3),
            "f1": round(entity_f1, 3),
            "by_type": entity_results,
        },
        "taxonomy": {
            "category_count": category_count,
            "total_pages_classified": total_pages,
            "actual_total_pages": actual_total,
            "coverage": round(coverage, 3),
        },
    }

    return results


def print_report(results: dict):
    """打印评测报告"""
    print("\n" + "=" * 60)
    print("  概念/实体抽取质量评测报告")
    print("=" * 60)

    c = results["concepts"]
    print(f"\n  === 概念抽取 ===")
    print(f"  Ground Truth: {c['gt_count']} 个概念")
    print(f"  系统抽取:     {c['sys_count']} 个概念")
    print(f"  匹配数:       {c['matched']}")
    print(f"  Precision:    {c['precision']:.1%}")
    print(f"  Recall:       {c['recall']:.1%}")
    print(f"  F1:           {c['f1']:.1%}")

    e = results["entities"]
    print(f"\n  === 实体识别 ===")
    print(f"  Ground Truth: {e['total_gt']} 个实体")
    print(f"  系统识别:     {e['total_sys']} 个实体")
    print(f"  匹配数:       {e['total_matched']}")
    print(f"  Precision:    {e['precision']:.1%}")
    print(f"  Recall:       {e['recall']:.1%}")
    print(f"  F1:           {e['f1']:.1%}")

    for etype, data in e["by_type"].items():
        label = {"persons": "人物", "products": "产品", "companies": "公司"}.get(etype, etype)
        print(f"    {label}: GT={data['gt_count']}, 系统={data['sys_count']}, "
              f"匹配={data['matched']}, P={data['precision']:.0%}, R={data['recall']:.0%}")

    t = results["taxonomy"]
    print(f"\n  === 分类覆盖 ===")
    print(f"  分类数:       {t['category_count']}")
    print(f"  已分类页面:   {t['total_pages_classified']}/{t['actual_total_pages']}")
    print(f"  覆盖率:       {t['coverage']:.1%}")

    # 达标判断
    print(f"\n  === 达标检查 ===")
    checks = [
        ("概念 Precision ≥ 80%", c["precision"] >= 0.80),
        ("概念 Recall ≥ 70%", c["recall"] >= 0.70),
        ("实体 Precision ≥ 85%", e["precision"] >= 0.85),
        ("实体 Recall ≥ 70%", e["recall"] >= 0.70),
        ("分类覆盖率 = 100%", t["coverage"] >= 0.99),
    ]
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

    passed_count = sum(1 for _, p in checks if p)
    print(f"\n  达标: {passed_count}/{len(checks)}")
    print()


if __name__ == "__main__":
    results = evaluate()
    print_report(results)
