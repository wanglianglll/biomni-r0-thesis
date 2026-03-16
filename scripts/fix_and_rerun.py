"""
纯离线修复：不调 API，只从第一次跑的 checkpoint 重新提取答案
"""
import json
import os
import re
import glob
from datetime import datetime

OUTPUT_DIR = "../results/baseline"


def extract_answer_v2(raw: str, task_name: str) -> str:
    """修复版答案提取"""
    if not raw:
        return ""

    # 先提取 FINAL ANSWER: xxx
    match = re.search(r'FINAL ANSWER:\s*(.+)', raw, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip().strip(".")
    else:
        extracted = raw.strip()

    if task_name == "crispr_delivery":
        m = re.search(r'\b([a-fA-F])\b', extracted)
        return m.group(1).lower() if m else extracted[:1].lower()

    elif task_name in ("lab_bench_dbqa", "lab_bench_seqqa"):
        # 1. FINAL ANSWER 后面的字母
        m = re.search(r'^([A-Ea-e])\b', extracted)
        if m:
            return m.group(1).upper()
        if len(extracted) == 1 and extracted.isalpha():
            return extracted.upper()
        # 2. 从整个 raw 里找各种模式
        for pattern in [
            r'(?:answer|Answer|ANSWER)\s*(?:is|:)\s*\(?([A-Ea-e])\)?',
            r'(?:option|Option)\s*\(?([A-Ea-e])\)?',
            r'\*\*([A-Ea-e])\*\*',                  # **A** 加粗格式
            r'(?:correct.*?)([A-Ea-e])\b',
            r'\(([A-Ea-e])\)\s*$',
        ]:
            m = re.search(pattern, raw)
            if m:
                return m.group(1).upper()
        # 3. 兜底：raw 里最后一个单独的 A-E
        letters = re.findall(r'\b([A-Ea-e])\b', raw)
        if letters:
            return letters[-1].upper()
        return ""

    elif task_name.startswith("gwas_causal_gene") or task_name == "screen_gene_retrieval":
        m = re.search(r'\b([A-Z][A-Z0-9]{1,15})\b', extracted)
        return m.group(1) if m else extracted.upper().strip()

    elif task_name == "gwas_variant_prioritization":
        m = re.search(r'(rs\d+)', extracted)
        return m.group(1) if m else extracted.strip()

    elif task_name == "rare_disease_diagnosis":
        m = re.search(r'(\d{6})', extracted)
        return json.dumps({"OMIM_ID": m.group(1)}) if m else extracted

    elif task_name == "patient_gene_detection":
        # 从整个 raw 里找 ENSG ID
        ensg_ids = re.findall(r'(ENSG\d{11})', raw)
        if ensg_ids:
            return json.dumps({"causal_gene": ensg_ids[:5]})
        # 普通基因名
        try:
            parsed = json.loads(extracted)
            if "causal_gene" in parsed:
                return extracted
        except Exception:
            pass
        skip = {"FINAL", "ANSWER", "THE", "AND", "FOR", "THIS", "THAT", "WITH",
                "FROM", "GENE", "PATIENT", "CAUSAL", "JSON", "BASED", "MOST", "ID"}
        genes = re.findall(r'\b([A-Z][A-Z0-9]{1,15})\b', extracted)
        genes = [g for g in genes if g not in skip]
        if genes:
            return json.dumps({"causal_gene": genes[:5]})
        return ""

    return extracted


def compute_score(task_name: str, user_answer: str, ground_truth: str) -> float:
    try:
        if task_name == "crispr_delivery":
            return 1.0 if user_answer.strip().lower() == ground_truth.strip().lower() else 0.0
        elif task_name.startswith("gwas_causal_gene"):
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0
        elif task_name == "gwas_variant_prioritization":
            return 1.0 if user_answer.strip() == ground_truth.strip() else 0.0
        elif task_name in ("lab_bench_dbqa", "lab_bench_seqqa"):
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0
        elif task_name == "screen_gene_retrieval":
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0
        elif task_name == "rare_disease_diagnosis":
            user_dict = json.loads(user_answer) if isinstance(user_answer, str) else user_answer
            gt_dict = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
            return 1.0 if user_dict.get("OMIM_ID") == gt_dict.get("OMIM_ID") else 0.0
        elif task_name == "patient_gene_detection":
            user_dict = json.loads(user_answer) if isinstance(user_answer, str) else user_answer
            predicted = user_dict.get("causal_gene", [])
            if not isinstance(predicted, list):
                predicted = [predicted]
            true_genes = [g.strip() for g in ground_truth.split(",")] if "," in ground_truth else [ground_truth]
            return 1.0 if predicted and set(true_genes) & set(predicted) else 0.0
        else:
            return 1.0 if user_answer.strip() == ground_truth.strip() else 0.0
    except Exception:
        return 0.0


def main():
    print("=" * 60)
    print("Offline Fix: re-extract answers from existing checkpoints")
    print("NO API calls needed!")
    print("=" * 60)

    # 找到所有原始 checkpoint（排除 _v2 文件）
    all_tasks = [
        "crispr_delivery",
        "gwas_causal_gene_gwas_catalog",
        "gwas_causal_gene_opentargets",
        "gwas_causal_gene_pharmaprojects",
        "gwas_variant_prioritization",
        "lab_bench_dbqa",
        "lab_bench_seqqa",
        "patient_gene_detection",
        "rare_disease_diagnosis",
        "screen_gene_retrieval",
    ]

    results_summary = {}
    total_correct_old = 0
    total_correct_new = 0
    total_count = 0

    for task_name in all_tasks:
        ckpt_path = f"{OUTPUT_DIR}/_ckpt_{task_name}.json"
        if not os.path.exists(ckpt_path):
            print(f"\n  {task_name}: checkpoint not found, skipping")
            continue

        with open(ckpt_path) as f:
            data = json.load(f)

        old_correct = sum(1 for d in data if d["score"] >= 1.0)
        new_correct = 0
        fixed_up = 0    # 提取修复后答对的
        fixed_down = 0  # 修复后反而答错的（不太可能）
        examples = []

        for d in data:
            raw = d.get("raw_preview", "")
            old_extracted = d["extracted"]
            old_score = d["score"]

            new_extracted = extract_answer_v2(raw, task_name)
            new_score = compute_score(task_name, new_extracted, d["ground_truth"])

            new_correct += int(new_score >= 1.0)

            if new_score > old_score:
                fixed_up += 1
                if len(examples) < 3:
                    examples.append({
                        "id": d["task_instance_id"],
                        "old": old_extracted,
                        "new": new_extracted,
                        "truth": d["ground_truth"][:30],
                    })
            elif new_score < old_score:
                fixed_down += 1

        n = len(data)
        old_acc = old_correct / n
        new_acc = new_correct / n
        change = new_acc - old_acc

        total_correct_old += old_correct
        total_correct_new += new_correct
        total_count += n

        arrow = "⬆️ " if change > 0.001 else ("⬇️ " if change < -0.001 else "   ")
        print(f"\n  {task_name}:")
        print(f"    OLD: {old_correct}/{n} = {old_acc:.3f}")
        print(f"    NEW: {new_correct}/{n} = {new_acc:.3f}  {arrow}({change:+.3f})")
        if fixed_up > 0:
            print(f"    Fixed ⬆️: {fixed_up} instances")
            for ex in examples:
                print(f"      ID={ex['id']}: '{ex['old']}' → '{ex['new']}' (truth={ex['truth']})")

        results_summary[task_name] = {
            "accuracy_old": round(old_acc, 4),
            "accuracy_new": round(new_acc, 4),
            "correct_old": old_correct,
            "correct_new": new_correct,
            "total": n,
            "fixed_up": fixed_up,
        }

    # 最终汇总
    old_overall = total_correct_old / total_count if total_count else 0
    new_overall = total_correct_new / total_count if total_count else 0

    print(f"\n\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Task':<42} {'OLD':>7} {'NEW':>7} {'Δ':>7}")
    print("-" * 65)
    for task_name in all_tasks:
        if task_name in results_summary:
            s = results_summary[task_name]
            d = s["accuracy_new"] - s["accuracy_old"]
            arrow = "⬆️" if d > 0.001 else ("⬇️" if d < -0.001 else "  ")
            print(f"{task_name:<42} {s['accuracy_old']:>6.3f} {s['accuracy_new']:>6.3f} {d:>+6.3f} {arrow}")
    print("-" * 65)
    d = new_overall - old_overall
    print(f"{'OVERALL':<42} {old_overall:>6.3f} {new_overall:>6.3f} {d:>+6.3f}")

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{OUTPUT_DIR}/summary_fixed_{timestamp}.json", "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {OUTPUT_DIR}/summary_fixed_{timestamp}.json")

    # 给出结论
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    print(f"raw_preview 只保存了前 300 字符，")
    print(f"如果模型的 FINAL ANSWER 在 300 字符之后，离线修复无法恢复。")
    print(f"")
    print(f"需要重新调 API 的任务（等网络恢复后）:")
    for task_name in ["lab_bench_seqqa", "patient_gene_detection"]:
        s = results_summary.get(task_name, {})
        if s.get("accuracy_new", 0) < 0.1:
            print(f"  ⚠️  {task_name}: {s.get('accuracy_new', 0):.3f} — 需要重跑")
        else:
            print(f"  ✅  {task_name}: {s.get('accuracy_new', 0):.3f} — 已修复")


if __name__ == "__main__":
    main()