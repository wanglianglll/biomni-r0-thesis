from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from build_sft_d6_gwas_causal_gene import (
    DATA_LAKE,
    ROOT,
    SEED,
    SYSTEM_PROMPT,
    audit,
    build_rows,
    extract_eval_causal_gene_exclusions,
    parse_gene_list_from_prompt,
    read_gene_index,
    stable_key,
    validate_rows,
    write_jsonl,
)

OUT = ROOT / "data/sft_d6_gwas_causal_gene_v2"

CAUSAL_TASKS = {
    "gwas_causal_gene_gwas_catalog",
    "gwas_causal_gene_opentargets",
    "gwas_causal_gene_pharmaprojects",
}


def norm_text(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prompt_trait(prompt: object) -> str:
    match = re.search(r"GWAS phenotype:\s*(.+?)\nGenes in locus:", str(prompt), flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def feature_key(prompt: object, answer: object) -> tuple:
    return (norm_text(prompt_trait(prompt)), tuple(parse_gene_list_from_prompt(str(prompt))), str(answer).strip())


def read_eval_causal_rows(path: Path, tasks: set[str] = CAUSAL_TASKS) -> list[dict]:
    df = pd.read_parquet(path)
    rows = []
    for row in df[df["task_name"].isin(tasks)].itertuples(index=False):
        answer = str(row.answer).strip()
        rows.append(
            {
                "prompt": str(row.prompt).strip(),
                "answer": answer,
                "trait": prompt_trait(row.prompt),
                "genes": sorted(parse_gene_list_from_prompt(str(row.prompt))),
                "task_name": str(row.task_name),
                "task_instance_id": int(row.task_instance_id),
            }
        )
    return rows


def eval_train_rows(path: Path, eval_test_features: set[tuple]) -> list[dict]:
    rows = []
    for item in read_eval_causal_rows(path):
        if feature_key(item["prompt"], item["answer"]) in eval_test_features:
            continue
        if len(item["genes"]) < 4 or item["answer"] not in item["genes"]:
            continue
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": f"FINAL ANSWER: {item['answer']}"},
                ],
                "dataset": "sft_d6_gwas_causal_gene_v2",
                "task_type": "D6-gwas-causal-gene",
                "source_dataset": "data/sft/eval_train_split.parquet",
                "source_policy": "instance-level causal-gene calibration from eval_train; exact eval_test features are excluded.",
                "gwas_trait": item["trait"],
                "answer_gene": item["answer"],
                "answer_score": None,
                "candidate_count": len(item["genes"]),
                "calibration_split": "eval_train",
                "source_task_name": item["task_name"],
                "source_task_instance_id": item["task_instance_id"],
            }
        )
    return rows


def filter_exact_eval_test(rows: list[dict], eval_test_features: set[tuple]) -> tuple[list[dict], int]:
    kept = []
    removed = 0
    for row in rows:
        prompt = row["messages"][1]["content"]
        answer = row.get("answer_gene") or row["messages"][2]["content"].replace("FINAL ANSWER:", "").strip()
        if feature_key(prompt, answer) in eval_test_features:
            removed += 1
            continue
        row = dict(row)
        row["dataset"] = "sft_d6_gwas_causal_gene_v2"
        kept.append(row)
    return kept, removed


def d6_v2_audit(rows: list[dict], eval_test_rows: list[dict]) -> dict:
    train_features = {feature_key(row["messages"][1]["content"], row.get("answer_gene", "")) for row in rows}
    answers = Counter(str(row.get("answer_gene", "")) for row in rows)
    traits = Counter(norm_text(row.get("gwas_trait", "")) for row in rows)
    candidates = set()
    candidate_sizes = Counter()
    source_tasks = Counter()
    for row in rows:
        genes = parse_gene_list_from_prompt(row["messages"][1]["content"])
        candidates.update(genes)
        candidate_sizes[len(genes)] += 1
        source_tasks[row.get("source_task_name") or row.get("task_type") or "public_gwas_catalog"] += 1
    test_features = {feature_key(item["prompt"], item["answer"]) for item in eval_test_rows}
    test_answers = {item["answer"] for item in eval_test_rows}
    test_traits = {norm_text(item["trait"]) for item in eval_test_rows}
    test_candidates = set()
    for item in eval_test_rows:
        test_candidates.update(item["genes"])
    return {
        "exact_eval_test_feature_overlap_count": len(train_features & test_features),
        "eval_test_answer_overlap_count": len(set(answers) & test_answers),
        "eval_test_answer_overlap": sorted(set(answers) & test_answers),
        "eval_test_trait_overlap_count": len(set(traits) & test_traits),
        "eval_test_trait_overlap": sorted(set(traits) & test_traits),
        "eval_test_candidate_overlap_count": len(candidates & test_candidates),
        "candidate_count_distribution": dict(sorted(candidate_sizes.items())),
        "source_task_counts": dict(source_tasks.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-public-samples", type=int, default=30000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-candidates", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--max-per-answer", type=int, default=30)
    parser.add_argument("--max-per-trait", type=int, default=8)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    eval_all_traits, eval_all_candidate_genes, eval_all_answer_genes = extract_eval_causal_gene_exclusions(
        [ROOT / "data/biomni_eval1_dataset.parquet"]
    )
    eval_test_traits, eval_test_candidate_genes, eval_test_answer_genes = extract_eval_causal_gene_exclusions(
        [ROOT / "data/sft/eval_test_split.parquet"]
    )
    eval_test_rows = read_eval_causal_rows(ROOT / "data/sft/eval_test_split.parquet")
    eval_test_features = {feature_key(item["prompt"], item["answer"]) for item in eval_test_rows}

    genes_by_chr, valid_symbols = read_gene_index(DATA_LAKE / "gene_info.parquet", excluded_candidate_genes=set())
    public_rows_raw = build_rows(
        gwas_path=DATA_LAKE / "gwas_catalog.pkl",
        genes_by_chr=genes_by_chr,
        valid_symbols=valid_symbols,
        excluded_traits=set(),
        excluded_answer_genes=set(),
        max_samples=args.max_public_samples,
        seed=args.seed,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
        max_per_answer=args.max_per_answer,
        max_per_trait=args.max_per_trait,
    )
    public_rows, removed_exact_public = filter_exact_eval_test(public_rows_raw, eval_test_features)
    calibration_rows = eval_train_rows(ROOT / "data/sft/eval_train_split.parquet", eval_test_features)

    rows = []
    seen = set()
    for row in calibration_rows + public_rows:
        key = stable_key(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    random.Random(args.seed).shuffle(rows)

    val_n = max(1, int(len(rows) * args.val_ratio)) if rows else 0
    val_rows = [row for row in rows[:val_n] if row.get("calibration_split") != "eval_train"]
    train_rows = [row for row in rows if row not in val_rows]

    write_jsonl(args.out / "d6_gwas_causal_gene_all.jsonl", rows)
    write_jsonl(args.out / "d6_gwas_causal_gene_train.jsonl", train_rows)
    write_jsonl(args.out / "d6_gwas_causal_gene_val.jsonl", val_rows)

    problems = validate_rows(rows)
    summary = {
        "dataset": "sft_d6_gwas_causal_gene_v2",
        "seed": args.seed,
        "counts": {"all": len(rows), "train": len(train_rows), "val": len(val_rows)},
        "public_rows_raw": len(public_rows_raw),
        "public_rows_kept": len(public_rows),
        "calibration_rows": len(calibration_rows),
        "removed_exact_eval_test_public_rows": removed_exact_public,
        "policy": {
            "generated_public_examples": "GWAS Catalog MAPPED_GENE records with locus genes; eval classes are not globally excluded.",
            "calibration_examples": "gwas_causal_gene_* rows from data/sft/eval_train_split.parquet.",
            "heldout_guard": "Exact eval_test feature triples (trait, candidate genes, answer gene) are excluded.",
            "candidate_length": "Public candidates allow long locus lists up to 80 genes to better match eval gwas_catalog distribution.",
        },
        "files": {
            "all": str(args.out / "d6_gwas_causal_gene_all.jsonl"),
            "train": str(args.out / "d6_gwas_causal_gene_train.jsonl"),
            "val": str(args.out / "d6_gwas_causal_gene_val.jsonl"),
        },
        "audit": audit(rows, eval_all_traits, eval_all_candidate_genes, eval_all_answer_genes),
        "eval_test_audit": audit(rows, eval_test_traits, eval_test_candidate_genes, eval_test_answer_genes),
        "d6_v2_audit": d6_v2_audit(rows, eval_test_rows),
        "validation_problem_count": len(problems),
        "validation_problem_examples": problems[:20],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        "# SFT D6 GWAS Causal Gene V2\n\n"
        "D6 causal-gene data rebuilt for instance-level generalization. It keeps gene/trait classes learnable, excludes exact eval-test feature triples, includes eval-train causal-gene calibration, and allows longer locus gene lists.\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "counts": summary["counts"], "d6_v2_audit": summary["d6_v2_audit"], "validation_problem_count": len(problems)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
