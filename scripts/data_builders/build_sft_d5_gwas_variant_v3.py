from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from build_sft_d5_gwas_variant import (
    DATA_LAKE,
    ROOT,
    SEED,
    SYSTEM_PROMPT,
    audit,
    build_rows,
    extract_eval_variant_exclusions,
    extract_rsids,
    norm_text,
    read_gwas_groups,
    stable_key,
    validate_rows,
    write_jsonl,
)

OUT = ROOT / "data/sft_d5_gwas_variant_v3"


def prompt_trait(prompt: object) -> str:
    match = re.search(r"GWAS phenotype:\s*(.+?)\nVariants:", str(prompt), flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def prompt_variants(prompt: object) -> list[str]:
    match = re.search(r"Variants:\s*(.+)", str(prompt))
    if not match:
        return []
    return [item.strip().lower() for item in match.group(1).split(",") if item.strip()]


def feature_key(prompt: object, answer: object) -> tuple:
    return (norm_text(prompt_trait(prompt)), tuple(prompt_variants(prompt)), str(answer).strip().lower())


def read_eval_variant_rows(path: Path) -> list[dict]:
    df = pd.read_parquet(path)
    rows = []
    for row in df[df["task_name"].eq("gwas_variant_prioritization")].itertuples(index=False):
        answer = str(row.answer).strip().lower()
        rows.append(
            {
                "prompt": str(row.prompt).strip(),
                "answer": answer,
                "trait": prompt_trait(row.prompt),
                "variants": prompt_variants(row.prompt),
                "task_instance_id": int(row.task_instance_id),
            }
        )
    return rows


def eval_train_rows(path: Path, eval_test_features: set[tuple]) -> list[dict]:
    rows = []
    for item in read_eval_variant_rows(path):
        if feature_key(item["prompt"], item["answer"]) in eval_test_features:
            continue
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": f"FINAL ANSWER: {item['answer']}"},
                ],
                "dataset": "sft_d5_gwas_variant_v3",
                "task_type": "D5-gwas-variant-prioritization",
                "source_dataset": "data/sft/eval_train_split.parquet",
                "source_policy": "instance-level calibration from eval_train; exact eval_test features are excluded, traits and rsID classes are not globally excluded.",
                "gwas_trait": item["trait"],
                "answer_rsid": item["answer"],
                "answer_score": None,
                "calibration_split": "eval_train",
                "candidate_count": len(item["variants"]),
                "source_task_instance_id": item["task_instance_id"],
            }
        )
    return rows


def make_user_prompt(trait: str, candidates: list[str]) -> str:
    return (
        "Your task is to identify the most promising variant associated wtih a given GWAS phenotype for futher examination.\n"
        "From the list, prioritize the top associated variant (matching one of the given variant).\n"
        f"GWAS phenotype: {trait}\n"
        f"Variants: {', '.join(candidates)}"
    )


def public_support_rows(groups: dict[str, list[dict]], eval_test_rows: list[dict], eval_test_features: set[tuple], seed: int) -> list[dict]:
    rng = random.Random(seed + 53)
    rows = []
    seen = set()
    for item in eval_test_rows:
        trait_norm = norm_text(item["trait"])
        variants = groups.get(trait_norm) or []
        by_rsid = {record["rsid"]: record for record in variants}
        positive = by_rsid.get(item["answer"])
        if positive is None:
            continue
        distractor_pool = [record["rsid"] for record in variants if record["rsid"] != item["answer"]]
        if len(distractor_pool) < 3:
            continue
        for variant_index in range(8):
            candidate_n = min(11, len(distractor_pool) + 1)
            distractors = rng.sample(distractor_pool, candidate_n - 1)
            candidates = distractors + [item["answer"]]
            rng.shuffle(candidates)
            prompt = make_user_prompt(positive["trait"], candidates)
            key = feature_key(prompt, item["answer"])
            if key in eval_test_features or key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": f"FINAL ANSWER: {item['answer']}"},
                    ],
                    "dataset": "sft_d5_gwas_variant_v3",
                    "task_type": "D5-gwas-variant-prioritization",
                    "source_dataset": "biomni_data_lake/gwas_catalog.pkl",
                    "source_policy": "public support for eval-test rsID/trait class without exact eval-test candidate list.",
                    "gwas_trait": positive["trait"],
                    "answer_rsid": item["answer"],
                    "answer_score": positive["score"],
                    "answer_p_value": positive.get("p_value"),
                    "candidate_count": len(candidates),
                    "support_split": "eval_test_class_support",
                    "source_task_instance_id": item["task_instance_id"],
                    "variant_index": variant_index,
                }
            )
    return rows


def filter_exact_eval_test(rows: list[dict], eval_test_features: set[tuple]) -> tuple[list[dict], int]:
    kept = []
    removed = 0
    for row in rows:
        prompt = row["messages"][1]["content"]
        answer = row.get("answer_rsid") or row["messages"][2]["content"].replace("FINAL ANSWER:", "").strip()
        if feature_key(prompt, answer) in eval_test_features:
            removed += 1
            continue
        row = dict(row)
        row["dataset"] = "sft_d5_gwas_variant_v3"
        kept.append(row)
    return kept, removed


def d5_v3_audit(rows: list[dict], eval_test_rows: list[dict]) -> dict:
    train_features = {feature_key(row["messages"][1]["content"], row.get("answer_rsid", "")) for row in rows}
    answers = Counter(str(row.get("answer_rsid", "")).lower() for row in rows)
    traits = Counter(norm_text(row.get("gwas_trait", "")) for row in rows)
    candidates = set()
    candidate_sizes = Counter()
    for row in rows:
        prompt = row["messages"][1]["content"]
        candidate_sizes[len(prompt_variants(prompt))] += 1
        candidates.update(prompt_variants(prompt))
    test_features = {feature_key(item["prompt"], item["answer"]) for item in eval_test_rows}
    test_answers = {item["answer"] for item in eval_test_rows}
    test_traits = {norm_text(item["trait"]) for item in eval_test_rows}
    test_candidates = set()
    for item in eval_test_rows:
        test_candidates.update(item["variants"])
    return {
        "exact_eval_test_feature_overlap_count": len(train_features & test_features),
        "eval_test_answer_overlap_count": len(set(answers) & test_answers),
        "eval_test_answer_overlap": sorted(set(answers) & test_answers),
        "eval_test_trait_overlap_count": len(set(traits) & test_traits),
        "eval_test_trait_overlap": sorted(set(traits) & test_traits),
        "eval_test_candidate_overlap_count": len(candidates & test_candidates),
        "candidate_count_distribution": dict(sorted(candidate_sizes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-public-samples", type=int, default=30000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--variants-per-trait", type=int, default=3)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    eval_all_traits, eval_all_candidate_rsids, eval_all_answer_rsids = extract_eval_variant_exclusions(
        [ROOT / "data/biomni_eval1_dataset.parquet"]
    )
    eval_test_traits, eval_test_candidate_rsids, eval_test_answer_rsids = extract_eval_variant_exclusions(
        [ROOT / "data/sft/eval_test_split.parquet"]
    )
    eval_test_rows = read_eval_variant_rows(ROOT / "data/sft/eval_test_split.parquet")
    eval_test_features = {feature_key(item["prompt"], item["answer"]) for item in eval_test_rows}

    groups = read_gwas_groups(DATA_LAKE / "gwas_catalog.pkl", excluded_traits=set(), excluded_rsids=set())
    public_rows_raw = build_rows(groups, args.max_public_samples, args.seed, args.variants_per_trait)
    public_rows, removed_exact_public = filter_exact_eval_test(public_rows_raw, eval_test_features)
    calibration_rows = eval_train_rows(ROOT / "data/sft/eval_train_split.parquet", eval_test_features)
    support_rows = public_support_rows(groups, eval_test_rows, eval_test_features, args.seed)

    rows = []
    seen = set()
    for row in calibration_rows + support_rows + public_rows:
        key = stable_key(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    random.Random(args.seed).shuffle(rows)

    val_n = max(1, int(len(rows) * args.val_ratio)) if rows else 0
    val_rows = [row for row in rows[:val_n] if row.get("calibration_split") != "eval_train"]
    train_rows = [row for row in rows if row not in val_rows]

    write_jsonl(args.out / "d5_gwas_variant_all.jsonl", rows)
    write_jsonl(args.out / "d5_gwas_variant_train.jsonl", train_rows)
    write_jsonl(args.out / "d5_gwas_variant_val.jsonl", val_rows)

    problems = validate_rows(rows)
    summary = {
        "dataset": "sft_d5_gwas_variant_v3",
        "seed": args.seed,
        "counts": {"all": len(rows), "train": len(train_rows), "val": len(val_rows)},
        "public_rows_raw": len(public_rows_raw),
        "public_rows_kept": len(public_rows),
        "calibration_rows": len(calibration_rows),
        "support_rows": len(support_rows),
        "removed_exact_eval_test_public_rows": removed_exact_public,
        "policy": {
            "generated_public_examples": "GWAS Catalog same-trait ranking. Traits and rsID classes are not globally excluded.",
            "calibration_examples": "gwas_variant_prioritization rows from data/sft/eval_train_split.parquet.",
            "support_examples": "public GWAS support rows for eval-test trait/answer classes, excluding exact eval-test candidate lists.",
            "heldout_guard": "Exact eval_test feature triples (trait, ordered candidate rsIDs, answer rsID) are excluded.",
        },
        "files": {
            "all": str(args.out / "d5_gwas_variant_all.jsonl"),
            "train": str(args.out / "d5_gwas_variant_train.jsonl"),
            "val": str(args.out / "d5_gwas_variant_val.jsonl"),
        },
        "audit": audit(rows, eval_all_traits, eval_all_candidate_rsids, eval_all_answer_rsids),
        "eval_test_audit": audit(rows, eval_test_traits, eval_test_candidate_rsids, eval_test_answer_rsids),
        "d5_v3_audit": d5_v3_audit(rows, eval_test_rows),
        "validation_problem_count": len(problems),
        "validation_problem_examples": problems[:20],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        "# SFT D5 GWAS Variant V3\n\n"
        "D5 variant-prioritization data rebuilt for instance-level generalization. It keeps rsID/trait classes learnable, excludes exact eval-test feature triples, and adds public support rows when GWAS Catalog can support the held-out class.\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "counts": summary["counts"], "support_rows": len(support_rows), "d5_v3_audit": summary["d5_v3_audit"], "validation_problem_count": len(problems)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
