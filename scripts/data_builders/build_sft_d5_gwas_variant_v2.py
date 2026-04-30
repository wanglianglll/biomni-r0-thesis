from __future__ import annotations

import argparse
import json
import random
import re
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
    read_gwas_groups,
    stable_key,
    validate_rows,
    write_jsonl,
)

OUT = ROOT / "data/sft_d5_gwas_variant_v2"


def norm_prompt_trait(prompt: str) -> str:
    match = re.search(r"GWAS phenotype:\s*(.+?)\nVariants:", prompt, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def eval_train_rows(path: Path, excluded_traits: set[str], excluded_answers: set[str]) -> list[dict]:
    df = pd.read_parquet(path)
    rows = []
    for row in df[df["task_name"].eq("gwas_variant_prioritization")].itertuples(index=False):
        answer = str(row.answer).strip().lower()
        trait = norm_prompt_trait(str(row.prompt))
        trait_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", trait.lower())).strip()
        if answer in excluded_answers or trait_norm in excluded_traits:
            continue
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(row.prompt).strip()},
                    {"role": "assistant", "content": f"FINAL ANSWER: {answer}"},
                ],
                "dataset": "sft_d5_gwas_variant_v2",
                "task_type": "D5-gwas-variant-prioritization",
                "source_dataset": "data/sft/eval_train_split.parquet",
                "source_policy": "calibration examples from the provided eval_train split; eval_test traits/answers are excluded from generated public examples",
                "gwas_trait": trait,
                "answer_rsid": answer,
                "answer_score": None,
                "calibration_split": "eval_train",
                "candidate_count": len(re.findall(r"rs\d+", str(row.prompt), flags=re.IGNORECASE)),
            }
        )
    return rows


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

    # V1 removed every eval candidate rsID, which made the task unrealistically
    # out-of-distribution. V2 protects the held-out split by excluding eval-test
    # traits and answers, but allows public GWAS rows to include common rsIDs.
    groups = read_gwas_groups(
        DATA_LAKE / "gwas_catalog.pkl",
        excluded_traits=eval_test_traits,
        excluded_rsids=eval_test_answer_rsids,
    )
    public_rows = build_rows(groups, args.max_public_samples, args.seed, args.variants_per_trait)
    calibration_rows = eval_train_rows(
        ROOT / "data/sft/eval_train_split.parquet",
        excluded_traits=eval_test_traits,
        excluded_answers=eval_test_answer_rsids,
    )

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
    val_rows = [r for r in rows[:val_n] if r.get("calibration_split") != "eval_train"]
    train_rows = [r for r in rows if r not in val_rows]

    write_jsonl(args.out / "d5_gwas_variant_all.jsonl", rows)
    write_jsonl(args.out / "d5_gwas_variant_train.jsonl", train_rows)
    write_jsonl(args.out / "d5_gwas_variant_val.jsonl", val_rows)

    problems = validate_rows(rows)
    summary = {
        "dataset": "sft_d5_gwas_variant_v2",
        "seed": args.seed,
        "counts": {"all": len(rows), "train": len(train_rows), "val": len(val_rows)},
        "public_rows": len(public_rows),
        "calibration_rows": len(calibration_rows),
        "policy": {
            "generated_public_examples": "GWAS Catalog same-trait ranking. Eval-test traits and answer rsIDs are excluded.",
            "calibration_examples": "gwas_variant_prioritization rows from data/sft/eval_train_split.parquet.",
            "heldout_guard": "data/sft/eval_test_split.parquet traits and answer rsIDs are excluded.",
            "rsid_policy": "Common candidate rsIDs are allowed when they appear naturally in public GWAS rows.",
        },
        "files": {
            "all": str(args.out / "d5_gwas_variant_all.jsonl"),
            "train": str(args.out / "d5_gwas_variant_train.jsonl"),
            "val": str(args.out / "d5_gwas_variant_val.jsonl"),
        },
        "audit": audit(rows, eval_all_traits, eval_all_candidate_rsids, eval_all_answer_rsids),
        "eval_test_audit": audit(rows, eval_test_traits, eval_test_candidate_rsids, eval_test_answer_rsids),
        "validation_problem_count": len(problems),
        "validation_problem_examples": problems[:20],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        "# SFT D5 GWAS Variant V2\n\n"
        "GWAS variant-prioritization targeted data with eval-train calibration and eval-test exclusion.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "counts": summary["counts"],
                "audit": summary["audit"],
                "eval_test_audit": summary["eval_test_audit"],
                "validation_problem_count": len(problems),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
