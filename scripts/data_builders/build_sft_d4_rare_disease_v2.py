from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

from build_sft_d4_rare_disease import (
    DATA_LAKE,
    ROOT,
    SEED,
    SYSTEM_PROMPT,
    audit,
    build_rows,
    read_eval_rare_omim_ids,
    read_gene_symbol_to_ensembl,
    read_hpo_ids,
    read_kg_disease_maps,
    read_omim_entries,
    stable_key,
    write_jsonl,
)

OUT = ROOT / "data/sft_d4_rare_disease_v2"


def eval_train_rows(path: Path, excluded_omim_ids: set[str]) -> list[dict]:
    df = pd.read_parquet(path)
    rows = []
    for row in df[df["task_name"].eq("rare_disease_diagnosis")].itertuples(index=False):
        answer = json.loads(row.answer)
        if str(answer.get("OMIM_ID", "")) in excluded_omim_ids:
            continue
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(row.prompt).strip()},
                    {"role": "assistant", "content": "FINAL ANSWER: " + json.dumps(answer, ensure_ascii=False)},
                ],
                "dataset": "sft_d4_rare_disease_v2",
                "task_type": "D4-rare-disease",
                "source_dataset": "data/sft/eval_train_split.parquet",
                "source_policy": "calibration examples from the provided eval_train split; eval_test OMIM IDs are excluded from generated public examples",
                "omim_id": str(answer.get("OMIM_ID", "")),
                "omim_disease_name": str(answer.get("disease_name", "")),
                "calibration_split": "eval_train",
                "candidate_gene_count": len(re.findall(r"ENSG\d{11}", str(row.prompt))),
                "hpo_count": len(re.findall(r"HP:\d{7}", str(row.prompt))),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-public-samples", type=int, default=12000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--variants-per-pair", type=int, default=14)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    valid_hpo_ids = read_hpo_ids(DATA_LAKE / "hp.obo")
    symbol_to_ensembl = read_gene_symbol_to_ensembl(DATA_LAKE / "gene_info.parquet")
    omim_by_norm = read_omim_entries(DATA_LAKE / "omim.parquet", symbol_to_ensembl)
    diseases = read_kg_disease_maps(DATA_LAKE / "kg.csv", valid_hpo_ids)
    eval_test_omim_ids = read_eval_rare_omim_ids([ROOT / "data/sft/eval_test_split.parquet"])
    eval_all_omim_ids = read_eval_rare_omim_ids([ROOT / "data/biomni_eval1_dataset.parquet"])

    public_rows = build_rows(
        diseases=diseases,
        omim_by_norm=omim_by_norm,
        symbol_to_ensembl=symbol_to_ensembl,
        max_samples=args.max_public_samples,
        seed=args.seed,
        exclude_omim_ids=eval_test_omim_ids,
        variants_per_pair=args.variants_per_pair,
    )
    calibration_rows = eval_train_rows(ROOT / "data/sft/eval_train_split.parquet", eval_test_omim_ids)

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

    write_jsonl(args.out / "d4_rare_disease_all.jsonl", rows)
    write_jsonl(args.out / "d4_rare_disease_train.jsonl", train_rows)
    write_jsonl(args.out / "d4_rare_disease_val.jsonl", val_rows)

    summary = {
        "dataset": "sft_d4_rare_disease_v2",
        "seed": args.seed,
        "counts": {"all": len(rows), "train": len(train_rows), "val": len(val_rows)},
        "public_rows": len(public_rows),
        "calibration_rows": len(calibration_rows),
        "policy": {
            "generated_public_examples": "KG HPO phenotype edges + OMIM disease/gene evidence; eval_test OMIM IDs excluded.",
            "calibration_examples": "rare_disease_diagnosis rows from data/sft/eval_train_split.parquet.",
            "heldout_guard": "data/sft/eval_test_split.parquet OMIM IDs are excluded.",
        },
        "files": {
            "all": str(args.out / "d4_rare_disease_all.jsonl"),
            "train": str(args.out / "d4_rare_disease_train.jsonl"),
            "val": str(args.out / "d4_rare_disease_val.jsonl"),
        },
        "audit": audit(rows, eval_all_omim_ids, eval_test_omim_ids),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        "# SFT D4 Rare Disease V2\n\n"
        "Gene-aware rare-disease targeted data with eval-train calibration and eval-test exclusion.\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "counts": summary["counts"], "audit": summary["audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
