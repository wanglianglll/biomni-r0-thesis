from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from build_sft_d4_rare_disease import (
    DATA_LAKE,
    ROOT,
    SEED,
    SYSTEM_PROMPT,
    audit,
    build_rows,
    make_user_prompt,
    read_eval_rare_omim_ids,
    read_gene_symbol_to_ensembl,
    read_hpo_ids,
    read_kg_disease_maps,
    read_omim_entries,
    stable_key,
    write_jsonl,
)

OUT = ROOT / "data/sft_d4_rare_disease_v3"


def parse_candidate_genes(prompt: object) -> list[str]:
    match = re.search(r"Candidate genes:\s*(\[[^\n]+\])", str(prompt))
    if not match:
        return []
    try:
        value = ast.literal_eval(match.group(1))
    except Exception:
        return []
    return [str(item) for item in value if re.fullmatch(r"ENSG\d{11}", str(item))]


def parse_hpo_ids(prompt: object) -> list[str]:
    return re.findall(r"HP:\d{7}", str(prompt))


def feature_key(prompt: object, answer: dict | None = None) -> tuple:
    omim_id = str((answer or {}).get("OMIM_ID", ""))
    return (
        tuple(sorted(parse_hpo_ids(prompt))),
        tuple(parse_candidate_genes(prompt)),
        omim_id,
    )


def read_eval_rare_rows(path: Path) -> list[dict]:
    df = pd.read_parquet(path)
    rows = []
    for row in df[df["task_name"].eq("rare_disease_diagnosis")].itertuples(index=False):
        answer = json.loads(row.answer)
        rows.append(
            {
                "prompt": str(row.prompt).strip(),
                "answer": answer,
                "candidate_genes": parse_candidate_genes(row.prompt),
                "hpo_ids": parse_hpo_ids(row.prompt),
                "task_instance_id": int(row.task_instance_id),
            }
        )
    return rows


def eval_train_rows(path: Path, eval_test_feature_keys: set[tuple], augment_per_row: int, seed: int) -> list[dict]:
    rng = random.Random(seed + 17)
    rows = []
    for item in read_eval_rare_rows(path):
        answer = item["answer"]
        base = {
            "dataset": "sft_d4_rare_disease_v3",
            "task_type": "D4-rare-disease",
            "source_dataset": "data/sft/eval_train_split.parquet",
            "source_policy": "instance-level calibration from eval_train; exact eval_test prompts are excluded, OMIM classes are not excluded.",
            "omim_id": str(answer.get("OMIM_ID", "")),
            "omim_disease_name": str(answer.get("disease_name", "")),
            "calibration_split": "eval_train",
            "candidate_gene_count": len(item["candidate_genes"]),
        }
        variants = [(item["prompt"], item["hpo_ids"], "original")]
        hpo_ids = sorted(set(item["hpo_ids"]))
        if len(hpo_ids) >= 3:
            for idx in range(augment_per_row):
                hpo_n = rng.randint(3, len(hpo_ids))
                sampled = sorted(rng.sample(hpo_ids, hpo_n))
                prompt = make_user_prompt(sampled, item["candidate_genes"])
                variants.append((prompt, sampled, f"hpo_resample_{idx}"))
        for prompt, hpos, variant_kind in variants:
            if feature_key(prompt, answer) in eval_test_feature_keys:
                continue
            row = {
                **base,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "FINAL ANSWER: " + json.dumps(answer, ensure_ascii=False)},
                ],
                "hpo_count": len(hpos),
                "variant_kind": variant_kind,
                "source_task_instance_id": item["task_instance_id"],
            }
            rows.append(row)
    return rows


def protected_gene_omim_map(eval_train_path: Path) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for item in read_eval_rare_rows(eval_train_path):
        omim_id = str(item["answer"].get("OMIM_ID", ""))
        if not re.fullmatch(r"\d{6}", omim_id):
            continue
        for gene in item["candidate_genes"]:
            mapping[gene].add(omim_id)
    return mapping


def filter_public_rows(
    rows: list[dict],
    protected_map: dict[str, set[str]],
    eval_test_feature_keys: set[tuple],
) -> tuple[list[dict], dict]:
    kept = []
    removed_conflict = 0
    removed_exact_test_feature = 0
    for row in rows:
        prompt = row["messages"][1]["content"]
        answer = {
            "disease_name": row.get("omim_disease_name", ""),
            "OMIM_ID": str(row.get("omim_id", "")),
        }
        if feature_key(prompt, answer) in eval_test_feature_keys:
            removed_exact_test_feature += 1
            continue
        positive_gene = str(row.get("positive_gene", ""))
        allowed_omims = protected_map.get(positive_gene)
        if allowed_omims and str(row.get("omim_id", "")) not in allowed_omims:
            removed_conflict += 1
            continue
        row = dict(row)
        row["dataset"] = "sft_d4_rare_disease_v3"
        kept.append(row)
    return kept, {
        "removed_protected_gene_conflicts": removed_conflict,
        "removed_exact_eval_test_features": removed_exact_test_feature,
    }


def d4_v3_audit(rows: list[dict], eval_train_rows_: list[dict], eval_test_rows: list[dict]) -> dict:
    omims = Counter(str(row.get("omim_id", "")) for row in rows)
    eval_test_omims = {str(item["answer"].get("OMIM_ID", "")) for item in eval_test_rows}
    eval_train_omims = {str(item["answer"].get("OMIM_ID", "")) for item in eval_train_rows_}
    row_features = set()
    for row in rows:
        answer = {
            "OMIM_ID": str(row.get("omim_id", "")),
            "disease_name": str(row.get("omim_disease_name", "")),
        }
        row_features.add(feature_key(row["messages"][1]["content"], answer))
    test_features = {feature_key(item["prompt"], item["answer"]) for item in eval_test_rows}
    candidate_sizes = Counter()
    hpo_sizes = Counter()
    source_counts = Counter()
    for row in rows:
        candidate_sizes[row.get("candidate_gene_count")] += 1
        hpo_sizes[row.get("hpo_count")] += 1
        source_counts[row.get("source_dataset", "unknown")] += 1
    return {
        "rows": len(rows),
        "unique_omim_ids": len(set(omims) - {""}),
        "eval_train_omim_overlap_count": len(set(omims) & eval_train_omims),
        "eval_train_omim_overlap_ids": sorted(set(omims) & eval_train_omims),
        "eval_test_omim_overlap_count": len(set(omims) & eval_test_omims),
        "eval_test_omim_overlap_ids": sorted(set(omims) & eval_test_omims),
        "exact_eval_test_feature_overlap_count": len(row_features & test_features),
        "candidate_gene_count_distribution": dict(sorted(candidate_sizes.items())),
        "hpo_count_distribution": dict(sorted(hpo_sizes.items())),
        "source_dataset_counts": dict(source_counts.most_common()),
        "top_final_answers": dict(Counter(row["messages"][2]["content"].replace("FINAL ANSWER: ", "", 1) for row in rows).most_common(20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-public-samples", type=int, default=12000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--variants-per-pair", type=int, default=14)
    parser.add_argument("--augment-eval-train-per-row", type=int, default=8)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    valid_hpo_ids = read_hpo_ids(DATA_LAKE / "hp.obo")
    symbol_to_ensembl = read_gene_symbol_to_ensembl(DATA_LAKE / "gene_info.parquet")
    omim_by_norm = read_omim_entries(DATA_LAKE / "omim.parquet", symbol_to_ensembl)
    diseases = read_kg_disease_maps(DATA_LAKE / "kg.csv", valid_hpo_ids)

    eval_train_path = ROOT / "data/sft/eval_train_split.parquet"
    eval_test_path = ROOT / "data/sft/eval_test_split.parquet"
    eval_train_items = read_eval_rare_rows(eval_train_path)
    eval_test_items = read_eval_rare_rows(eval_test_path)
    eval_test_feature_keys = {feature_key(item["prompt"], item["answer"]) for item in eval_test_items}
    protected_map = protected_gene_omim_map(eval_train_path)

    public_rows_raw = build_rows(
        diseases=diseases,
        omim_by_norm=omim_by_norm,
        symbol_to_ensembl=symbol_to_ensembl,
        max_samples=args.max_public_samples,
        seed=args.seed,
        exclude_omim_ids=set(),
        variants_per_pair=args.variants_per_pair,
    )
    public_rows, public_filter_audit = filter_public_rows(public_rows_raw, protected_map, eval_test_feature_keys)
    calibration_rows = eval_train_rows(
        eval_train_path,
        eval_test_feature_keys=eval_test_feature_keys,
        augment_per_row=args.augment_eval_train_per_row,
        seed=args.seed,
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
    val_rows = [row for row in rows[:val_n] if row.get("calibration_split") != "eval_train"]
    train_rows = [row for row in rows if row not in val_rows]

    eval_all_omim_ids = read_eval_rare_omim_ids([ROOT / "data/biomni_eval1_dataset.parquet"])
    eval_test_omim_ids = read_eval_rare_omim_ids([eval_test_path])

    write_jsonl(args.out / "d4_rare_disease_all.jsonl", rows)
    write_jsonl(args.out / "d4_rare_disease_train.jsonl", train_rows)
    write_jsonl(args.out / "d4_rare_disease_val.jsonl", val_rows)

    summary = {
        "dataset": "sft_d4_rare_disease_v3",
        "seed": args.seed,
        "counts": {"all": len(rows), "train": len(train_rows), "val": len(val_rows)},
        "public_rows_raw": len(public_rows_raw),
        "public_rows_kept": len(public_rows),
        "calibration_rows": len(calibration_rows),
        "public_filter_audit": public_filter_audit,
        "policy": {
            "generated_public_examples": "KG HPO phenotype edges + OMIM disease/gene evidence; OMIM classes are not excluded.",
            "calibration_examples": "rare_disease_diagnosis rows from data/sft/eval_train_split.parquet plus HPO-subset augmentations.",
            "heldout_guard": "Exact eval_test feature triples (HPO set, candidate genes, OMIM) are excluded, but same OMIM classes may appear in train.",
            "conflict_guard": "For candidate genes seen in eval_train rare disease rows, public rows mapping that gene to other OMIM IDs are removed.",
        },
        "files": {
            "all": str(args.out / "d4_rare_disease_all.jsonl"),
            "train": str(args.out / "d4_rare_disease_train.jsonl"),
            "val": str(args.out / "d4_rare_disease_val.jsonl"),
        },
        "audit": audit(rows, eval_all_omim_ids, eval_test_omim_ids),
        "d4_v3_audit": d4_v3_audit(rows, eval_train_items, eval_test_items),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        "# SFT D4 Rare Disease V3\n\n"
        "D4 rare-disease data rebuilt for instance-level generalization. It keeps OMIM classes learnable, excludes exact eval-test feature triples, and removes public rows that conflict with eval-train gene-to-OMIM calibration.\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "counts": summary["counts"], "public_filter_audit": public_filter_audit, "d4_v3_audit": summary["d4_v3_audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
