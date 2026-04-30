from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from build_sft_targeted_100k_v2 import (
    ROOT,
    SEED,
    audit,
    family_task,
    sample_stream,
    stable_key,
    write_jsonl,
)

OUT = ROOT / "data/sft_targeted_100k_v3"

SPECS = [
    ("D1", ROOT / "data/sft_d1_datalake_v2/d1_datalake_train.jsonl", 45000),
    ("D2", ROOT / "data/sft_d2_contrast_v1/d2_contrast_train.jsonl", 4000),
    ("D3", ROOT / "data/sft_d3_verifiable_v1/d3_verifiable_sft_train.jsonl", 4000),
    ("D4", ROOT / "data/sft_d4_rare_disease_v3/d4_rare_disease_train.jsonl", 10000),
    ("D5", ROOT / "data/sft_d5_gwas_variant_v2/d5_gwas_variant_train.jsonl", 20000),
    ("D6", ROOT / "data/sft_d6_gwas_causal_gene_v1/d6_gwas_causal_gene_train.jsonl", 15000),
]


def normalize_schema(obj: dict, mixture_part: str) -> dict:
    return {
        "messages": obj["messages"],
        "mixture_dataset": "sft_targeted_100k_v3",
        "mixture_part": mixture_part,
        "task_type": family_task(obj),
        "source_dataset": str(obj.get("source_dataset") or obj.get("source_file") or obj.get("source") or "unknown"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = {
        "dataset": "sft_targeted_100k_v3",
        "seed": args.seed,
        "policy": "D1-heavy mixture with D4 rare-disease v3, calibrated D5 v2, and reduced D6. Assistant messages are single FINAL ANSWER lines.",
        "parts": {},
    }
    global_seen = set()
    for index, (name, path, target_n) in enumerate(SPECS):
        rows = sample_stream(path, target_n, args.seed + index)
        part_rows = []
        for row in rows:
            key = stable_key(row)
            if key in global_seen:
                continue
            global_seen.add(key)
            new_row = normalize_schema(row, name)
            part_rows.append(new_row)
            all_rows.append(new_row)
        out_path = args.out / f"{name.lower()}_targeted_train.jsonl"
        write_jsonl(out_path, part_rows)
        summary["parts"][name] = {
            "source": str(path),
            "target": target_n,
            "actual": len(part_rows),
            "file": str(out_path),
            "audit": audit(part_rows),
        }

    random.Random(args.seed).shuffle(all_rows)
    write_jsonl(args.out / "sft_targeted_100k_train_messages.jsonl", all_rows)
    write_jsonl(args.out / "sft_targeted_100k_train.jsonl", all_rows)
    summary["total_samples"] = len(all_rows)
    summary["part_counts"] = dict(Counter(row.get("mixture_part") for row in all_rows).most_common())
    summary["audit"] = audit(all_rows)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        "# SFT Targeted 100K V3\n\n"
        "A short-validation SFT mixture using D4 rare-disease v3 while keeping the other parts stable.\n\n"
        "- D1: 45K direct datalake samples\n"
        "- D2: 4K contrast samples\n"
        "- D3: 4K verifiable samples\n"
        "- D4: 10K rare-disease targeted v3 samples\n"
        "- D5: 20K GWAS variant-prioritization targeted v2 samples\n"
        "- D6: 15K GWAS causal-gene targeted samples\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "total": len(all_rows), "part_counts": summary["part_counts"], "audit": summary["audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
