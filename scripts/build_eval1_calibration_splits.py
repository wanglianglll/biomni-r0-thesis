from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path("/root/autodl-tmp/Biomni-main")
SRC = ROOT / "data/biomni_eval1_dataset.parquet"
OUT = ROOT / "data/sft"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe BiomniEval1 calibration/test splits.")
    parser.add_argument("--source", type=Path, default=SRC)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--calibration-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.calibration_ratio < 1:
        raise ValueError("--calibration-ratio must be between 0 and 1")

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.source)
    if "split" in df.columns:
        df = df[df["split"].eq("val")].copy()

    calibration_parts = []
    heldout_parts = []
    task_summary = {}
    for task_name in sorted(df["task_name"].unique()):
        task_df = df[df["task_name"].eq(task_name)].sample(frac=1, random_state=args.seed).reset_index(drop=True)
        n = len(task_df)
        n_calibration = max(1, int(n * args.calibration_ratio))
        n_calibration = min(n - 1, n_calibration) if n > 1 else n_calibration
        calibration = task_df.iloc[:n_calibration]
        heldout = task_df.iloc[n_calibration:]
        calibration_parts.append(calibration)
        heldout_parts.append(heldout)
        task_summary[task_name] = {
            "total": int(n),
            "calibration_train": int(len(calibration)),
            "heldout_test": int(len(heldout)),
        }

    calibration_df = pd.concat(calibration_parts, ignore_index=True)
    heldout_df = pd.concat(heldout_parts, ignore_index=True)
    train_path = args.out / "eval_train_split.parquet"
    test_path = args.out / "eval_test_split.parquet"
    calibration_df.to_parquet(train_path, index=False)
    heldout_df.to_parquet(test_path, index=False)

    summary = {
        "source": str(args.source),
        "seed": args.seed,
        "calibration_ratio": args.calibration_ratio,
        "policy": "Use eval_train_split only as task-format calibration. Use eval_test_split as held-out short-train benchmark data.",
        "files": {"calibration_train": str(train_path), "heldout_test": str(test_path)},
        "counts": {
            "total": int(len(df)),
            "calibration_train": int(len(calibration_df)),
            "heldout_test": int(len(heldout_df)),
        },
        "tasks": task_summary,
    }
    (args.out / "eval_split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
