from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("/root/autodl-tmp/Biomni-main")
SRC = ROOT / "data/sft/eval_test_split.parquet"
OUT = ROOT / "data/sft_eval1_taskwise_v1/eval_splits"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SRC)
    summary = {
        "source": str(SRC),
        "policy": "Per-task held-out BiomniEval1 splits for taskwise short training. These come from data/sft/eval_test_split.parquet and must not be used for calibration/training.",
        "tasks": {},
    }
    for task, group in df.groupby("task_name", sort=True):
        path = OUT / f"{task}_eval_test.parquet"
        group.to_parquet(path, index=False)
        summary["tasks"][task] = {
            "rows": int(len(group)),
            "file": str(path),
        }
    summary["all_tasks"] = {"rows": int(len(df)), "file": str(SRC)}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "tasks": {k: v["rows"] for k, v in summary["tasks"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
