from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/Biomni-main")
DEFAULT_PLAN = ROOT / "data/sft_eval1_taskwise_v1/short_train_eval_plan.json"
DEFAULT_BASE = ROOT / "configs/train_llama31_qlora_eval1_taskwise_short_v1.json"
DEFAULT_OUT = ROOT / "configs/eval1_taskwise_short_v1"
EVAL_SPLITS = ROOT / "data/sft_eval1_taskwise_v1/eval_splits"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create per-task short-training configs for Eval1 taskwise SFT data.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-prefix", default="llama31")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    for stage in plan["stages"]:
        name = stage["stage"]
        cfg = dict(base)
        cfg["train_file"] = [stage["train_file"]]
        cfg["epochs"] = stage.get("suggested_epochs", cfg.get("epochs", 0.15))
        if name == "mixed_all_tasks":
            cfg["benchmark_eval_dataset"] = str(ROOT / "data/biomni_eval1_dataset.parquet")
            cfg["benchmark_eval_subset_per_task"] = 0
        else:
            cfg["benchmark_eval_dataset"] = str(EVAL_SPLITS / f"{name}_eval_test.parquet")
            cfg["benchmark_eval_subset_per_task"] = 0
        cfg["shutdown_after"] = False
        cfg_path = args.out_dir / f"train_{args.model_prefix}_qlora_{name}.json"
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        commands.append(
            {
                "stage": name,
                "eval_focus": stage.get("eval_focus", "all"),
                "config": str(cfg_path),
                "command": f"/root/miniconda3/envs/biomni/bin/python scripts/run_train.py --config {cfg_path.relative_to(ROOT)}",
            }
        )

    commands_path = args.out_dir / "commands.json"
    commands_path.write_text(json.dumps(commands, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Eval1 Taskwise Short Training Commands", ""]
    for item in commands:
        md.append(f"## {item['stage']}")
        md.append(f"Focus: `{item['eval_focus']}`")
        md.append("")
        md.append("```bash")
        md.append(item["command"])
        md.append("```")
        md.append("")
    (args.out_dir / "COMMANDS.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "stages": len(commands), "commands_file": str(commands_path)}, indent=2))


if __name__ == "__main__":
    main()
