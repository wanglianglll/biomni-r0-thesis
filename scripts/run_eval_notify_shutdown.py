#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autodl_notify import send_autodl_message
from scripts.eval_common import default_dataset_path, evaluate_model, resolve_output_dir


def _compact_summary(summary: dict) -> str:
    overall = summary.get("_overall", {})
    lines = [
        f"Overall: {overall.get('accuracy', 0):.4f} "
        f"({overall.get('correct', 0)}/{overall.get('total', 0)})"
    ]
    task_items = [(k, v) for k, v in summary.items() if not k.startswith("_")]
    task_items.sort(key=lambda item: item[0])
    for task, item in task_items:
        lines.append(
            f"{task}: {item.get('accuracy', 0):.3f} "
            f"({item.get('correct', 0)}/{item.get('total', 0)})"
        )
    return "\n".join(lines)


def _shutdown(delay_seconds: int) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    subprocess.Popen("/usr/bin/shutdown -h now", shell=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Biomni evaluation, notify WeChat, then shut down.")
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--variant", default="sft", choices=["base", "sft"])
    parser.add_argument("--dataset", default=default_dataset_path())
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-between", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--shutdown-delay-seconds", type=int, default=20)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.model, args.variant, args.output_dir)
    send_autodl_message(
        "Biomni evaluation started",
        name=f"{args.model}-{args.variant}",
        content=f"Dataset: {args.dataset}\nOutput: {output_dir}\nSubset: {args.subset or 'full'}",
    )
    try:
        summary = evaluate_model(
            model_family=args.model,
            variant=args.variant,
            dataset_path=args.dataset,
            output_dir=output_dir,
            subset=args.subset,
            max_retries=args.max_retries,
            sleep_between=args.sleep_between,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            config_path=None,
        )
        content = _compact_summary(summary)
        summary_path = Path(output_dir) / f"{summary['_overall']['run_stem']}_summary.json"
        send_autodl_message(
            "Biomni evaluation completed",
            name=f"{args.model}-{args.variant}",
            content=f"{content}\n\nSummary: {summary_path}",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        err = traceback.format_exc()
        send_autodl_message(
            "Biomni evaluation failed",
            name=f"{args.model}-{args.variant}",
            content=err[-1700:],
        )
        print(err, file=sys.stderr)
        return 1
    finally:
        send_autodl_message(
            "Biomni instance shutting down",
            name=f"{args.model}-{args.variant}",
            content=f"Shutdown scheduled in {args.shutdown_delay_seconds}s after evaluation task.",
        )
        _shutdown(args.shutdown_delay_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
