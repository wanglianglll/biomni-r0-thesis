#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_config import apply_config_defaults
from scripts.eval_common import default_dataset_path, evaluate_model, resolve_output_dir


def build_parser():
    parser = argparse.ArgumentParser(description="Unified Biomni evaluation runner")
    parser.add_argument("--config", default=None, help="Optional JSON/TOML config file")
    parser.add_argument("--model", choices=["qwen2.5", "qwen3.5", "mistral", "llama3.1", "deepseek"])
    parser.add_argument("--variant", default="base", choices=["base", "sft"])
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
    return parser


def main(argv=None):
    parser = build_parser()
    parser, remaining, config_path, _config = apply_config_defaults(parser, argv)
    args = parser.parse_args(remaining)
    if not args.model:
        parser.error("--model is required unless provided by --config")
    output_dir = resolve_output_dir(args.model, args.variant, args.output_dir)
    evaluate_model(
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
        config_path=config_path,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
