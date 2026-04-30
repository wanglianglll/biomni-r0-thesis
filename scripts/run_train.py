#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_config import apply_config_defaults
from scripts.autodl_notify import send_autodl_message
from scripts.model_registry import LOCAL_MODELS
from scripts.train_common import default_mixture_paths, default_train_path, train_sft


def build_parser():
    parser = argparse.ArgumentParser(description="Unified Biomni LoRA/QLoRA SFT training runner")
    parser.add_argument("--config", default=None, help="Optional JSON/TOML config file")
    parser.add_argument("--model", choices=sorted(LOCAL_MODELS))
    parser.add_argument(
        "--train-file",
        action="append",
        default=None,
        help="JSONL training file. Can be passed multiple times. Config may use a string or a list.",
    )
    parser.add_argument(
        "--use-d123-mixture",
        action="store_true",
        help="Use D1 train + D2 train + D3-SFT train as training files when --train-file is not set.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--test-size", type=float, default=0.02)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--save-strategy", default="steps", choices=["no", "steps", "epoch", "best"])
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--peft-method", choices=["lora", "qlora"], default="qlora")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="auto", help="Comma-separated modules or auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--bnb-4bit-compute-dtype", default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    parser.add_argument("--model-max-memory-gpu", default=None, help="Optional max GPU memory for device_map loading, e.g. 28GiB")
    parser.add_argument("--model-max-memory-cpu", default=None, help="Optional max CPU memory for device_map loading, e.g. 100GiB")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and write metadata without loading model or training")
    parser.add_argument("--no-gradient-checkpointing", action="store_true", help="Disable gradient checkpointing for faster training when memory is enough")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metric-for-best-model", default="eval_loss")
    parser.add_argument("--greater-is-better", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--notify-interval-seconds", type=int, default=3600)
    parser.add_argument("--benchmark-eval-steps", type=int, default=0, help="Run a small benchmark every N training steps; 0 disables it")
    parser.add_argument("--benchmark-eval-dataset", default=None, help="Parquet eval dataset for periodic benchmark checks")
    parser.add_argument("--benchmark-eval-subset-per-task", type=int, default=3)
    parser.add_argument("--benchmark-eval-start-step", type=int, default=0)
    parser.add_argument("--benchmark-eval-patience", type=int, default=2)
    parser.add_argument("--benchmark-eval-min-delta", type=float, default=0.0)
    parser.add_argument("--benchmark-eval-max-new-tokens", type=int, default=96)
    parser.add_argument("--benchmark-template-issue-threshold", type=float, default=0.25)
    parser.add_argument("--shutdown-after", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    parser, remaining, config_path, _config = apply_config_defaults(parser, argv)
    args = parser.parse_args(remaining)
    if not args.model:
        parser.error("--model is required unless provided by --config")
    train_files = args.train_file
    if train_files is None and args.use_d123_mixture:
        train_files = default_mixture_paths()
    if train_files is None:
        train_files = [default_train_path()]
    train_sft(
        model_family=args.model,
        train_file=train_files,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        test_size=args.test_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy=args.save_strategy,
        shutdown_after=args.shutdown_after,
        config_path=config_path,
        peft_method=args.peft_method,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        torch_dtype=args.torch_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        dry_run=args.dry_run,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        notify_interval_seconds=args.notify_interval_seconds,
        benchmark_eval_steps=args.benchmark_eval_steps,
        benchmark_eval_dataset=args.benchmark_eval_dataset,
        benchmark_eval_subset_per_task=args.benchmark_eval_subset_per_task,
        benchmark_eval_start_step=args.benchmark_eval_start_step,
        benchmark_eval_patience=args.benchmark_eval_patience,
        benchmark_eval_min_delta=args.benchmark_eval_min_delta,
        benchmark_eval_max_new_tokens=args.benchmark_eval_max_new_tokens,
        benchmark_template_issue_threshold=args.benchmark_template_issue_threshold,
        model_max_memory_gpu=args.model_max_memory_gpu,
        model_max_memory_cpu=args.model_max_memory_cpu,
    )


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as exc:
        send_autodl_message("Biomni????", "run_train.py", f"{type(exc).__name__}: {exc}")
        raise
