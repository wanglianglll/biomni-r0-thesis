from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import datasets
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerCallback

from scripts.model_registry import PROJECT_ROOT, SCRIPT_OUTPUT_DIR, DATA_DIR, get_local_model_spec
from scripts.autodl_notify import send_autodl_message
from scripts.eval_common import (
    SYSTEM_PROMPT,
    build_user_prompt,
    compute_score,
    default_dataset_path,
    extract_answer,
    task_max_new_tokens,
)


class LiveTrainingStatusCallback(TrainerCallback):
    def __init__(self, output_dir: str, run_stem: str, metadata: dict, notify_interval_seconds: int = 3600):
        self.output_dir = output_dir
        self.run_stem = run_stem
        self.metadata = metadata
        self.started_at = time.time()
        self.status_path = os.path.join(output_dir, "live_status.json")
        self.events_path = os.path.join(output_dir, "live_events.jsonl")
        self.last_status = {}
        self.notify_interval_seconds = notify_interval_seconds
        self.last_notify_at = 0.0
        self._write({"status": "initialized", "message": "Trainer initialized"})

    def _gpu_snapshot(self):
        if not torch.cuda.is_available():
            return {"available": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "memory_free_gb": round(free / 1024**3, 3),
            "memory_total_gb": round(total / 1024**3, 3),
            "memory_used_gb": round((total - free) / 1024**3, 3),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        }

    def _write(self, extra: dict):
        now = time.time()
        elapsed = now - self.started_at
        step = int(extra.get("global_step") or self.last_status.get("global_step") or 0)
        max_steps = int(extra.get("max_steps") or self.last_status.get("max_steps") or 0)
        progress = (step / max_steps) if max_steps else 0.0
        seconds_per_step = (elapsed / step) if step else None
        eta_seconds = (seconds_per_step * (max_steps - step)) if (seconds_per_step and max_steps and step <= max_steps) else None
        status = {
            "run_stem": self.run_stem,
            "output_dir": self.output_dir,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "status": extra.get("status", self.last_status.get("status", "running")),
            "global_step": step,
            "max_steps": max_steps,
            "progress": round(progress, 6),
            "progress_percent": round(progress * 100, 2),
            "elapsed_seconds": round(elapsed, 1),
            "seconds_per_step": round(seconds_per_step, 3) if seconds_per_step else None,
            "eta_seconds": round(eta_seconds, 1) if eta_seconds else None,
            "epoch": extra.get("epoch", self.last_status.get("epoch")),
            "latest_log": extra.get("latest_log", self.last_status.get("latest_log", {})),
            "metadata": self.metadata,
            "gpu": self._gpu_snapshot(),
        }
        self.last_status = status
        tmp = self.status_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.status_path)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(status, ensure_ascii=False) + "\n")
        return status

    def _summary(self, status: dict) -> str:
        eta = status.get("eta_seconds")
        eta_text = "unknown" if eta is None else f"{eta/3600:.2f}h"
        log = status.get("latest_log") or {}
        loss = log.get("loss", log.get("eval_loss", "-"))
        return (
            f"status={status.get('status')}\n"
            f"step={status.get('global_step')}/{status.get('max_steps')} ({status.get('progress_percent')}%)\n"
            f"epoch={status.get('epoch')} loss={loss}\n"
            f"speed={status.get('seconds_per_step')}s/step ETA={eta_text}\n"
            f"output={self.output_dir}"
        )

    def on_train_begin(self, args, state, control, **kwargs):
        status = self._write({"status": "running", "global_step": state.global_step, "max_steps": state.max_steps, "epoch": state.epoch})
        send_autodl_message("Biomni????", self.run_stem, self._summary(status))
        self.last_notify_at = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        status = self._write({"status": "running", "global_step": state.global_step, "max_steps": state.max_steps, "epoch": state.epoch, "latest_log": logs or {}})
        now = time.time()
        if self.notify_interval_seconds and now - self.last_notify_at >= self.notify_interval_seconds:
            send_autodl_message("Biomni????", self.run_stem, self._summary(status))
            self.last_notify_at = now

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._write({"status": "evaluating", "global_step": state.global_step, "max_steps": state.max_steps, "epoch": state.epoch, "latest_log": metrics or {}})

    def on_save(self, args, state, control, **kwargs):
        self._write({"status": "saving", "global_step": state.global_step, "max_steps": state.max_steps, "epoch": state.epoch})

    def on_train_end(self, args, state, control, **kwargs):
        status = self._write({"status": "finished", "global_step": state.global_step, "max_steps": state.max_steps, "epoch": state.epoch})
        send_autodl_message("Biomni????", self.run_stem, self._summary(status))


class PeriodicBenchmarkCallback(TrainerCallback):
    def __init__(
        self,
        model_family: str,
        tokenizer,
        output_dir: str,
        run_stem: str,
        dataset_path: str,
        eval_steps: int,
        subset_per_task: int,
        start_step: int,
        patience: int,
        min_delta: float,
        max_new_tokens: int,
        template_issue_threshold: float,
        seed: int = 42,
    ):
        self.model_family = model_family
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.run_stem = run_stem
        self.dataset_path = dataset_path
        self.eval_steps = int(eval_steps or 0)
        self.subset_per_task = int(subset_per_task or 0)
        self.start_step = int(start_step or 0)
        self.patience = int(patience or 0)
        self.min_delta = float(min_delta)
        self.max_new_tokens = int(max_new_tokens)
        self.template_issue_threshold = float(template_issue_threshold)
        self.seed = seed
        self.best_score = None
        self.bad_rounds = 0
        self.last_eval_step = 0
        self.rows = self._load_rows()
        self.events_path = os.path.join(output_dir, "benchmark_events.jsonl")
        self.summary_path = os.path.join(output_dir, "benchmark_latest.json")

    def _load_rows(self) -> list[dict]:
        if not self.eval_steps:
            return []
        path = Path(self.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark eval dataset not found: {path}")
        df = pd.read_parquet(path)
        if "split" in df.columns:
            df = df[df["split"] == "val"]
        rows = []
        for task in sorted(df["task_name"].unique()):
            task_df = df[df["task_name"] == task]
            if self.subset_per_task > 0:
                task_df = task_df.sample(
                    n=min(self.subset_per_task, len(task_df)),
                    random_state=self.seed,
                ).sort_values("task_instance_id")
            for _, row in task_df.iterrows():
                rows.append(
                    {
                        "task_name": row["task_name"],
                        "task_instance_id": int(row["task_instance_id"]),
                        "prompt": row["prompt"],
                        "answer": row["answer"],
                    }
                )
        return rows

    def _build_inputs(self, task_name: str, prompt: str):
        prompt = build_user_prompt(task_name, prompt)
        if getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if self.model_family == "qwen3.5":
                kwargs["enable_thinking"] = False
            prompt_text = self.tokenizer.apply_chat_template(messages, **kwargs)
        else:
            prompt_text = SYSTEM_PROMPT + "\n\n" + prompt
        return self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)

    def _has_template_issue(self, raw: str, extracted: str) -> bool:
        text = f"{raw}\n{extracted}"
        bad_phrases = [
            "Key evidence",
            "same-task distractors",
            "supported answer",
            "matches the supported",
            "<YOUR ANSWER HERE>",
        ]
        bad_defaults = ["rs12345", "ENSG00000123456", '{"OMIM_ID": "123456"}']
        return any(p in text for p in bad_phrases + bad_defaults)

    def _generate_one(self, model, row: dict) -> tuple[str, str, float, bool]:
        task_name = row["task_name"]
        inputs = self._build_inputs(task_name, row["prompt"])
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=task_max_new_tokens(task_name, self.max_new_tokens),
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        raw = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        extracted = extract_answer(raw, task_name, row["prompt"])
        score = compute_score(task_name, extracted, row["answer"])
        return raw, extracted, score, self._has_template_issue(raw, extracted)

    def _run_benchmark(self, model, state) -> dict:
        was_training = model.training
        model.eval()
        details = []
        task_stats = {}
        correct = 0.0
        template_issues = 0
        try:
            for row in self.rows:
                raw, extracted, score, has_issue = self._generate_one(model, row)
                correct += score
                template_issues += int(has_issue)
                stats = task_stats.setdefault(row["task_name"], {"correct": 0.0, "total": 0})
                stats["correct"] += score
                stats["total"] += 1
                details.append(
                    {
                        "task_name": row["task_name"],
                        "task_instance_id": row["task_instance_id"],
                        "score": score,
                        "extracted": extracted,
                        "ground_truth": row["answer"],
                        "template_issue": has_issue,
                        "raw_preview": raw[:300],
                    }
                )
        finally:
            if was_training:
                model.train()
        total = len(self.rows) or 1
        accuracy = correct / total
        template_issue_rate = template_issues / total
        task_summary = {
            task: {
                "accuracy": round(values["correct"] / values["total"], 4),
                "correct": values["correct"],
                "total": values["total"],
            }
            for task, values in sorted(task_stats.items())
        }
        composite_score = accuracy - template_issue_rate
        event = {
            "run_stem": self.run_stem,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "global_step": int(state.global_step),
            "max_steps": int(state.max_steps),
            "accuracy": round(accuracy, 6),
            "template_issue_rate": round(template_issue_rate, 6),
            "composite_score": round(composite_score, 6),
            "num_examples": len(self.rows),
            "task_summary": task_summary,
            "details": details,
        }
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        tmp = self.summary_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.summary_path)
        return event

    def _should_stop(self, event: dict) -> tuple[bool, str]:
        score = event["composite_score"]
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.bad_rounds = 0
        else:
            self.bad_rounds += 1
        if self.patience > 0 and self.bad_rounds >= self.patience:
            return True, f"benchmark composite score failed to improve for {self.bad_rounds} checks"
        if event["template_issue_rate"] >= self.template_issue_threshold:
            self.bad_rounds += 1
            if self.patience > 0 and self.bad_rounds >= self.patience:
                return True, f"template issue rate too high: {event['template_issue_rate']}"
        return False, ""

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self.eval_steps or not self.rows or model is None:
            return control
        step = int(state.global_step)
        if step <= 0 or step < self.start_step:
            return control
        if step == self.last_eval_step or step % self.eval_steps != 0:
            return control
        self.last_eval_step = step
        event = self._run_benchmark(model, state)
        stop, reason = self._should_stop(event)
        msg = (
            f"step={step}/{state.max_steps}\n"
            f"accuracy={event['accuracy']:.4f}\n"
            f"template_issue_rate={event['template_issue_rate']:.4f}\n"
            f"composite={event['composite_score']:.4f}\n"
            f"best={self.best_score:.4f}\n"
            f"bad_rounds={self.bad_rounds}/{self.patience}\n"
            f"output={self.output_dir}"
        )
        send_autodl_message("Biomni benchmark check", self.run_stem, msg)
        if stop:
            control.should_training_stop = True
            send_autodl_message("Biomni early stopping", self.run_stem, f"{reason}\n{msg}")
        return control


def default_train_path() -> str:
    return str(DATA_DIR / "sft" / "sft_train.jsonl")


def default_mixture_paths() -> list[str]:
    return [
        str(DATA_DIR / "sft_d1_datalake_v2" / "d1_datalake_train.jsonl"),
        str(DATA_DIR / "sft_d2_contrast_v1" / "d2_contrast_train.jsonl"),
        str(DATA_DIR / "sft_d3_verifiable_v1" / "d3_verifiable_sft_train.jsonl"),
    ]


def build_train_stem(model_family: str, timestamp: str, peft_method: str) -> str:
    return f"{timestamp}_{model_family}_{peft_method}_sft"


def build_default_output_dir(model_family: str, timestamp: str, peft_method: str) -> str:
    return str(SCRIPT_OUTPUT_DIR / build_train_stem(model_family, timestamp, peft_method))


def normalize_train_files(train_file: str | Iterable[str] | None) -> list[str]:
    if train_file is None:
        return [default_train_path()]
    if isinstance(train_file, (list, tuple)):
        return [str(p) for p in train_file]
    return [str(train_file)]


def require_package(import_name: str, install_hint: str):
    try:
        return __import__(import_name)
    except Exception as exc:
        raise RuntimeError(
            f"Missing dependency `{import_name}` for PEFT training. Install it first, for example: {install_hint}"
        ) from exc


def parse_target_modules(value: str | list[str] | tuple[str, ...] | None) -> list[str] | str:
    if value is None or value == "auto":
        return "auto"
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def infer_target_modules(model_family: str, target_modules: str | list[str] | None) -> list[str]:
    parsed = parse_target_modules(target_modules)
    if parsed != "auto":
        return parsed
    if model_family.startswith("qwen") or model_family.startswith("llama") or model_family == "mistral":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "v_proj"]


def resolve_torch_dtype(dtype: str):
    dtype = (dtype or "auto").lower()
    if dtype == "auto":
        return "auto"
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def messages_to_text(messages, tokenizer=None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    parts = []
    for msg in messages:
        role = msg["role"]
        parts.append(f"<|{role}|>\n{msg['content']}\n")
    return "".join(parts)


def load_training_dataset(train_files: list[str], seed: int | None = None):
    missing = [p for p in train_files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Training file(s) not found: {missing}")
    datasets_list = [datasets.Dataset.from_json(path) for path in train_files]
    if len(datasets_list) == 1:
        ds = datasets_list[0]
    else:
        ds = datasets.concatenate_datasets(datasets_list)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    return ds


def build_model(model_dir: str, trust_remote_code: bool, peft_method: str, torch_dtype: str,
                bnb_4bit_quant_type: str, bnb_4bit_compute_dtype: str,
                model_max_memory_gpu: str | None = None, model_max_memory_cpu: str | None = None):
    peft_method = peft_method.lower()
    dtype = resolve_torch_dtype(torch_dtype)
    kwargs = {"trust_remote_code": trust_remote_code}
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["torch_dtype"] = "auto"
    if peft_method == "qlora":
        require_package("bitsandbytes", "pip install bitsandbytes")
        from transformers import BitsAndBytesConfig
        compute_dtype = resolve_torch_dtype(bnb_4bit_compute_dtype)
        if compute_dtype == "auto":
            compute_dtype = torch.bfloat16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["device_map"] = "auto"
        kwargs["low_cpu_mem_usage"] = True
        max_memory = {}
        if model_max_memory_gpu:
            gpu_limits = [part.strip() for part in str(model_max_memory_gpu).split(",") if part.strip()]
            if len(gpu_limits) == 1 and torch.cuda.is_available():
                gpu_limits = gpu_limits * torch.cuda.device_count()
            for idx, limit in enumerate(gpu_limits):
                max_memory[idx] = limit
        if model_max_memory_cpu:
            max_memory["cpu"] = model_max_memory_cpu
        if max_memory:
            kwargs["max_memory"] = max_memory
    return AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)


def apply_peft(model, model_family: str, peft_method: str, target_modules: str | list[str] | None,
               lora_r: int, lora_alpha: int, lora_dropout: float):
    require_package("peft", "pip install peft")
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if peft_method.lower() == "qlora":
        model = prepare_model_for_kbit_training(model)
    modules = infer_target_modules(model_family, target_modules)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=modules,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model, modules


def train_sft(model_family: str, train_file: str | list[str] | None = None, output_dir: str | None = None,
              epochs: float = 1.0, batch_size: int = 1, learning_rate: float = 2e-4,
              max_seq_length: int = 1024, test_size: float = 0.02,
              gradient_accumulation_steps: int = 8, save_strategy: str = "steps",
              shutdown_after: bool = False, config_path: str | None = None,
              peft_method: str = "qlora", lora_r: int = 16, lora_alpha: int = 32,
              lora_dropout: float = 0.05, target_modules: str | list[str] | None = "auto",
              torch_dtype: str = "auto", bnb_4bit_quant_type: str = "nf4",
              bnb_4bit_compute_dtype: str = "bf16", warmup_ratio: float = 0.03,
              weight_decay: float = 0.0, logging_steps: int = 20, save_steps: int = 500,
              eval_strategy: str = "steps", eval_steps: int = 500,
              save_total_limit: int = 1, seed: int = 42, dry_run: bool = False,
              gradient_checkpointing: bool = True, dataloader_num_workers: int = 0,
              load_best_model_at_end: bool = True, metric_for_best_model: str = "eval_loss",
              greater_is_better: bool = False, notify_interval_seconds: int = 3600,
              benchmark_eval_steps: int = 0, benchmark_eval_dataset: str | None = None,
              benchmark_eval_subset_per_task: int = 3, benchmark_eval_start_step: int = 0,
              benchmark_eval_patience: int = 2, benchmark_eval_min_delta: float = 0.0,
              benchmark_eval_max_new_tokens: int = 96, benchmark_template_issue_threshold: float = 0.25,
              model_max_memory_gpu: str | None = None, model_max_memory_cpu: str | None = None):
    peft_method = peft_method.lower()
    if peft_method not in {"lora", "qlora"}:
        raise ValueError("peft_method must be either 'lora' or 'qlora'")

    spec = get_local_model_spec(model_family)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_stem = build_train_stem(model_family, timestamp, peft_method)
    output_dir = output_dir or build_default_output_dir(model_family, timestamp, peft_method)
    os.makedirs(output_dir, exist_ok=True)

    train_files = normalize_train_files(train_file)
    target_modules_resolved = infer_target_modules(model_family, target_modules)
    metadata = {
        "run_stem": run_stem,
        "model_family": model_family,
        "model_label": spec.label,
        "base_model_dir": spec.base_model_dir,
        "train_files": train_files,
        "output_dir": output_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "test_size": test_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "save_strategy": save_strategy,
        "shutdown_after": shutdown_after,
        "peft_method": peft_method,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules_resolved,
        "torch_dtype": torch_dtype,
        "bnb_4bit_quant_type": bnb_4bit_quant_type,
        "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "save_total_limit": save_total_limit,
        "seed": seed,
        "config_path": config_path,
        "timestamp": timestamp,
        "dry_run": dry_run,
        "gradient_checkpointing": gradient_checkpointing,
        "dataloader_num_workers": dataloader_num_workers,
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "notify_interval_seconds": notify_interval_seconds,
        "benchmark_eval_steps": benchmark_eval_steps,
        "benchmark_eval_dataset": benchmark_eval_dataset or default_dataset_path(),
        "benchmark_eval_subset_per_task": benchmark_eval_subset_per_task,
        "benchmark_eval_start_step": benchmark_eval_start_step,
        "benchmark_eval_patience": benchmark_eval_patience,
        "benchmark_eval_min_delta": benchmark_eval_min_delta,
        "benchmark_eval_max_new_tokens": benchmark_eval_max_new_tokens,
        "benchmark_template_issue_threshold": benchmark_template_issue_threshold,
        "model_max_memory_gpu": model_max_memory_gpu,
        "model_max_memory_cpu": model_max_memory_cpu,
    }
    metadata_path = os.path.join(output_dir, f"{run_stem}_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Prepared {peft_method.upper()} SFT run for {model_family}")
    print(f"Run stem: {run_stem}")
    print(f"Output dir: {output_dir}")
    print(f"Train files: {train_files}")
    print(f"Metadata: {metadata_path}")
    if dry_run:
        for path in train_files:
            if not Path(path).exists():
                raise FileNotFoundError(path)
        print("Dry run complete. No model or dataset was loaded, and training was not started.")
        return metadata

    tokenizer = AutoTokenizer.from_pretrained(spec.base_model_dir, trust_remote_code=spec.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = build_model(
        spec.base_model_dir,
        spec.trust_remote_code,
        peft_method,
        torch_dtype,
        bnb_4bit_quant_type,
        bnb_4bit_compute_dtype,
        model_max_memory_gpu,
        model_max_memory_cpu,
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model, target_modules_resolved = apply_peft(
        model, model_family, peft_method, target_modules, lora_r, lora_alpha, lora_dropout
    )

    ds = load_training_dataset(train_files, seed=seed).train_test_split(test_size=test_size, seed=seed)

    def preprocess(example):
        text = messages_to_text(example["messages"], tokenizer=tokenizer)
        result = tokenizer(text, truncation=True, padding="max_length", max_length=max_seq_length)
        result["labels"] = result["input_ids"].copy()
        return result

    train_ds = ds["train"].map(preprocess, remove_columns=ds["train"].column_names, batched=False)
    eval_ds = ds["test"].map(preprocess, remove_columns=ds["test"].column_names, batched=False)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        remove_unused_columns=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        report_to="none",
        seed=seed,
        dataloader_num_workers=dataloader_num_workers,
        bf16=(bnb_4bit_compute_dtype.lower() in {"bf16", "bfloat16"} or torch_dtype.lower() in {"bf16", "bfloat16"}),
        fp16=(bnb_4bit_compute_dtype.lower() in {"fp16", "float16"} or torch_dtype.lower() in {"fp16", "float16"}),
    )

    live_callback = LiveTrainingStatusCallback(output_dir, run_stem, metadata, notify_interval_seconds=notify_interval_seconds)
    callbacks = [live_callback]
    if benchmark_eval_steps and benchmark_eval_steps > 0:
        callbacks.append(
            PeriodicBenchmarkCallback(
                model_family=model_family,
                tokenizer=tokenizer,
                output_dir=output_dir,
                run_stem=run_stem,
                dataset_path=benchmark_eval_dataset or default_dataset_path(),
                eval_steps=benchmark_eval_steps,
                subset_per_task=benchmark_eval_subset_per_task,
                start_step=benchmark_eval_start_step,
                patience=benchmark_eval_patience,
                min_delta=benchmark_eval_min_delta,
                max_new_tokens=benchmark_eval_max_new_tokens,
                template_issue_threshold=benchmark_template_issue_threshold,
                seed=seed,
            )
        )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    print(f"Starting {peft_method.upper()} SFT for {model_family} -> {output_dir}")
    trainer.train()
    log_history_path = os.path.join(output_dir, f"{run_stem}_train_log.json")
    with open(log_history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "train_log_history.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
    if torch.cuda.is_available():
        print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
        print(f"Peak GPU memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training finished. Logs: {log_history_path}")
    if shutdown_after:
        send_autodl_message("Biomni??????", run_stem, f"?????????? /usr/bin/shutdown?\noutput={output_dir}")
        os.system("/usr/bin/shutdown -h now")
    return metadata
