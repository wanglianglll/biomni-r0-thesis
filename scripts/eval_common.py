from __future__ import annotations

import json
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.model_registry import DATA_DIR, get_local_model_spec

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - PEFT is only required for adapter evaluation.
    PeftModel = None

SYSTEM_PROMPT = (
    "You are a biomedical assistant. Follow the requested answer format strictly. "
    "Do not output placeholder text such as <YOUR ANSWER HERE> or <your answer here>. "
    "Do not repeat the question, do not explain your reasoning, and do not output sample answers."
)

TASK_FORMAT_HINTS = {
    "crispr_delivery": "Choose exactly one option letter from the listed options a-f. End with exactly one line starting with FINAL ANSWER: followed by the chosen letter.",
    "lab_bench_dbqa": "Choose exactly one option letter from the listed options A-F. End with exactly one line starting with FINAL ANSWER: followed by the chosen letter.",
    "lab_bench_seqqa": "Choose exactly one option letter from the listed options A-F. End with exactly one line starting with FINAL ANSWER: followed by the chosen letter.",
    "gwas_variant_prioritization": "Copy exactly one rsID from the provided variant list. End with exactly one line starting with FINAL ANSWER: followed by that rsID.",
    "rare_disease_diagnosis": "Return a JSON object containing the predicted OMIM_ID from the evidence. End with exactly one line starting with FINAL ANSWER: followed by the JSON.",
    "patient_gene_detection": "Pick exactly one gene from the provided candidate genes. End with exactly one line starting with FINAL ANSWER: followed by a JSON object with key causal_gene and a one-item list.",
}


def default_dataset_path() -> str:
    return str(DATA_DIR / "biomni_eval1_dataset.parquet")


def build_run_stem(model_family: str, variant: str, timestamp: str) -> str:
    return f"{timestamp}_{model_family}_{variant}"


def task_format_hint(task_name: str) -> str:
    if task_name in TASK_FORMAT_HINTS:
        return TASK_FORMAT_HINTS[task_name]
    if task_name.startswith("gwas_causal_gene") or task_name == "screen_gene_retrieval":
        return "Copy exactly one gene symbol from the provided candidate gene list. End with exactly one line starting with FINAL ANSWER: followed by that gene."
    return "Reply with exactly one final line starting with FINAL ANSWER:"


def build_user_prompt(task_name: str, prompt: str) -> str:
    extra = "Do not add any extra line after the final answer."
    if task_name in ("crispr_delivery", "lab_bench_dbqa", "lab_bench_seqqa"):
        extra += " Do not write words before the final answer line."
    if task_name == "patient_gene_detection":
        extra += " Do not mention HPO or HP codes in the final answer."
    if task_name == "rare_disease_diagnosis":
        extra += " Output only the OMIM_ID JSON on the final answer line."
    return f"{prompt.rstrip()}\n\n{task_format_hint(task_name)}\n{extra}"


def normalize_raw_output(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return text
    text = re.sub(r"^\s*assistant\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("<|im_end|>", " ").strip()
    return text


def _find_answer_tags(text: str) -> list[str]:
    return re.findall(r"\[ANSWER\]\s*(.*?)\s*\[/ANSWER\]", text, flags=re.IGNORECASE | re.DOTALL)


def _find_final_answers(text: str) -> list[str]:
    return [m.strip() for m in re.findall(r"FINAL ANSWER:\s*(.+?)(?=\nFINAL ANSWER:|$)", text, flags=re.IGNORECASE | re.DOTALL) if m.strip()]


def _candidate_genes_from_text(text: str) -> list[str]:
    match = re.search(r"(?:Genes in locus|Candidate genes):\s*(.+?)(?:\n\n|\n[A-Z][A-Za-z ]+:|$)", text, flags=re.DOTALL)
    if not match:
        return []
    block = match.group(1)
    return re.findall(r"ENSG\d{11}|[A-Z][A-Z0-9-]{1,25}", block)


def _candidate_variants_from_text(text: str) -> list[str]:
    match = re.search(r"Variants:\s*(.+?)(?:\n\n|\n[A-Z][A-Za-z ]+:|$)", text, flags=re.DOTALL)
    block = match.group(1) if match else text
    return re.findall(r"rs\d+", block, flags=re.IGNORECASE)


def _snap_to_candidate(value: str, candidates: list[str]) -> str:
    if not value or not candidates:
        return value
    exact = {c.upper(): c for c in candidates}
    key = value.upper()
    if key in exact:
        return exact[key]
    # Generation often appends repetitive suffixes, e.g. SUCLG2A2B2 or KEAP1B1-AS1.
    pref = [c for c in candidates if key.startswith(c.upper())]
    if pref:
        return max(pref, key=len)
    contained = [c for c in candidates if c.upper() in key]
    if contained:
        return max(contained, key=len)
    return value


def _extract_choice_answer(text: str, choices: str) -> str:
    for candidate in _find_answer_tags(text):
        m = re.search(rf"\b([{choices}])\b", candidate, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    for candidate in _find_final_answers(text):
        m = re.search(rf"\b([{choices}])\b", candidate, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    tail = text[-240:]
    labels = re.findall(rf"(?:^|[:\s\[(])([{choices}])(?:[\]\)\s,.;]|$)", tail, flags=re.IGNORECASE | re.MULTILINE)
    return labels[-1] if labels else ""


def _extract_gene_like_answer(text: str) -> str:
    for candidate in _find_final_answers(text):
        m = re.search(r"\b([A-Z][A-Z0-9-]{1,20})\b", candidate)
        if m:
            return m.group(1)
    m = re.search(r"\b([A-Z][A-Z0-9-]{1,20})\b", text)
    return m.group(1) if m else ""


def _extract_variant_answer(text: str) -> str:
    for candidate in _find_final_answers(text):
        m = re.search(r"(rs\d+)", candidate, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    m = re.search(r"(rs\d+)", text, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def _extract_omim_answer(text: str) -> str:
    for candidate in _find_final_answers(text) + [text]:
        m = re.search(r'"OMIM_ID"\s*:\s*"?(\d{6})"?', candidate)
        if m:
            return json.dumps({"OMIM_ID": m.group(1)})
        m = re.search(r"'OMIM_ID'\s*:\s*'?(\d{6})'?", candidate)
        if m:
            return json.dumps({"OMIM_ID": m.group(1)})
        m = re.search(r"\b(\d{6})\b", candidate)
        if m:
            return json.dumps({"OMIM_ID": m.group(1)})
    return ""


def _extract_patient_gene_answer(text: str) -> str:
    blacklist = {"HP", "HPO", "FINAL", "ANSWER", "OMIM"}
    for candidate in _find_final_answers(text) + [text]:
        m = re.search(r'"causal_gene"\s*:\s*\[(.*?)\]', candidate, flags=re.DOTALL)
        if not m:
            m = re.search(r"'causal_gene'\s*:\s*\[(.*?)\]", candidate, flags=re.DOTALL)
        if m:
            genes = re.findall(r"ENSG\d{11}|[A-Z][A-Z0-9-]{1,15}", m.group(1))
            genes = [g for g in genes if g not in blacklist]
            if genes:
                return json.dumps({"causal_gene": genes[:5]})
    ensgs = re.findall(r"ENSG\d{11}", text)
    if ensgs:
        return json.dumps({"causal_gene": [ensgs[-1]]})
    genes = re.findall(r"\b([A-Z][A-Z0-9-]{1,15})\b", text)
    genes = [g for g in genes if g not in blacklist]
    return json.dumps({"causal_gene": genes[:5]}) if genes else ""


def extract_answer(raw: str, task_name: str, prompt: str | None = None) -> str:
    raw = normalize_raw_output(raw)
    if task_name == "crispr_delivery":
        choice = _extract_choice_answer(raw, "a-fA-F")
        return choice.lower() if choice else ""
    if task_name in ("lab_bench_dbqa", "lab_bench_seqqa"):
        choice = _extract_choice_answer(raw, "A-Fa-f")
        return choice.upper() if choice else ""
    if task_name.startswith("gwas_causal_gene") or task_name == "screen_gene_retrieval":
        value = _extract_gene_like_answer(raw)
        return _snap_to_candidate(value, _candidate_genes_from_text(prompt or "")) if prompt else value
    if task_name == "gwas_variant_prioritization":
        value = _extract_variant_answer(raw)
        return _snap_to_candidate(value, _candidate_variants_from_text(prompt or "")) if prompt else value
    if task_name == "rare_disease_diagnosis":
        return _extract_omim_answer(raw)
    if task_name == "patient_gene_detection":
        value = _extract_patient_gene_answer(raw)
        if prompt:
            candidates = _candidate_genes_from_text(prompt)
            try:
                parsed = json.loads(value)
                genes = parsed.get("causal_gene") or []
                if genes:
                    return json.dumps({"causal_gene": [_snap_to_candidate(str(genes[0]), candidates)]})
            except Exception:
                pass
        return value
    match = re.search(r"FINAL ANSWER:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    extracted = match.group(1).strip() if match else raw.strip()
    extracted = extracted.splitlines()[0].strip().strip('.')
    return "" if extracted.upper() in {"<YOUR ANSWER HERE>", "YOUR ANSWER HERE"} else extracted


def compute_score(task_name: str, user_answer: str, ground_truth: str) -> float:
    try:
        if task_name == "crispr_delivery":
            return 1.0 if user_answer.strip().lower() == ground_truth.strip().lower() else 0.0
        if task_name.startswith("gwas_causal_gene"):
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0
        if task_name == "gwas_variant_prioritization":
            return 1.0 if user_answer.strip() == ground_truth.strip() else 0.0
        if task_name in ("lab_bench_dbqa", "lab_bench_seqqa"):
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0
        if task_name == "screen_gene_retrieval":
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0
        if task_name == "rare_disease_diagnosis":
            user_dict = json.loads(user_answer) if isinstance(user_answer, str) else user_answer
            gt_dict = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
            return 1.0 if user_dict.get("OMIM_ID") == gt_dict.get("OMIM_ID") else 0.0
        if task_name == "patient_gene_detection":
            user_dict = json.loads(user_answer) if isinstance(user_answer, str) else user_answer
            predicted = user_dict.get("causal_gene", [])
            if not isinstance(predicted, list):
                predicted = [predicted]
            true_genes = [g.strip() for g in ground_truth.split(",")] if "," in ground_truth else [ground_truth]
            return 1.0 if predicted and set(true_genes) & set(predicted) else 0.0
        return 1.0 if user_answer.strip() == ground_truth.strip() else 0.0
    except Exception:
        return 0.0


def task_max_new_tokens(task_name: str, requested: int) -> int:
    caps = {
        "crispr_delivery": 48,
        "lab_bench_dbqa": 64,
        "lab_bench_seqqa": 64,
        "gwas_variant_prioritization": 32,
        "rare_disease_diagnosis": 96,
        "patient_gene_detection": 128,
        "screen_gene_retrieval": 48,
    }
    if task_name.startswith("gwas_causal_gene"):
        return min(requested, 48)
    return min(requested, caps.get(task_name, requested))


class LocalGenerator:
    def __init__(self, model_family: str, model_dir: str, trust_remote_code: bool, max_retries: int = 3,
                 adapter_dir: str | None = None):
        self.model_family = model_family
        self.model_dir = model_dir
        self.trust_remote_code = trust_remote_code
        self.max_retries = max_retries
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=trust_remote_code,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
            if torch.cuda.is_available():
                self.model.to("cuda")
        if adapter_dir:
            if PeftModel is None:
                raise RuntimeError("PEFT is required to evaluate an adapter, but peft is not importable.")
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()

    def _build_inputs(self, prompt: str):
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
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
        return prompt_text, inputs

    def __call__(self, task_name: str, prompt: str, gen_cfg: dict) -> str:
        prompt = build_user_prompt(task_name, prompt)
        try:
            _, inputs = self._build_inputs(prompt)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            kwargs = dict(
                max_new_tokens=task_max_new_tokens(task_name, gen_cfg["max_new_tokens"]),
                do_sample=gen_cfg["do_sample"],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if gen_cfg["do_sample"]:
                kwargs.update(
                    temperature=gen_cfg["temperature"],
                    top_p=gen_cfg["top_p"],
                    top_k=gen_cfg["top_k"],
                )
            input_len = inputs["input_ids"].shape[1]
            for attempt in range(self.max_retries):
                try:
                    with torch.inference_mode():
                        outputs = self.model.generate(**inputs, **kwargs)
                    new_tokens = outputs[0][input_len:]
                    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    return normalize_raw_output(text)
                except RuntimeError:
                    torch.cuda.empty_cache()
                    time.sleep(2 * (attempt + 1))
            return ""
        except Exception:
            traceback.print_exc()
            return ""


class DeepSeekGenerator:
    def __init__(self, temperature: float, max_new_tokens: int, max_retries: int = 3):
        load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
        for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
            os.environ.pop(key, None)
        self.max_retries = max_retries
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=temperature,
            max_tokens=max_new_tokens,
            base_url=os.getenv("CUSTOM_MODEL_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("CUSTOM_MODEL_API_KEY"),
            timeout=120,
            max_retries=max_retries,
        )

    def __call__(self, task_name: str, prompt: str, _gen_cfg: dict) -> str:
        prompt = build_user_prompt(task_name, prompt)
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(messages)
                return response.content
            except Exception as exc:
                print(f"      Retry {attempt + 1}/{self.max_retries}: {exc}")
                time.sleep(3 * (attempt + 1))
        return ""


def build_generator(model_family: str, variant: str, max_retries: int, temperature: float, max_new_tokens: int):
    if model_family == "deepseek":
        if variant != "base":
            raise ValueError("DeepSeek currently only supports variant='base'.")
        return DeepSeekGenerator(temperature=temperature, max_new_tokens=max_new_tokens, max_retries=max_retries), "deepseek-chat (DeepSeek-V3)"
    spec = get_local_model_spec(model_family)
    if variant == "base":
        model_dir = spec.base_model_dir
        adapter_dir = None
    else:
        model_dir = spec.base_model_dir
        adapter_dir = spec.sft_model_dir
    return LocalGenerator(
        model_family=model_family,
        model_dir=model_dir,
        trust_remote_code=spec.trust_remote_code,
        max_retries=max_retries,
        adapter_dir=adapter_dir,
    ), spec.label


def resolve_output_dir(model_family: str, variant: str, override: str | None = None) -> str:
    if override:
        return override
    if model_family == "deepseek":
        return str(Path(__file__).resolve().parent.parent / "results" / "baseline")
    spec = get_local_model_spec(model_family)
    return spec.base_results_dir if variant == "base" else spec.sft_results_dir


def evaluate_model(model_family: str, variant: str, dataset_path: str, output_dir: str, subset: int = 0,
                   max_retries: int = 3, sleep_between: float = 0.1, max_new_tokens: int = 128,
                   temperature: float = 0.0, top_p: float = 0.95, top_k: int = 50, do_sample: bool = False,
                   config_path: str | None = None):
    os.makedirs(output_dir, exist_ok=True)
    generator, model_label = build_generator(model_family, variant, max_retries, temperature, max_new_tokens)
    gen_cfg = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": do_sample,
    }
    df = pd.read_parquet(dataset_path)
    tasks = sorted(df["task_name"].unique())
    all_results: Dict[str, List[dict]] = {}
    summary: Dict[str, dict] = {}
    total_correct = 0
    total_count = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_stem = build_run_stem(model_family, variant, timestamp)
    print("=" * 60)
    print(f"Evaluating {model_family} ({variant})")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Dataset:", dataset_path)
    print("Output:", output_dir)
    print("Run stem:", run_stem)
    print("=" * 60)
    for task_name in tasks:
        print(f"\n{'=' * 60}\nTask: {task_name}\n{'=' * 60}")
        task_df = df[(df["task_name"] == task_name) & (df["split"] == "val")]
        if subset and subset > 0:
            task_df = task_df.head(subset)
        n = len(task_df)
        print("Instances:", n)
        if n == 0:
            continue
        task_results = []
        task_correct = 0
        for i, (_, row) in enumerate(task_df.iterrows()):
            print(f"  [{i+1}/{n}] ID={row['task_instance_id']}...", end=" ", flush=True)
            raw_answer = generator(task_name, row["prompt"], gen_cfg)
            extracted = extract_answer(raw_answer, task_name, row["prompt"]) if raw_answer else ""
            score = compute_score(task_name, extracted, row["answer"])
            task_correct += int(score >= 1.0)
            marker = "OK" if score >= 1.0 else "NO"
            print(f"[{marker}] answer={str(extracted)[:30]:<30} truth={str(row['answer'])[:30]}")
            task_results.append({
                "task_instance_id": int(row["task_instance_id"]),
                "score": score,
                "extracted": extracted,
                "ground_truth": row["answer"],
                "raw_preview": raw_answer[:500] if raw_answer else "",
            })
            if (i + 1) % 10 == 0 or i + 1 == n:
                with open(os.path.join(output_dir, f"{run_stem}_ckpt_{task_name}.json"), "w", encoding="utf-8") as f:
                    json.dump(task_results, f, indent=2, ensure_ascii=False)
            time.sleep(sleep_between)
        acc = task_correct / n
        summary[task_name] = {"accuracy": round(acc, 4), "correct": task_correct, "total": n}
        all_results[task_name] = task_results
        total_correct += task_correct
        total_count += n
        print(f"\n  >>> {task_name}: {acc:.3f} ({task_correct}/{n})")
    overall_acc = total_correct / total_count if total_count else 0
    summary["_overall"] = {
        "accuracy": round(overall_acc, 4),
        "correct": total_correct,
        "total": total_count,
        "model": model_label,
        "timestamp": timestamp,
        "run_stem": run_stem,
        "config_path": config_path,
    }
    with open(os.path.join(output_dir, f"{run_stem}_detail.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, f"{run_stem}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS - {model_family} ({variant})")
    print("=" * 60)
    print(f"{'Task':<42} {'Acc':>7} {'Correct':>8} {'Total':>6}")
    print("-" * 65)
    for task_name in tasks:
        if task_name in summary:
            s = summary[task_name]
            print(f"{task_name:<42} {s['accuracy']:>6.3f} {s['correct']:>7d} {s['total']:>5d}")
    print("-" * 65)
    print(f"{'OVERALL':<42} {overall_acc:>6.3f} {total_correct:>7d} {total_count:>5d}")
    print(f"\nSaved to: {output_dir}/{run_stem}_summary.json")
    return summary
