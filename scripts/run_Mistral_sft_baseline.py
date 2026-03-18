"""
Mistral-7B-Instruct SFT微调模型基准评测脚本
本脚本仿照 run_qwen2.5_sft_baseline.py，读取经过SFT训练后的 Mistral-7B-Instruct 模型。
"""

import json
import os
import re
import time
import traceback
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 配置：训练后的Mistral模型路径（与训练脚本 OUTPUT_DIR 保持一致）
LOCAL_MODEL_DIR = "/root/autodl-tmp/Biomni-main/scripts/output/mistral_sft"
OUTPUT_DIR = "../results/mistral_sft"
LOCAL_PARQUET = "../data/biomni_eval1_dataset.parquet"
MAX_RETRIES = 3
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 推理生成参数（可自行调整）
GEN_MAX_NEW_TOKENS = 512
GEN_TEMPERATURE = 0.0
GEN_DO_SAMPLE = False
GEN_TOP_P = 0.95
GEN_TOP_K = 50
SLEEP_BETWEEN = 0.1

print(f"Using local finetuned model dir: {LOCAL_MODEL_DIR}")
print("Loading tokenizer and model (this may take a while)...")

# 加载 tokenizer 和 model（device_map="auto"、float16，兼容大显存）
try:
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
except Exception as e:
    print("Failed to load tokenizer:", e)
    raise

model = None
try:
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("Model loaded with device_map='auto' and torch_dtype=float16.")
except Exception as e:
    print("device_map auto load failed:", e)
    print("Fallback to low_cpu_mem_usage=True...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_DIR,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            model.to("cuda")
        print("Fallback model load succeeded.")
    except Exception as e2:
        print("Fallback load failed:", e2)
        raise

SYSTEM_PROMPT = """You are a helpful biologist and expert geneticist.
You are given a biomedical question. Analyze it carefully and provide your answer.

IMPORTANT: At the very end of your response, clearly state your final answer on a new line in this exact format:
FINAL ANSWER: <your answer here>

For multiple choice questions, FINAL ANSWER should be just the letter (e.g., FINAL ANSWER: a).
For gene identification, FINAL ANSWER should be just the gene symbol (e.g., FINAL ANSWER: BRCA1).
For variant questions, FINAL ANSWER should be the variant ID (e.g., FINAL ANSWER: rs12345).
For disease diagnosis, FINAL ANSWER should be the OMIM ID as JSON (e.g., FINAL ANSWER: {"OMIM_ID": "123456"}).
For patient gene detection, FINAL ANSWER should be JSON with gene list (e.g., FINAL ANSWER: {"causal_gene": ["GENE1"]}).
"""

def extract_answer(raw: str, task_name: str) -> str:
    match = re.search(r'FINAL ANSWER:\s*(.+)', raw, re.IGNORECASE)
    extracted = match.group(1).strip().strip(".") if match else raw.strip()
    if task_name == "crispr_delivery":
        m = re.search(r'\b([a-fA-F])\b', extracted)
        return m.group(1).lower() if m else extracted[:1].lower()
    elif task_name in ("lab_bench_dbqa", "lab_bench_seqqa"):
        m = re.search(r'\b([A-Za-z])\b', extracted)
        return m.group(1).upper() if m else extracted[:1].upper()
    elif task_name.startswith("gwas_causal_gene") or task_name == "screen_gene_retrieval":
        m = re.search(r'\b([A-Z][A-Z0-9]{1,15})\b', extracted)
        return m.group(1) if m else extracted.upper().strip()
    elif task_name == "gwas_variant_prioritization":
        m = re.search(r'(rs\d+)', extracted)
        return m.group(1) if m else extracted.strip()
    elif task_name == "rare_disease_diagnosis":
        m = re.search(r'(\d{6})', extracted)
        return json.dumps({"OMIM_ID": m.group(1)}) if m else extracted
    elif task_name == "patient_gene_detection":
        try:
            parsed = json.loads(extracted)
            if "causal_gene" in parsed:
                return extracted
        except Exception:
            pass
        genes = re.findall(r'\b([A-Z][A-Z0-9]{1,10})\b', extracted)
        return json.dumps({"causal_gene": genes[:5]}) if genes else extracted
    return extracted

def compute_score(task_name: str, user_answer: str, ground_truth: str) -> float:
    try:
        if task_name == "crispr_delivery":
            return 1.0 if user_answer.strip().lower() == ground_truth.strip().lower() else 0.0
        elif task_name.startswith("gwas_causal_gene"):
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0
        elif task_name == "gwas_variant_prioritization":
            return 1.0 if user_answer.strip() == ground_truth.strip() else 0.0
        elif task_name in ("lab_bench_dbqa", "lab_bench_seqqa"):
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0
        elif task_name == "screen_gene_retrieval":
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0
        elif task_name == "rare_disease_diagnosis":
            if isinstance(user_answer, str):
                try:
                    user_dict = json.loads(user_answer)
                except Exception:
                    return 0.0
            else:
                user_dict = user_answer
            gt_dict = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
            return 1.0 if user_dict.get("OMIM_ID") == gt_dict.get("OMIM_ID") else 0.0
        elif task_name == "patient_gene_detection":
            if isinstance(user_answer, str):
                try:
                    user_dict = json.loads(user_answer)
                except Exception:
                    return 0.0
            else:
                user_dict = user_answer
            predicted = user_dict.get("causal_gene", [])
            if not isinstance(predicted, list):
                predicted = [predicted]
            true_genes = [g.strip() for g in ground_truth.split(",")] if "," in ground_truth else [ground_truth]
            return 1.0 if predicted and set(true_genes) & set(predicted) else 0.0
        else:
            return 1.0 if user_answer.strip() == ground_truth.strip() else 0.0
    except Exception:
        return 0.0

def run_single_local(task_name: str, prompt: str) -> str:
    full_prompt = SYSTEM_PROMPT + "\n" + prompt
    try:
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen_kwargs = dict(
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=GEN_DO_SAMPLE,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            top_k=GEN_TOP_K,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        for attempt in range(MAX_RETRIES):
            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if text.startswith(full_prompt):
                    text = text[len(full_prompt):].strip()
                return text
            except RuntimeError as e:
                torch.cuda.empty_cache()
                time.sleep(2 * (attempt + 1))
        return ""
    except Exception as e:
        traceback.print_exc()
        return ""

def _save_checkpoint(task_name, results):
    path = f"{OUTPUT_DIR}/_ckpt_{task_name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    print("=" * 60)
    print("Finetuned Mistral-7B-Instruct Baseline Evaluation")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    # 加载数据
    df = pd.read_parquet(LOCAL_PARQUET)
    tasks = sorted(df["task_name"].unique())
    print("Tasks:", tasks)
    print("Total instances:", len(df))

    all_results = {}
    summary = {}
    total_correct = 0
    total_count = 0

    for task_name in tasks:
        print(f"\n{'='*60}")
        print("Task:", task_name)
        print(f"{'='*60}")

        task_df = df[(df["task_name"] == task_name) & (df["split"] == "val")]
        n = len(task_df)
        print("Instances:", n)
        if n == 0:
            continue

        task_results = []
        task_correct = 0

        for i, (_, row) in enumerate(task_df.iterrows()):
            print(f"  [{i+1}/{n}] ID={row['task_instance_id']}...", end=" ", flush=True)
            raw_answer = run_single_local(task_name, row["prompt"])
            extracted = extract_answer(raw_answer, task_name) if raw_answer else ""
            score = compute_score(task_name, extracted, row["answer"])
            task_correct += int(score >= 1.0)
            emoji = "✅" if score >= 1.0 else "❌"
            print(f"{emoji} answer={str(extracted)[:30]:<30} truth={str(row['answer'])[:30]}")
            task_results.append({
                "task_instance_id": int(row["task_instance_id"]),
                "score": score,
                "extracted": extracted,
                "ground_truth": row["answer"],
                "raw_preview": raw_answer[:300] if raw_answer else "",
            })
            _save_checkpoint(task_name, task_results)
            time.sleep(SLEEP_BETWEEN)

        acc = task_correct / n
        summary[task_name] = {
            "accuracy": round(acc, 4),
            "correct": task_correct,
            "total": n,
        }
        all_results[task_name] = task_results
        total_correct += task_correct
        total_count += n
        print(f"\n  >>> {task_name}: {acc:.3f} ({task_correct}/{n})")

    # 保存结果
    overall_acc = total_correct / total_count if total_count else 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary["_overall"] = {
        "accuracy": round(overall_acc, 4),
        "correct": total_correct,
        "total": total_count,
        "model": LOCAL_MODEL_DIR,
        "timestamp": timestamp,
    }

    with open(f"{OUTPUT_DIR}/detail_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(f"{OUTPUT_DIR}/summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印终表格
    print("\n\n" + "=" * 60)
    print("FINAL RESULTS — Finetuned Mistral Local Model")
    print("=" * 60)
    print(f"{'Task':<42} {'Acc':>7} {'Correct':>8} {'Total':>6}")
    print("-" * 65)
    for task_name in tasks:
        if task_name in summary:
            s = summary[task_name]
            print(f"{task_name:<42} {s['accuracy']:>6.3f} {s['correct']:>7d} {s['total']:>5d}")
    print("-" * 65)
    print(f"{'OVERALL':<42} {overall_acc:>6.3f} {total_correct:>7d} {total_count:>5d}")
    print(f"\nSaved to: {OUTPUT_DIR}/summary_{timestamp}.json")

if __name__ == "__main__":
    main()