"""
DeepSeek-V3 基线评测（本地模型版）— 批量跑 BiomniEval1 全部 433 个实例（或本地数据）
将原 run_deepseek_baseline.py 中对远端 API 的调用替换为本地 transformers 推理。
使用前：
 - 把 LOCAL_MODEL_DIR 指向你的微调后模型目录（例如 output/sft_qwen2_7b）
 - 确保有足够显存或使用 device_map="auto" + offload/quantize 等
"""
import json
import os
import re
import time
import traceback
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- config ---
load_dotenv("../.env", override=True)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

OUTPUT_DIR = "../results/baseline"
LOCAL_PARQUET = "../data/biomni_eval1_dataset.parquet"
MAX_RETRIES = 3
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 指定本地微调后模型目录（修改为你的路径）
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "output/sft_qwen2_7b")

# Generation config (可根据需要调整)
GEN_MAX_NEW_TOKENS = int(os.getenv("GEN_MAX_NEW_TOKENS", "512"))
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.0"))
GEN_DO_SAMPLE = os.getenv("GEN_DO_SAMPLE", "false").lower() in ("1", "true", "yes")
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.95"))
GEN_TOP_K = int(os.getenv("GEN_TOP_K", "50"))
SLEEP_BETWEEN = float(os.getenv("SLEEP_BETWEEN", "0.1"))  # 原脚本 0.5，local 可小些

print(f"Using local model dir: {LOCAL_MODEL_DIR}")
print("Loading tokenizer and model (this may take a while)...")

# 加载 tokenizer 和 model（尽量使用 device_map="auto"）
try:
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, trust_remote_code=True)
except Exception as e:
    print("Failed to load tokenizer from LOCAL_MODEL_DIR:", e)
    raise

# model load: try device_map="auto" with dtype float16; fallback to cpu if fails
model = None
try:
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("Model loaded with device_map='auto' and torch_dtype=float16.")
except Exception as e:
    print("device_map auto load failed:", e)
    print("Trying fallback load with low_cpu_mem_usage=True (may still OOM if insufficient memory)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_DIR,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        # move to device 0 if cuda available
        if torch.cuda.is_available():
            model.to("cuda")
        print("Fallback model load succeeded.")
    except Exception as e2:
        print("Fallback load failed:", e2)
        raise

# ========== System Prompt (复用原脚本内容) ==========
SYSTEM_PROMPT = """You are a helpful biologist and expert geneticist.
You are given a biomedical question. Analyze it carefully and provide your answer.

IMPORTANT: At the very end of your response, clearly state your final answer on a new line in this exact format:
FINAL ANSWER: <your answer here>

For multiple choice questions, FINAL ANSWER should be just the letter (e.g., FINAL ANSWER: a).
For gene identification, FINAL ANSWER should be just the gene symbol (e.g., FINAL ANSWER: BRCA1).
For variant questions, FINAL ANSWER should be the variant ID (e.g., FINAL ANSWER: rs12345).
For disease diagnosis, FINAL ANSWER should be the OMIM ID as JSON (e.g., FINAL ANSWER: {\"OMIM_ID\": \"123456\"}).
For patient gene detection, FINAL ANSWER should be JSON with gene list (e.g., FINAL ANSWER: {\"causal_gene\": [\"GENE1\"]}).
"""

# ========== 答案提取（与原脚本一致） ==========
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

# ========== 评分（复用 BiomniEval1 的逻辑） ==========
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

# ========== 本地生成实现 ==========
def run_single_local(task_name: str, prompt: str) -> str:
    """
    使用本地模型生成，与原脚本 llm.invoke(...) 的返回内容格式兼容（纯文本字符串）。
    """
    # 组装完整 prompt（与训练/微调时使用的 prompt 风格保持一致）
    full_prompt = SYSTEM_PROMPT + "\n" + prompt

    # 编码、生成
    try:
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024)
        # 将输入移到模型所在设备
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
                # 如果模型返回包含 full_prompt（有时模型会回放 prompt），去掉开头重复部分
                if text.startswith(full_prompt):
                    text = text[len(full_prompt):].strip()
                return text
            except RuntimeError as e:
                print(f"Generate runtime error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                # OOM 之类错误时尝试清理并 sleep
                torch.cuda.empty_cache()
                time.sleep(2 * (attempt + 1))
        return ""
    except Exception as e:
        print("run_single_local unexpected exception:", e)
        traceback.print_exc()
        return ""

# ========== 保存中间 checkpoint（同原脚本） ==========
def _save_checkpoint(task_name, results):
    path = f"{OUTPUT_DIR}/_ckpt_{task_name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# ========== 主函数（大体沿用原逻辑） ==========
def main():
    print(f"{'='*60}")
    print("Local Model Baseline Evaluation")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"{'='*60}")

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

    # ========== 保存结果 ==========
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

    # 打印最终表格
    print(f"\n\n{'='*60}")
    print("FINAL RESULTS — Local Model Baseline")
    print(f"{'='*60}")
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