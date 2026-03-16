"""
DeepSeek-V3 基线评测 — 批量跑 BiomniEval1 全部 433 个实例
预估: 费用 ¥5-15, 耗时 30-60 分钟
"""
import json
import os
import re
import time
import traceback
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv("../.env", override=True)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ========== 配置 ==========
DEEPSEEK_BASE_URL = os.getenv("CUSTOM_MODEL_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_API_KEY = os.getenv("CUSTOM_MODEL_API_KEY")
OUTPUT_DIR = "../results/baseline"
LOCAL_PARQUET = "../data/biomni_eval1_dataset.parquet"
MAX_RETRIES = 3
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 创建 LLM ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY,
)

# ========== System Prompt ==========
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


# ========== 答案提取 ==========
def extract_answer(raw: str, task_name: str) -> str:
    # 先提取 FINAL ANSWER: xxx
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
        # 先尝试解析 JSON
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


# ========== 单实例推理 ==========
def run_single(task_name: str, prompt: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    for attempt in range(MAX_RETRIES):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"      Retry {attempt+1}/{MAX_RETRIES}: {e}")
            time.sleep(3 * (attempt + 1))
    return ""


# ========== 主函数 ==========
def main():
    print(f"{'='*60}")
    print(f"DeepSeek-V3 Baseline Evaluation")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # 加载数据
    df = pd.read_parquet(LOCAL_PARQUET)
    tasks = sorted(df["task_name"].unique())
    print(f"Tasks: {tasks}")
    print(f"Total instances: {len(df)}")

    all_results = {}
    summary = {}
    total_correct = 0
    total_count = 0

    for task_name in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        task_df = df[(df["task_name"] == task_name) & (df["split"] == "val")]
        n = len(task_df)
        print(f"Instances: {n}")

        if n == 0:
            continue

        task_results = []
        task_correct = 0

        for i, (_, row) in enumerate(task_df.iterrows()):
            print(f"  [{i+1}/{n}] ID={row['task_instance_id']}...", end=" ", flush=True)

            # 调用 LLM
            raw_answer = run_single(task_name, row["prompt"])

            # 提取答案
            extracted = extract_answer(raw_answer, task_name) if raw_answer else ""

            # 评分
            score = compute_score(task_name, extracted, row["answer"])
            task_correct += int(score >= 1.0)

            emoji = "✅" if score >= 1.0 else "❌"
            print(f"{emoji} answer={extracted[:30]:<30} truth={row['answer'][:30]}")

            task_results.append({
                "task_instance_id": int(row["task_instance_id"]),
                "score": score,
                "extracted": extracted,
                "ground_truth": row["answer"],
                "raw_preview": raw_answer[:300] if raw_answer else "",
            })

            # 每做完一个实例，保存一次 checkpoint
            _save_checkpoint(task_name, task_results)

            # API 限流
            time.sleep(0.5)

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
        "model": "deepseek-chat (DeepSeek-V3)",
        "timestamp": timestamp,
    }

    with open(f"{OUTPUT_DIR}/detail_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(f"{OUTPUT_DIR}/summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印最终表格
    print(f"\n\n{'='*60}")
    print(f"FINAL RESULTS — DeepSeek-V3 Baseline")
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


def _save_checkpoint(task_name, results):
    path = f"{OUTPUT_DIR}/_ckpt_{task_name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()