"""
重跑 lab_bench_seqqa 和 patient_gene_detection
修复: 加大超时、加网络重试、跳过代理
"""
import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv("../.env", override=True)

# 关键：绕过 AutoDL 的 squid 代理
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

OUTPUT_DIR = "../results/baseline"
LOCAL_PARQUET = "../data/biomni_eval1_dataset.parquet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    base_url=os.getenv("CUSTOM_MODEL_BASE_URL"),
    api_key=os.getenv("CUSTOM_MODEL_API_KEY"),
    timeout=120,        # 超时 120 秒
    max_retries=5,      # 内置重试 5 次
)

# 验证
print("Testing connection...", end=" ", flush=True)
try:
    r = llm.invoke("Say OK")
    print(f"✅ Connected: {r.content}")
except Exception as e:
    print(f"❌ Failed: {e}")
    print("\nTry: source /etc/network_turbo")
    exit(1)

df = pd.read_parquet(LOCAL_PARQUET)

PROMPTS = {
    "lab_bench_seqqa": """You are an expert biologist answering a sequence-based question.
Read the question and all options carefully, then select the best answer.

You MUST end your response with exactly this format on the last line:
FINAL ANSWER: X
where X is a single letter (A, B, C, D, or E). Nothing else on that line.""",

    "patient_gene_detection": """You are an expert clinical geneticist.
Analyze the patient case and identify the most likely causal gene.
The answer should be an Ensembl Gene ID (ENSG format).

You MUST end your response with exactly this format on the last line:
FINAL ANSWER: ENSGXXXXXXXXXXX
where ENSGXXXXXXXXXXX is the 15-character Ensembl Gene ID.""",
}


def extract_answer(raw, task_name):
    if not raw:
        return ""
    match = re.search(r'FINAL ANSWER:\s*(.+)', raw, re.IGNORECASE)
    extracted = match.group(1).strip().strip(".") if match else raw.strip()

    if task_name == "lab_bench_seqqa":
        m = re.search(r'\b([A-Ea-e])\b', extracted)
        if m:
            return m.group(1).upper()
        for p in [r'(?:answer|Answer)\s*(?:is|:)\s*\(?([A-Ea-e])\)?',
                  r'\*\*([A-Ea-e])\*\*']:
            m = re.search(p, raw)
            if m:
                return m.group(1).upper()
        letters = re.findall(r'\b([A-Ea-e])\b', raw)
        return letters[-1].upper() if letters else ""

    elif task_name == "patient_gene_detection":
        ensg = re.findall(r'(ENSG\d{11})', raw)
        if ensg:
            return json.dumps({"causal_gene": ensg[:5]})
        skip = {"FINAL", "ANSWER", "THE", "AND", "FOR", "GENE", "PATIENT", "CAUSAL"}
        genes = re.findall(r'\b([A-Z][A-Z0-9]{2,15})\b', extracted)
        genes = [g for g in genes if g not in skip]
        return json.dumps({"causal_gene": genes[:5]}) if genes else ""
    return extracted


def compute_score(task_name, user_answer, ground_truth):
    try:
        if task_name == "lab_bench_seqqa":
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0
        elif task_name == "patient_gene_detection":
            user_dict = json.loads(user_answer) if isinstance(user_answer, str) else user_answer
            predicted = user_dict.get("causal_gene", [])
            if not isinstance(predicted, list):
                predicted = [predicted]
            true_genes = [g.strip() for g in ground_truth.split(",")] if "," in ground_truth else [ground_truth]
            return 1.0 if predicted and set(true_genes) & set(predicted) else 0.0
    except Exception:
        return 0.0
    return 0.0


for task_name in ["lab_bench_seqqa", "patient_gene_detection"]:
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    task_df = df[(df["task_name"] == task_name) & (df["split"] == "val")]
    n = len(task_df)
    results = []
    correct = 0

    for i, (_, row) in enumerate(task_df.iterrows()):
        print(f"  [{i+1}/{n}] ID={row['task_instance_id']}...", end=" ", flush=True)

        messages = [
            SystemMessage(content=PROMPTS[task_name]),
            HumanMessage(content=row["prompt"]),
        ]

        raw = ""
        for attempt in range(5):
            try:
                resp = llm.invoke(messages)
                raw = resp.content
                break
            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"err(wait {wait}s)...", end=" ", flush=True)
                time.sleep(wait)

        extracted = extract_answer(raw, task_name)
        score = compute_score(task_name, extracted, row["answer"])
        correct += int(score >= 1.0)

        emoji = "✅" if score >= 1.0 else "❌"
        print(f"{emoji} ans={extracted[:35]:<35} truth={row['answer'][:20]}")

        results.append({
            "task_instance_id": int(row["task_instance_id"]),
            "score": score,
            "extracted": extracted,
            "ground_truth": row["answer"],
            "raw_preview": raw[:800],
        })

        # 保存 checkpoint
        with open(f"{OUTPUT_DIR}/_ckpt_{task_name}_rerun.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(1)

    acc = correct / n
    print(f"\n  >>> {task_name}: {acc:.3f} ({correct}/{n})")

# ========== 合并最终结果 ==========
print(f"\n\n{'='*60}")
print("FINAL BASELINE (论文用)")
print(f"{'='*60}")

final = {
    "crispr_delivery":                 {"accuracy": 0.400, "correct": 4,  "total": 10},
    "gwas_causal_gene_gwas_catalog":   {"accuracy": 0.580, "correct": 29, "total": 50},
    "gwas_causal_gene_opentargets":    {"accuracy": 0.780, "correct": 39, "total": 50},
    "gwas_causal_gene_pharmaprojects": {"accuracy": 0.720, "correct": 36, "total": 50},
    "gwas_variant_prioritization":     {"accuracy": 0.209, "correct": 9,  "total": 43},
    "lab_bench_dbqa":                  {"accuracy": 0.240, "correct": 12, "total": 50},
    "screen_gene_retrieval":           {"accuracy": 0.460, "correct": 23, "total": 50},
    "rare_disease_diagnosis":          {"accuracy": 0.033, "correct": 1,  "total": 30},
}

for task_name in ["lab_bench_seqqa", "patient_gene_detection"]:
    ckpt = f"{OUTPUT_DIR}/_ckpt_{task_name}_rerun.json"
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            data = json.load(f)
        c = sum(1 for d in data if d["score"] >= 1.0)
        final[task_name] = {"accuracy": round(c/len(data), 4), "correct": c, "total": len(data)}

tc = sum(v["correct"] for v in final.values())
tt = sum(v["total"] for v in final.values())

print(f"{'Task':<42} {'Acc':>7} {'Correct':>8} {'Total':>6}")
print("-" * 65)
for t in sorted(final.keys()):
    s = final[t]
    print(f"{t:<42} {s['accuracy']:>6.3f} {s['correct']:>7d} {s['total']:>5d}")
print("-" * 65)
print(f"{'OVERALL':<42} {tc/tt:>6.3f} {tc:>7d} {tt:>5d}")

final["_overall"] = {"accuracy": round(tc/tt, 4), "correct": tc, "total": tt,
                      "model": "deepseek-chat (DeepSeek-V3)"}
with open(f"{OUTPUT_DIR}/summary_final.json", "w") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {OUTPUT_DIR}/summary_final.json")