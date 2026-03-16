"""
SFT 训练数据收集
Step 1: 将 BiomniEval1 val 拆为 train/test (70/30)
Step 2: 用 DeepSeek-V3 为 train 部分生成 Chain-of-Thought 推理过程
输出: data/sft/sft_train.jsonl — 可直接用于 SFT 训练
"""
import json
import os
import re
import time
import random
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv("../.env", override=True)

# 绕过代理
for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(key, None)

OUTPUT_DIR = "../data/sft"
LOCAL_PARQUET = "../data/biomni_eval1_dataset.parquet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 固定随机种子，保证可复现
SEED = 42
random.seed(SEED)

# ========== Step 1: 拆分数据集 ==========
print("=" * 60)
print("Step 1: Split val into train/test (70/30)")
print("=" * 60)

df = pd.read_parquet(LOCAL_PARQUET)
val_df = df[df["split"] == "val"].copy()
print(f"Total val instances: {len(val_df)}")

# 按任务分层抽样，保证每个任务的 train/test 比例一致
train_dfs = []
test_dfs = []

for task_name in sorted(val_df["task_name"].unique()):
    task_df = val_df[val_df["task_name"] == task_name].copy()
    task_df = task_df.sample(frac=1, random_state=SEED)  # shuffle

    n = len(task_df)
    n_train = max(1, int(n * 0.7))  # 至少 1 条训练

    train_dfs.append(task_df.iloc[:n_train])
    test_dfs.append(task_df.iloc[n_train:])

    print(f"  {task_name}: total={n}, train={n_train}, test={n - n_train}")

train_df = pd.concat(train_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

print(f"\nTotal: train={len(train_df)}, test={len(test_df)}")

# 保存 test split（后续评测用）
test_df.to_parquet(f"{OUTPUT_DIR}/eval_test_split.parquet")
train_df.to_parquet(f"{OUTPUT_DIR}/eval_train_split.parquet")
print(f"Saved splits to {OUTPUT_DIR}/")

# ========== Step 2: 用 DeepSeek-V3 生成 CoT 推理 ==========
print(f"\n{'=' * 60}")
print("Step 2: Generate CoT reasoning with DeepSeek-V3")
print("=" * 60)

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    base_url=os.getenv("CUSTOM_MODEL_BASE_URL"),
    api_key=os.getenv("CUSTOM_MODEL_API_KEY"),
    timeout=120,
    max_retries=5,
)

# 验证连接
print("Testing connection...", end=" ", flush=True)
try:
    r = llm.invoke("Say OK")
    print(f"✅ {r.content}")
except Exception as e:
    print(f"❌ {e}")
    print("Run: source /etc/network_turbo")
    exit(1)

# 针对每个任务的 CoT 提示模板
COT_SYSTEM_PROMPTS = {
    "crispr_delivery": """You are an expert in CRISPR gene editing and delivery methods.
Given the case description, think step by step:
1. Identify the cell type/organism
2. Consider the delivery constraints (in vivo vs in vitro, cell type properties)
3. Evaluate each delivery method's suitability
4. Select the best option with clear reasoning

Format your response as:
<think>
[Your step-by-step reasoning here]
</think>
FINAL ANSWER: [letter a-f]""",

    "gwas_causal_gene_gwas_catalog": """You are an expert geneticist specializing in GWAS analysis.
Given the GWAS information, think step by step:
1. Analyze the phenotype/trait described
2. Consider known gene-trait associations
3. Evaluate candidate genes based on biological function
4. Identify the most likely causal gene

Format your response as:
<think>
[Your step-by-step reasoning here]
</think>
FINAL ANSWER: [gene symbol]""",

    "gwas_causal_gene_opentargets": """You are an expert geneticist specializing in GWAS analysis and drug target identification.
Given the GWAS and target information, think step by step:
1. Analyze the disease/trait and associated locus
2. Consider gene function and expression patterns
3. Evaluate evidence from OpenTargets
4. Identify the most likely causal gene

Format your response as:
<think>
[Your step-by-step reasoning here]
</think>
FINAL ANSWER: [gene symbol]""",

    "gwas_causal_gene_pharmaprojects": """You are an expert in pharmacogenomics and drug development.
Given the pharmaceutical project information, think step by step:
1. Analyze the disease indication
2. Consider known drug targets for this indication
3. Evaluate the genetic evidence
4. Identify the most likely causal gene

Format your response as:
<think>
[Your step-by-step reasoning here]
</think>
FINAL ANSWER: [gene symbol]""",

    "gwas_variant_prioritization": """You are an expert in statistical genetics and variant analysis.
Given the GWAS variant information, think step by step:
1. Examine the genomic region and nearby genes
2. Consider the functional annotation of variants
3. Evaluate statistical evidence (p-values, effect sizes)
4. Prioritize the most likely causal variant

Format your response as:
<think>
[Your step-by-step reasoning here]
</think>
FINAL ANSWER: [rs ID]""",

    "lab_bench_dbqa": """You are an expert biologist answering a database-based question.
Think step by step:
1. Understand what the question is asking
2. Recall relevant biological facts
3. Evaluate each option carefully
4. Select the best answer

Format your response as:
<think>
[Your step-by-step reasoning here]
</think>
FINAL ANSWER: [letter A-E]""",

    "lab_bench_seqqa": """You are an expert biologist answering a sequence-based question.
Think step by step:
1. Analyze the sequence information provided
2. Consider relevant biological properties (structure, function, conservation)
3. Evaluate each option
4. Select the best answer

Format your response as:
<think>
[Your step-by-step reasoning here]
</think>
FINAL ANSWER: [letter A-E]""",

    "patient_gene_detection": """You are an expert clinical geneticist.
Given the patient case, think step by step:
1. Identify key symptoms and clinical features
2. Consider differential diagnoses
3. Map symptoms to known genetic disorders
4. Identify the most likely causal gene (Ensembl ID)

Format your response as:
<think>
[Your step-by-step reasoning here]
</think>
FINAL ANSWER: [ENSG ID]""",

    "rare_disease_diagnosis": """You are an expert in rare disease diagnosis.
Given the patient description, think step by step:
1. List the key clinical features
2. Consider known rare diseases matching these features
3. Narrow down using distinguishing symptoms
4. Identify the OMIM disease ID

Format your response as:
<think>
[Your step-by-step reasoning here]
</think>
FINAL ANSWER: {{"OMIM_ID": "[6-digit ID]"}}""",

    "screen_gene_retrieval": """You are an expert in functional genomics and genetic screens.
Given the screen description, think step by step:
1. Understand the experimental setup (screen type, cell line, readout)
2. Consider known gene functions relevant to the screen
3. Evaluate candidate genes
4. Identify the most relevant gene

Format your response as:
<think>
[Your step-by-step reasoning here]
</think>
FINAL ANSWER: [gene symbol]""",
}


def generate_cot(task_name: str, prompt: str, answer: str) -> dict | None:
    """用 DeepSeek-V3 生成 CoT，并验证答案正确性"""
    system_prompt = COT_SYSTEM_PROMPTS.get(task_name, COT_SYSTEM_PROMPTS["lab_bench_dbqa"])

    # 在 system prompt 中注入正确答案提示（teacher forcing）
    hint = f"\n\nNote: The correct answer is {answer}. Make sure your reasoning leads to this answer."
    messages = [
        SystemMessage(content=system_prompt + hint),
        HumanMessage(content=prompt),
    ]

    for attempt in range(3):
        try:
            resp = llm.invoke(messages)
            raw = resp.content

            # 验证生成的回答包含正确答案
            if answer.lower() in raw.lower() or answer.upper() in raw.upper():
                return {
                    "task_name": task_name,
                    "prompt": prompt,
                    "response": raw,
                    "answer": answer,
                    "has_think_tag": "<think>" in raw,
                }
            else:
                # 答案不匹配，再试一次
                continue

        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"err(wait {wait}s)...", end=" ", flush=True)
            time.sleep(wait)

    return None


# ========== 主循环：为每个训练实例生成 CoT ==========
sft_data = []
failed = 0
checkpoint_path = f"{OUTPUT_DIR}/_sft_checkpoint.jsonl"

# 检查是否有之前的 checkpoint
existing_ids = set()
if os.path.exists(checkpoint_path):
    with open(checkpoint_path) as f:
        for line in f:
            d = json.loads(line)
            existing_ids.add(f"{d['task_name']}_{d.get('task_instance_id', '')}")
            sft_data.append(d)
    print(f"Loaded {len(sft_data)} from checkpoint")

total = len(train_df)
print(f"\nGenerating CoT for {total} instances...")
print(f"Estimated cost: ¥{total * 0.01:.1f} - ¥{total * 0.03:.1f}")
print(f"Estimated time: {total * 3 / 60:.0f} - {total * 5 / 60:.0f} minutes\n")

for i, (_, row) in enumerate(train_df.iterrows()):
    task_name = row["task_name"]
    instance_id = f"{task_name}_{row['task_instance_id']}"

    # 跳过已处理的
    if instance_id in existing_ids:
        continue

    print(f"  [{i+1}/{total}] {task_name} ID={row['task_instance_id']}...", end=" ", flush=True)

    result = generate_cot(task_name, row["prompt"], row["answer"])

    if result:
        result["task_instance_id"] = int(row["task_instance_id"])
        sft_data.append(result)

        # 实时保存 checkpoint
        with open(checkpoint_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        emoji = "✅" if result["has_think_tag"] else "⚠️"
        print(f"{emoji} generated ({len(result['response'])} chars)")
    else:
        failed += 1
        print("❌ failed")

    time.sleep(1)

# ========== Step 3: 转换为 SFT 训练格式 ==========
print(f"\n{'=' * 60}")
print("Step 3: Convert to SFT training format")
print("=" * 60)

print(f"Total: {len(sft_data)} success, {failed} failed")

# 格式 1: ChatML / messages 格式（适配大多数训练框架）
sft_messages = []
for d in sft_data:
    sft_messages.append({
        "messages": [
            {"role": "system", "content": "You are a helpful biologist and expert geneticist. Think step by step before giving your final answer."},
            {"role": "user", "content": d["prompt"]},
            {"role": "assistant", "content": d["response"]},
        ],
        "task_name": d["task_name"],
    })

# 保存
with open(f"{OUTPUT_DIR}/sft_train.jsonl", "w") as f:
    for item in sft_messages:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\nSaved: {OUTPUT_DIR}/sft_train.jsonl ({len(sft_messages)} samples)")

# 统计
print(f"\n{'=' * 60}")
print("SFT Data Summary")
print("=" * 60)
task_counts = {}
for d in sft_data:
    task_counts[d["task_name"]] = task_counts.get(d["task_name"], 0) + 1

print(f"{'Task':<42} {'Train':>6} {'CoT':>6}")
print("-" * 56)
for task in sorted(task_counts.keys()):
    t_count = len(train_df[train_df["task_name"] == task])
    print(f"{task:<42} {t_count:>5d} {task_counts[task]:>5d}")
print("-" * 56)
print(f"{'TOTAL':<42} {total:>5d} {len(sft_data):>5d}")

print(f"\n✅ SFT 数据收集完成！")
print(f"   训练数据: {OUTPUT_DIR}/sft_train.jsonl")
print(f"   测试数据: {OUTPUT_DIR}/eval_test_split.parquet")
print(f"   训练拆分: {OUTPUT_DIR}/eval_train_split.parquet")