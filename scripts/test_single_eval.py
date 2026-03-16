"""
Step 0: 验证 DeepSeek-V3 + BiomniEval1 全流程
不�� react Agent（避免工具依赖），直接调 LLM
"""
import json
import os
import re

from dotenv import load_dotenv

load_dotenv("../.env", override=True)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ========== 1. 配置 ==========
DEEPSEEK_BASE_URL = os.getenv("CUSTOM_MODEL_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_API_KEY = os.getenv("CUSTOM_MODEL_API_KEY")
print(f"Base URL: {DEEPSEEK_BASE_URL}")
print(f"API Key: {DEEPSEEK_API_KEY[:8]}..." if DEEPSEEK_API_KEY else "API Key: NOT SET!")

# ========== 2. 创建 LLM（直接用 langchain_openai，不经过 biomni 框架） ==========
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY,
)

# 快速验证 LLM 是否能通
test_resp = llm.invoke("Say hello in one word.")
print(f"LLM test: {test_resp.content}")

# ========== 3. 加载评测集 ==========
# 直接 import 评测类（它只依赖 pandas，不依赖工具模块）
import pandas as pd

# 如果本地有 parquet 就读本地，否则从 HuggingFace 拉
LOCAL_PARQUET = "../data/biomni_eval1_dataset.parquet"
if os.path.exists(LOCAL_PARQUET):
    df = pd.read_parquet(LOCAL_PARQUET)
    print(f"Loaded from local: {LOCAL_PARQUET}")
else:
    df = pd.read_parquet("hf://datasets/biomni/Eval1/biomni_eval1_dataset.parquet")
    os.makedirs("../data", exist_ok=True)
    df.to_parquet(LOCAL_PARQUET)
    print(f"Downloaded and saved to: {LOCAL_PARQUET}")

print(f"Dataset: {len(df)} instances, {df['task_name'].nunique()} tasks")

# 显示每个任务的数据量
for t in sorted(df["task_name"].unique()):
    val_count = len(df[(df["task_name"] == t) & (df["split"] == "val")])
    train_count = len(df[(df["task_name"] == t) & (df["split"] == "train")])
    print(f"  {t}: train={train_count}, val={val_count}")

# ========== 4. 选一个任务测试 ==========
task_name = "crispr_delivery"
task_df = df[(df["task_name"] == task_name) & (df["split"] == "val")]
instance = task_df.iloc[0]

print(f"\n{'='*60}")
print(f"Task: {task_name}")
print(f"Instance ID: {instance['task_instance_id']}")
print(f"Ground truth: {instance['answer']}")
print(f"Prompt preview:\n{instance['prompt'][:500]}...")
print(f"{'='*60}")

# ========== 5. 构造 System Prompt + 调用 LLM ==========
SYSTEM_PROMPT = """You are a helpful biologist and expert geneticist.
You are given a biomedical question. Analyze it carefully and provide your answer.

IMPORTANT: At the very end of your response, clearly state your final answer on a new line in this exact format:
FINAL ANSWER: <your answer here>

For multiple choice questions, FINAL ANSWER should be just the letter (e.g., FINAL ANSWER: a).
For gene identification, FINAL ANSWER should be just the gene symbol (e.g., FINAL ANSWER: BRCA1).
For variant questions, FINAL ANSWER should be the variant ID (e.g., FINAL ANSWER: rs12345).
"""

from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=instance["prompt"]),
]

print("\n>>> Calling DeepSeek-V3...")
response = llm.invoke(messages)
answer_raw = response.content
print(f">>> Raw response:\n{answer_raw}")

# ========== 6. 提取最终答案 ==========
def extract_answer(raw: str, task_name: str) -> str:
    """从 LLM 回答中提取结构化答案"""
    # 先尝试提取 FINAL ANSWER: xxx 格式
    match = re.search(r'FINAL ANSWER:\s*(.+)', raw, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip().strip(".")
    else:
        extracted = raw.strip()

    # 根据任务类型做进一步清理
    if task_name == "crispr_delivery":
        # 期望 a-f 的单个字母
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
        if m:
            return json.dumps({"OMIM_ID": m.group(1)})
        return extracted

    elif task_name == "patient_gene_detection":
        genes = re.findall(r'\b([A-Z][A-Z0-9]{1,10})\b', extracted)
        if genes:
            return json.dumps({"causal_gene": genes[:5]})
        return extracted

    return extracted

extracted = extract_answer(answer_raw, task_name)
print(f"\n>>> Extracted answer: {extracted}")

# ========== 7. 评分 ==========
# 直接用 BiomniEval1 的评分逻辑（不需要实例化整个类，手动调评分）
ground_truth = instance["answer"]
if task_name == "crispr_delivery":
    score = 1.0 if extracted.strip().lower() == ground_truth.strip().lower() else 0.0
else:
    score = 1.0 if extracted.strip() == ground_truth.strip() else 0.0

print(f">>> Score: {score}")
print(f">>> Ground truth: {ground_truth}")
print(f">>> Our answer:   {extracted}")

if score >= 1.0:
    print("\n✅ CORRECT! Pipeline works!")
else:
    print("\n❌ WRONG answer, but pipeline is functional!")
    print("   (This is expected — not every question will be correct)")