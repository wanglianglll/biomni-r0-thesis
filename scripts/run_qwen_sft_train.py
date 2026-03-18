"""
SFT 微调 Biomni-A1（以 Qwen2.5-7B-Chat 为例）
输入: data/sft/sft_train.jsonl (ChatML格式)
输出: output/sft_qwen2_7b
"""
import torch
torch.cuda.empty_cache()
import glob, shutil

import os,json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

BASE_MODEL = "Qwen/Qwen1.5-7B-Chat"
OUTPUT_DIR = "output/sft_qwen2_7b"
MAX_LEN = 1024

# 加载 ChatML 格式数据
ds = load_dataset("json", data_files="/root/autodl-tmp/Biomni-main/data/sft/sft_train.jsonl")["train"]
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

def preprocess(example):
    messages = example["messages"]
    text = ""
    for msg in messages:
        role = msg["role"]
        if role == "system":
            text += f"<|system|>\n{msg['content']}\n"
        elif role == "user":
            text += f"<|user|>\n{msg['content']}\n"
        elif role == "assistant":
            text += f"<|assistant|>\n{msg['content']}\n"
    result = tokenizer(text, truncation=True, padding="max_length", max_length=1024)
    result["labels"] = result["input_ids"]  # 修正此处
    return result

ds = ds.map(preprocess, remove_columns=ds.column_names, batched=False)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto", trust_remote_code=True)

args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    output_dir=OUTPUT_DIR,
    logging_steps=10,
    save_strategy="no",       # 不自动保存 checkpoint（节省磁盘写入）
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    # tokenizer=tokenizer,
)
# import math
# world_size = 1  # 单卡训练：1；多卡时设置为实际卡数
# B = args.per_device_train_batch_size
# G = args.gradient_accumulation_steps
# N = len(ds)
# batches_per_epoch = math.ceil(N / (B * world_size))
# steps_per_epoch = math.ceil(batches_per_epoch / G)
# total_expected_steps = steps_per_epoch * args.num_train_epochs

# print(f"dataset size N={N}")
# print(f"B={B}, G={G}, world_size={world_size}")
# print(f"batches_per_epoch={batches_per_epoch}, steps_per_epoch={steps_per_epoch}")
# print(f"expected total optimization steps (for {args.num_train_epochs} epochs) = {total_expected_steps}")

trainer.train()
# 假设 trainer 是你创建的 Trainer 实例，OUTPUT_DIR 是你的输出目录
log_path = os.path.join(OUTPUT_DIR, "train_log_history.json")
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)

print("Saved log history to", log_path)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("训练完成，准备关机")
os.system("shutdown -h now")
