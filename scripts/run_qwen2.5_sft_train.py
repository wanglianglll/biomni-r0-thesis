"""
Qwen2.5-7B-Instruct 全参数SFT微调训练脚本
所有路径均写死，无需命令行参数。
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import datasets
os.environ["TRANSFORMERS_NO_BF16"] = "true"
# 路径配置
MODEL_DIR = "/root/autodl-tmp/Biomni-main/models/Qwen2.5-7B-Instruct/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
TRAIN_FILE = "/root/autodl-tmp/Biomni-main/data/sft/sft_train.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/Biomni-main/scripts/output/qwen2.5_sft"

EPOCHS = 20
BATCH_SIZE = 1
LR = 2e-5
MAX_SEQ_LENGTH = 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)
model.gradient_checkpointing_enable()  # 建议开启节省显存

# 加载/预处理数据集（jsonl格式，每行含"prompt"和"completion"字段）
ds = datasets.Dataset.from_json(TRAIN_FILE).train_test_split(test_size=0.02)

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
    result = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH)
    result["labels"] = result["input_ids"]  # 必须加标签
    return result

ds_tokenized = ds["train"].map(preprocess, remove_columns=ds["train"].column_names, batched=False)
eval_ds = ds["test"].map(preprocess, remove_columns=ds["test"].column_names, batched=False)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    logging_steps=50,
    save_strategy="no",  # 可加此行消除自动checkpoint保存
    # bf16=False,   # 禁用BFloat16
    # fp16=True,
    remove_unused_columns=False,
    gradient_accumulation_steps=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_tokenized,
    eval_dataset=eval_ds,
    # tokenizer=tokenizer,
    data_collator=data_collator,
)

print("开始全量SFT训练 ...")
trainer.train()
print("训练完成，正在保存权重和tokenizer ...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("全部任务结束。")

print("训练完成，准备关机")
os.system("shutdown -h now")