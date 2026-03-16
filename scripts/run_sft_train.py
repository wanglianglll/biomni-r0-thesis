"""
SFT 微调 Biomni-A1（以 Qwen2.5-7B-Chat 为例）
输入: data/sft/sft_train.jsonl (ChatML格式)
输出: output/sft_qwen2_7b
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

BASE_MODEL = "Qwen/Qwen1.5-7B-Chat"
OUTPUT_DIR = "output/sft_qwen2_7b"
MAX_LEN = 2048

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
    result = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    result["labels"] = result["input_ids"].clone()
    return result

ds = ds.map(preprocess, remove_columns=ds.column_names, batched=False)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto", trust_remote_code=True)

args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    bf16=True, # 或 fp16
    output_dir=OUTPUT_DIR,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)