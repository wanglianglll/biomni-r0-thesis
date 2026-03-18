import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# 文件路径定义
logs = {
    "qwen1.5_sft": "scripts/output/sft_qwen2_7b/train_log_history.json",
    "qwen2.5_sft": "scripts/output/qwen2.5_sft/train_log_history.json",
    "mistral_sft": "scripts/output/mistral_sft/train_log_history.json"
}

# 加载所有日志
log_data = {}
for key, filepath in logs.items():
    with open(filepath, 'r', encoding='utf-8') as f:
        log_data[key] = json.load(f)

# 提取与整理成DataFrame
dfs = {}
for key, log in log_data.items():
    # 仅保留训练过程条目（过滤掉最后整体统计信息的字典）
    data = [item for item in log if 'loss' in item]
    df = pd.DataFrame(data)
    dfs[key] = df

# 可视化loss曲线
plt.figure(figsize=(10,6))
for key, df in dfs.items():
    plt.plot(df['epoch'], df['loss'], label=key)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SFT训练loss曲线对比')
plt.grid(True)
plt.tight_layout()
plt.savefig('sft_train_loss_curve.png')
plt.show()

# 可视化grad_norm曲线
plt.figure(figsize=(10,6))
for key, df in dfs.items():
    plt.plot(df['epoch'], df['grad_norm'], label=key)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Grad Norm')
plt.title('SFT训练梯度范数曲线对比')
plt.grid(True)
plt.tight_layout()
plt.savefig('sft_train_gradnorm_curve.png')
plt.show()

# 可视化learning_rate曲线
plt.figure(figsize=(10,6))
for key, df in dfs.items():
    plt.plot(df['epoch'], df['learning_rate'], label=key)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('SFT训练学习率曲线对比')
plt.grid(True)
plt.tight_layout()
plt.savefig('sft_train_lr_curve.png')
plt.show()

# 单模型loss曲线分别输出
for key, df in dfs.items():
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"{key} Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'sft_loss_{key}.png')
    plt.close()

print("曲线已保存为 png 文件。支持 epoch/loss/grad_norm/learning_rate 多模型对比与单模型曲线展示。")