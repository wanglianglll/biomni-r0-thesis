
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
prop = font_manager.FontProperties(fname=font_path)

logs = {
    'qwen2.5_sft': 'scripts/output/qwen2.5_sft/train_log_history.json',
    'mistral_sft': 'scripts/output/mistral_sft/train_log_history.json',
}

log_data = {}
for key, filepath in logs.items():
    with open(filepath, 'r', encoding='utf-8') as f:
        log_data[key] = json.load(f)

dfs = {key: pd.DataFrame([item for item in log if 'loss' in item]) for key, log in log_data.items()}

for metric, title, filename in [
    ('loss', 'SFT??loss????', 'sft_train_loss_curve.png'),
    ('grad_norm', 'SFT??????????', 'sft_train_gradnorm_curve.png'),
    ('learning_rate', 'SFT?????????', 'sft_train_lr_curve.png'),
]:
    plt.figure(figsize=(10, 6))
    for key, df in dfs.items():
        if metric in df:
            plt.plot(df['epoch'], df[metric], label=key)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(title, fontproperties=prop)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

for key, df in dfs.items():
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{key} Loss Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'sft_loss_{key}.png')
    plt.close()

print('?????? png ???')
