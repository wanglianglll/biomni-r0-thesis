import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 配置文件路径和说明
file_paths = {
    "deepseek_base": "results/baseline/summary_final.json",
    "qwen1.5_sft": "results/baseline/summary_20260317_151518.json",
    "qwen1.5_base": "results/qwen_base/summary_20260317_144334.json",
    "qwen2.5_base": "scripts/results/qwen2.5_base/summary_20260318_141734.json",
    "qwen2.5_sft": "results/qwen2.5_sft/summary_20260318_212358.json",   # 待补充
    "mistral_base": "scripts/results/mistral_base/summary_20260318_202828.json",
    "mistral_sft": "scripts/results/mistral_sft/summary_xxx.json"    # 待补充
}

labels_display = {
    "deepseek_base": "DeepSeek Base",
    "qwen1.5_base": "Qwen1.5-7B Base",
    "qwen1.5_sft": "Qwen1.5-7B SFT",
    "qwen2.5_base": "Qwen2.5-7B Base",
    "qwen2.5_sft": "Qwen2.5-7B SFT",
    "mistral_base": "Mistral Base",
    "mistral_sft": "Mistral SFT"
}

# 读取所有结果
results = {}
for k, path in file_paths.items():
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        if "_overall" in summary:
            summary.pop("_overall")
        results[k] = summary
    else:
        results[k] = None  # 未提供则空

# 生成每个模型各自的测试表（分别保存为excel/csv，也可以画热力图/柱状图）
for key in ["qwen1.5_base", "qwen1.5_sft", "qwen2.5_base", "qwen2.5_sft", "mistral_base", "mistral_sft", "deepseek_base"]:
    data = results.get(key)
    if data:
        df = pd.DataFrame.from_dict(data, orient="index")
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Task"}, inplace=True)
        df.to_csv(f"{key}_result.csv", index=False)
        # 条形图用于指标展示
        plt.figure(figsize=(12,6))
        sns.barplot(x="Task", y="accuracy", data=df)
        plt.title(f"{labels_display[key]} Test Results")
        plt.xticks(rotation=60)
        plt.tight_layout()
        plt.savefig(f"{key}_result.png")
        plt.close()
    else:
        print(f"{labels_display[key]}: 文件未提供，将留空，建议后续补充。")

# 各模型SFT前后对比表
for model in ["qwen1.5", "qwen2.5", "mistral"]:
    base_key = f"{model}_base"
    sft_key = f"{model}_sft"
    base_data = results.get(base_key)
    sft_data = results.get(sft_key)
    task_set = set(base_data.keys() if base_data else [])
    task_set |= set(sft_data.keys() if sft_data else [])
    vs_df = []
    for task in task_set:
        row = {"Task": task}
        row["Base"] = base_data[task]["accuracy"] if base_data and task in base_data else None
        row["SFT"] = sft_data[task]["accuracy"] if sft_data and task in sft_data else None
        vs_df.append(row)
    vs_df = pd.DataFrame(vs_df)
    vs_df.to_csv(f"{model}_sft_vs_base.csv", index=False)
    # 可视化对比
    if vs_df["Base"].notnull().any() or vs_df["SFT"].notnull().any():
        plt.figure(figsize=(12,6))
        plt.plot(vs_df["Task"], vs_df["Base"], label="Base", marker="o")
        plt.plot(vs_df["Task"], vs_df["SFT"], label="SFT", marker="o")
        plt.legend()
        plt.xticks(rotation=60)
        plt.title(f"{model.upper()} SFT前后准确率对比")
        plt.tight_layout()
        plt.savefig(f"{model}_sft_vs_base.png")
        plt.close()
    else:
        print(f"{model}: sft对比数据缺失。")

# 模型横向对比（训练前后分别出一表）
for stage, keys in [("base", ["qwen1.5_base", "qwen2.5_base", "mistral_base", "deepseek_base"]), 
                    ("sft", ["qwen1.5_sft", "qwen2.5_sft", "mistral_sft"])]:
    tasks_union = set()
    for k in keys:
        if results.get(k):
            tasks_union |= set(results[k].keys())
    table = []
    for task in tasks_union:
        row = {"Task": task}
        for k in keys:
            row[labels_display[k]] = results[k][task]["accuracy"] if results.get(k) and task in results[k] else None
        table.append(row)
    df = pd.DataFrame(table)
    df.to_csv(f"models_{stage}_compare.csv", index=False)
    # 画图（分阶段，每个模型一条线/一组柱状）
    plt.figure(figsize=(12,6))
    for k in keys:
        if k in labels_display and df[labels_display[k]].notnull().any():
            plt.plot(df["Task"], df[labels_display[k]], marker="o", label=labels_display[k])
    plt.title(f"不同模型{stage.upper()}阶段横向对比")
    plt.legend()
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(f"models_{stage}_compare.png")
    plt.close()

print("所有单表、对比表已生成。未提供数据的表自动留空，后续补上相应JSON即可自动更新。")