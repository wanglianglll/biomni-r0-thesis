import pandas as pd

# 文件路径（你实际仓库中的路径）
DATA_PATH = 'data/biomni_eval1_dataset.parquet'

# 加载评测数据
df = pd.read_parquet(DATA_PATH)

# 只选 validation 子集（或者全体，按需调整）
val_df = df[df['split'] == 'val'] if 'split' in df else df

# 按任务类别分组，挑几个经典任务各挑一个
examples = []
for task in ['crispr_delivery', 'lab_bench_dbqa', 'gwas_causal_gene', 'rare_disease_diagnosis']:
    task_rows = val_df[val_df['task_name'] == task]
    if len(task_rows) > 0:
        row = task_rows.iloc[0]
        examples.append({
            'task': task,
            'prompt': row['prompt'],
            'answer': row['answer']
        })

# 若还不够五个，补齐前几条
if len(examples) < 5:
    for idx, row in val_df.iterrows():
        if len(examples) >= 5:
            break
        examples.append({
            'task': row['task_name'],
            'prompt': row['prompt'],
            'answer': row['answer']
        })

# 打印可粘贴到 Gradio 的格式
print("------经典评测样例（可直接粘贴到 Gradio 对话框）------")
for i, eg in enumerate(examples, 1):
    print(f"【任务】{eg['task']}\n【输入】{eg['prompt']}\n【参考答案】{eg['answer']}\n---\n")