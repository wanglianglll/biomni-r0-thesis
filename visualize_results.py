from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parent
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
prop = font_manager.FontProperties(fname=font_path)

RESULT_LOCATIONS = {
    'deepseek_base': [PROJECT_ROOT / 'results' / 'baseline'],
    'qwen2.5_base': [PROJECT_ROOT / 'scripts' / 'results' / 'qwen2.5_base'],
    'qwen2.5_sft': [PROJECT_ROOT / 'results' / 'qwen2.5_sft'],
    'mistral_base': [PROJECT_ROOT / 'scripts' / 'results' / 'mistral_base'],
    'mistral_sft': [PROJECT_ROOT / 'results' / 'mistral_sft'],
}

LABELS = {
    'deepseek_base': 'DeepSeek Base',
    'qwen2.5_base': 'Qwen2.5-7B Base',
    'qwen2.5_sft': 'Qwen2.5-7B SFT',
    'mistral_base': 'Mistral Base',
    'mistral_sft': 'Mistral SFT',
}

RUN_PATTERN = re.compile(r'^(\d{8}_\d{6})_([^.]+)_(base|sft)_summary\.json$')


def resolve_latest_summary(model_key: str) -> Path | None:
    candidates = []
    for directory in RESULT_LOCATIONS[model_key]:
        if not directory.exists():
            continue
        for path in directory.glob('*_summary.json'):
            match = RUN_PATTERN.match(path.name)
            if not match:
                continue
            timestamp, model_name, variant = match.groups()
            expected_model, expected_variant = model_key.rsplit('_', 1)
            if model_name != expected_model or variant != expected_variant:
                continue
            candidates.append((timestamp, path))
        legacy = directory / 'summary_final.json'
        if legacy.exists() and model_key == 'deepseek_base':
            candidates.append(('00000000_000000', legacy))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def load_summary(path: Path | None):
    if path is None or not path.exists():
        return None
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    data.pop('_overall', None)
    return data


def export_single_model_charts(results: dict):
    for key, data in results.items():
        if not data:
            print(f'{LABELS[key]}: no summary found, skipped.')
            continue
        df = pd.DataFrame.from_dict(data, orient='index').reset_index().rename(columns={'index': 'Task'})
        df.to_csv(PROJECT_ROOT / 'artifacts' / 'tables' / f'{key}_result.csv', index=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Task', y='accuracy', data=df)
        plt.title(f'{LABELS[key]} Test Results')
        plt.xticks(rotation=60)
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'artifacts' / 'figures' / f'{key}_result.png')
        plt.close()


def export_sft_vs_base(results: dict, model: str):
    base_key = f'{model}_base'
    sft_key = f'{model}_sft'
    base_data = results.get(base_key)
    sft_data = results.get(sft_key)
    task_set = set(base_data.keys() if base_data else []) | set(sft_data.keys() if sft_data else [])
    rows = []
    for task in sorted(task_set):
        rows.append({
            'Task': task,
            'Base': base_data[task]['accuracy'] if base_data and task in base_data else None,
            'SFT': sft_data[task]['accuracy'] if sft_data and task in sft_data else None,
        })
    df = pd.DataFrame(rows, columns=['Task', 'Base', 'SFT'])
    df.to_csv(PROJECT_ROOT / 'artifacts' / 'tables' / f'{model}_sft_vs_base.csv', index=False)
    if df.empty or not df[['Base', 'SFT']].notnull().any().any():
        print(f'{model}: base/sft comparison skipped because no usable data was found.')
        return
    plt.figure(figsize=(12, 6))
    plt.plot(df['Task'], df['Base'], label='Base', marker='o')
    plt.plot(df['Task'], df['SFT'], label='SFT', marker='o')
    plt.legend()
    plt.xticks(rotation=60)
    plt.title(f'{model.upper()} Base vs SFT Accuracy', fontproperties=prop)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'artifacts' / 'figures' / f'{model}_sft_vs_base.png')
    plt.close()


def export_stage_compare(results: dict, stage: str, keys: list[str]):
    tasks_union = set()
    for key in keys:
        if results.get(key):
            tasks_union |= set(results[key].keys())
    rows = []
    for task in sorted(tasks_union):
        row = {'Task': task}
        for key in keys:
            row[LABELS[key]] = results[key][task]['accuracy'] if results.get(key) and task in results[key] else None
        rows.append(row)
    columns = ['Task'] + [LABELS[key] for key in keys]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(PROJECT_ROOT / 'artifacts' / 'tables' / f'models_{stage}_compare.csv', index=False)
    if df.empty:
        print(f'{stage}: stage comparison skipped because no summaries were found.')
        return
    plt.figure(figsize=(12, 6))
    plotted = False
    for key in keys:
        col = LABELS[key]
        if col in df and df[col].notnull().any():
            plt.plot(df['Task'], df[col], marker='o', label=col)
            plotted = True
    if not plotted:
        plt.close()
        print(f'{stage}: stage comparison skipped because all values were empty.')
        return
    plt.title(f'Model Comparison ({stage.upper()})', fontproperties=prop)
    plt.legend()
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'artifacts' / 'figures' / f'models_{stage}_compare.png')
    plt.close()


def main():
    (PROJECT_ROOT / 'artifacts' / 'figures').mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / 'artifacts' / 'tables').mkdir(parents=True, exist_ok=True)

    resolved = {key: resolve_latest_summary(key) for key in RESULT_LOCATIONS}
    for key, path in resolved.items():
        print(f'{key}: {path}' if path else f'{key}: no summary found')

    results = {key: load_summary(path) for key, path in resolved.items()}

    export_single_model_charts(results)
    export_sft_vs_base(results, 'qwen2.5')
    export_sft_vs_base(results, 'mistral')
    export_stage_compare(results, 'base', ['qwen2.5_base', 'mistral_base', 'deepseek_base'])
    export_stage_compare(results, 'sft', ['qwen2.5_sft', 'mistral_sft'])
    print('All charts and tables updated from latest timestamped summaries.')


if __name__ == '__main__':
    main()
