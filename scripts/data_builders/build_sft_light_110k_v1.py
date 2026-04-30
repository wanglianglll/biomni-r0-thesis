
from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/root/autodl-tmp/Biomni-main')
OUT = ROOT / 'data/sft_light_110k_v1'
SEED = 20260427
SPECS = [
    ('D1', ROOT / 'data/sft_d1_datalake_v2/d1_datalake_train.jsonl', 35000),
    ('D2', ROOT / 'data/sft_d2_contrast_v1/d2_contrast_train.jsonl', 40000),
    ('D3', ROOT / 'data/sft_d3_verifiable_v1/d3_verifiable_sft_train.jsonl', 35000),
]

def key_for(obj):
    msgs = obj.get('messages') or []
    user = '\n'.join(m.get('content', '') for m in msgs if m.get('role') == 'user')
    ans = str(obj.get('answer') or obj.get('gold_answer') or '')
    return hashlib.sha1((user + '\0' + ans).encode('utf-8', 'ignore')).hexdigest()

def family_task(obj):
    return obj.get('base_task_type') or obj.get('task_type') or 'unknown'

def source_key(obj):
    return obj.get('source_file') or obj.get('source_dataset') or obj.get('source') or 'unknown'

def load_rows(path):
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('messages'):
                rows.append(obj)
    return rows

def stratified_sample(rows, n, rng):
    buckets = defaultdict(list)
    for obj in rows:
        buckets[family_task(obj)].append(obj)
    selected = []
    used = set()
    tasks = sorted(buckets)
    base = n // len(tasks)
    rem = n % len(tasks)
    leftovers = []
    for i, task in enumerate(tasks):
        items = buckets[task][:]
        rng.shuffle(items)
        quota = base + (1 if i < rem else 0)
        take = items[:min(quota, len(items))]
        for obj in take:
            k = key_for(obj)
            if k not in used:
                selected.append(obj)
                used.add(k)
        leftovers.extend(items[min(quota, len(items)):])
    rng.shuffle(leftovers)
    for obj in leftovers:
        if len(selected) >= n:
            break
        k = key_for(obj)
        if k not in used:
            selected.append(obj)
            used.add(k)
    rng.shuffle(selected)
    return selected[:n]

def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = {
        'dataset': 'sft_light_110k_v1',
        'seed': args.seed,
        'parts': {},
        'policy': 'Stratified by base task type within D1/D2/D3; duplicates removed by user prompt + answer.',
    }
    for name, path, target_n in SPECS:
        rows = load_rows(path)
        sample = stratified_sample(rows, target_n, rng)
        part_rows = []
        for obj in sample:
            new_obj = dict(obj)
            new_obj['mixture_dataset'] = 'sft_light_110k_v1'
            new_obj['mixture_part'] = name
            part_rows.append(new_obj)
            all_rows.append(new_obj)
        outp = args.out / f'{name.lower()}_light_train.jsonl'
        write_jsonl(outp, part_rows)
        summary['parts'][name] = {
            'source': str(path),
            'target': target_n,
            'actual': len(part_rows),
            'file': str(outp),
            'task_counts': dict(Counter(family_task(o) for o in part_rows).most_common()),
            'source_counts_top': dict(Counter(source_key(o) for o in part_rows).most_common(20)),
        }
    rng.shuffle(all_rows)
    write_jsonl(args.out / 'sft_light_110k_train.jsonl', all_rows)
    summary['total_samples'] = len(all_rows)
    summary['total_task_counts'] = dict(Counter(family_task(o) for o in all_rows).most_common())
    summary['part_counts'] = dict(Counter(o.get('mixture_part') for o in all_rows).most_common())
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    readme = '# SFT Light 110K V1\n\nA lightweight selected SFT mixture for 8B/9B models.\n\n- D1: 35K\n- D2: 40K\n- D3-SFT: 35K\n- Total: 110K\n\nSampling is stratified by base task type within each part to reduce training time while preserving task coverage.\n'
    (args.out / 'README.md').write_text(readme, encoding='utf-8')
    print(json.dumps({'out': str(args.out), 'total': len(all_rows), 'part_counts': summary['part_counts']}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
