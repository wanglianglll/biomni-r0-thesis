
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/root/autodl-tmp/Biomni-main')
DEFAULT_D1 = ROOT / 'data/sft_d1_datalake_v2/d1_datalake_all.jsonl'
DEFAULT_OUT = ROOT / 'data/sft_d2_contrast_v1'
SYSTEM = 'You are Biomni, a biomedical assistant. Use the provided structured evidence and answer concisely. End with exactly one FINAL ANSWER line.'
SEED = 20260427
TRAIN_RATIO = 0.98
VERIFY_PER_TASK = 6000
CORRECT_PER_TASK = 4000
MCQ_PER_TASK = 3000
POSITIVE_VERIFY_RATIO = 0.25

def clean(x: object) -> str:
    if x is None:
        return ''
    return re.sub(r'\s+', ' ', str(x)).strip()

def final_answer(text: str) -> str:
    m = re.findall(r'FINAL ANSWER:\s*(.+)', text or '', flags=re.I)
    return clean(m[-1]) if m else ''

def user_content(obj: dict) -> str:
    for m in obj.get('messages', []):
        if m.get('role') == 'user':
            return m.get('content', '')
    return ''

def sha(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8', 'ignore')).hexdigest()[:16]

def load_d1(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            task = obj.get('task_type')
            ans = clean(obj.get('answer')) or final_answer('\n'.join(m.get('content', '') for m in obj.get('messages', []) if m.get('role') == 'assistant'))
            user = user_content(obj)
            if not task or not ans or not user:
                continue
            if len(ans) > 240:
                continue
            rows.append({
                'task_type': task,
                'source': obj.get('source', ''),
                'source_file': obj.get('source_file', ''),
                'source_row_id': obj.get('source_row_id'),
                'answer': ans,
                'user': user,
                'metadata': obj.get('metadata') or {},
            })
    return rows

def build_answer_pools(rows: list[dict]) -> dict[str, list[str]]:
    pools = defaultdict(list)
    seen = defaultdict(set)
    for r in rows:
        key = r['task_type']
        ans = r['answer']
        if ans and ans not in seen[key]:
            pools[key].append(ans)
            seen[key].add(ans)
    return pools

def pick_wrong(r: dict, pools: dict[str, list[str]], rng: random.Random) -> str:
    pool = pools.get(r['task_type'], [])
    if len(pool) < 2:
        return ''
    answer_l = r['answer'].lower()
    for _ in range(80):
        cand = rng.choice(pool)
        if cand.lower() != answer_l:
            return cand
    return ''

def make_record(kind: str, base: dict, user: str, answer: str, evidence: str, metadata: dict, flags: list[str]) -> dict:
    assistant = f'Key evidence: {clean(evidence)}\nFINAL ANSWER: {clean(answer)}'
    return {
        'dataset': 'D2_contrast_v1',
        'task_type': kind,
        'base_task_type': base['task_type'],
        'source': base['source'],
        'source_file': base['source_file'],
        'source_row_id': base['source_row_id'],
        'answer': clean(answer),
        'metadata': metadata,
        'quality_flags': flags,
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': assistant},
        ],
    }

def make_verify(base: dict, proposed: str, is_supported: bool) -> dict:
    label = 'YES' if is_supported else 'NO'
    user = (
        'Decide whether the proposed answer is supported by the structured biomedical evidence.\n'
        'Answer only YES or NO after considering the evidence.\n\n'
        f'{base["user"]}\n\n'
        f'Proposed answer: {proposed}'
    )
    ev = f'The evidence supports {base["answer"]}; the proposed answer {proposed} is ' + ('supported.' if is_supported else 'not supported.')
    return make_record(
        'd2_verify_supported_answer', base, user, label, ev,
        {'correct_answer': base['answer'], 'proposed_answer': proposed, 'is_supported': is_supported, 'base_task_type': base['task_type']},
        ['contrastive_verification', 'positive_control' if is_supported else 'negative_control']
    )

def make_correction(base: dict, wrong: str) -> dict:
    user = (
        'The proposed answer below may be wrong. Use the evidence to return the corrected answer.\n\n'
        f'{base["user"]}\n\n'
        f'Proposed answer to check: {wrong}\n\n'
        'Return the corrected answer supported by the evidence.'
    )
    ev = f'The proposed answer {wrong} conflicts with the evidence; the supported answer is {base["answer"]}.'
    return make_record(
        'd2_correct_wrong_answer', base, user, base['answer'], ev,
        {'correct_answer': base['answer'], 'wrong_answer': wrong, 'base_task_type': base['task_type']},
        ['contrastive_correction', 'negative_control']
    )

def make_mcq(base: dict, distractors: list[str], rng: random.Random) -> dict:
    choices = [base['answer']] + distractors[:3]
    rng.shuffle(choices)
    letters = ['A', 'B', 'C', 'D']
    correct_idx = choices.index(base['answer'])
    choice_text = '\n'.join(f'{letters[i]}. {c}' for i, c in enumerate(choices))
    user = (
        'Choose the option that is best supported by the structured biomedical evidence.\n\n'
        f'{base["user"]}\n\n'
        f'Options:\n{choice_text}\n\n'
        'Return the letter and answer.'
    )
    final = f'{letters[correct_idx]}. {base["answer"]}'
    ev = f'Option {letters[correct_idx]} matches the supported answer {base["answer"]}; the other options are same-task distractors.'
    return make_record(
        'd2_select_from_distractors', base, user, final, ev,
        {'correct_answer': base['answer'], 'choices': dict(zip(letters, choices)), 'correct_letter': letters[correct_idx], 'base_task_type': base['task_type']},
        ['contrastive_selection', 'same_task_distractors']
    )

def sample_by_task(rows: list[dict], per_task: int, rng: random.Random) -> list[dict]:
    by_task = defaultdict(list)
    for r in rows:
        by_task[r['task_type']].append(r)
    selected = []
    for _, items in sorted(by_task.items()):
        items = items[:]
        rng.shuffle(items)
        selected.extend(items[:min(per_task, len(items))])
    rng.shuffle(selected)
    return selected

def dedup(records: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for r in records:
        user = next(m['content'] for m in r['messages'] if m['role'] == 'user')
        key = sha(user + '\0' + r['answer'])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def write_jsonl(path: Path, data: list[dict]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--d1', type=Path, default=DEFAULT_D1)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_d1(args.d1)
    pools = build_answer_pools(rows)
    eligible = [r for r in rows if len(pools.get(r['task_type'], [])) >= 4]

    records = []
    for r in sample_by_task(eligible, VERIFY_PER_TASK, rng):
        is_pos = rng.random() < POSITIVE_VERIFY_RATIO
        proposed = r['answer'] if is_pos else pick_wrong(r, pools, rng)
        if proposed:
            records.append(make_verify(r, proposed, is_pos))

    for r in sample_by_task(eligible, CORRECT_PER_TASK, rng):
        wrong = pick_wrong(r, pools, rng)
        if wrong:
            records.append(make_correction(r, wrong))

    for r in sample_by_task(eligible, MCQ_PER_TASK, rng):
        pool = [x for x in pools[r['task_type']] if x.lower() != r['answer'].lower()]
        if len(pool) < 3:
            continue
        records.append(make_mcq(r, rng.sample(pool, 3), rng))

    records = dedup(records)
    rng.shuffle(records)
    split = int(len(records) * TRAIN_RATIO)
    train, val = records[:split], records[split:]

    write_jsonl(args.out / 'd2_contrast_all.jsonl', records)
    write_jsonl(args.out / 'd2_contrast_train.jsonl', train)
    write_jsonl(args.out / 'd2_contrast_val.jsonl', val)

    summary = {
        'dataset': 'D2_contrast_v1',
        'description': 'Contrastive SFT data derived from D1 high-priority datalake samples. It contains supported-answer verification, wrong-answer correction, and same-task multiple-choice distractor samples.',
        'source_d1': str(args.d1),
        'seed': args.seed,
        'total_samples': len(records),
        'train_samples': len(train),
        'val_samples': len(val),
        'd1_loaded_samples': len(rows),
        'eligible_d1_samples': len(eligible),
        'task_type_counts': dict(Counter(r['task_type'] for r in records).most_common()),
        'base_task_type_counts': dict(Counter(r['base_task_type'] for r in records).most_common()),
        'source_file_counts': dict(Counter(r['source_file'] for r in records).most_common()),
        'quality_flag_counts': dict(Counter(f for r in records for f in r.get('quality_flags', [])).most_common()),
        'construction_policy': {
            'answer_source': 'D1 structured evidence answer field',
            'negative_source': 'same-base-task answer pool excluding the true answer',
            'accuracy_guardrail': 'No free-form label generation; every correct answer is copied from D1, and every distractor comes from the same task type.',
            'train_ratio': TRAIN_RATIO,
        },
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    readme = "\n".join([
        '# D2 Contrast SFT Dataset',
        '',
        'This dataset is derived from `D1_datalake_v2` and is designed to teach evidence checking rather than raw biomedical fact memorization.',
        '',
        '## Files',
        '',
        '- `d2_contrast_all.jsonl`: full dataset',
        '- `d2_contrast_train.jsonl`: training split',
        '- `d2_contrast_val.jsonl`: validation split',
        '- `summary.json`: construction metadata and distributions',
        '',
        '## Counts',
        '',
        f'- Total: {len(records)}',
        f'- Train: {len(train)}',
        f'- Val: {len(val)}',
        '',
        '## D2 Task Types',
        '',
        '- `d2_verify_supported_answer`: decide whether a proposed answer is supported by the evidence.',
        '- `d2_correct_wrong_answer`: correct an intentionally wrong same-task answer.',
        '- `d2_select_from_distractors`: choose the correct answer from same-task distractors.',
        '',
        '## Accuracy Policy',
        '',
        'D2 does not invent labels with an LLM. Correct answers are copied from D1, and distractors are sampled from answers of the same base task type. This gives us useful negative controls while keeping labels mechanically verifiable.',
        ''
    ])
    (args.out / 'README.md').write_text(readme, encoding='utf-8')
    print(json.dumps({
        'out': str(args.out),
        'total': len(records),
        'train': len(train),
        'val': len(val),
        'task_type_counts': summary['task_type_counts'],
        'base_task_type_counts_top': dict(list(summary['base_task_type_counts'].items())[:20]),
    }, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
