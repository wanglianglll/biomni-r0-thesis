
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path('/root/autodl-tmp/Biomni-main')
DEFAULT_D1 = ROOT / 'data/sft_d1_datalake_v2/d1_datalake_all.jsonl'
DEFAULT_D2 = ROOT / 'data/sft_d2_contrast_v1/d2_contrast_all.jsonl'
DEFAULT_OUT = ROOT / 'data/sft_d3_verifiable_v1'
SEED = 20260427
TRAIN_RATIO = 0.98
SYSTEM = 'You are Biomni, a biomedical assistant. Use the provided structured evidence and answer concisely. End with exactly one FINAL ANSWER line.'

D1_LIMITS = {
    'gwas_trait_to_gene': 8000,
    'gene_record_to_symbol': 8000,
    'omim_record_to_mim': 8000,
    'protein_atlas_gene_profile': 6000,
    'drug_interaction_level': 6000,
    'mirna_to_target_gene': 5000,
    'gene_to_disorder': 5000,
    'variant_record_to_rsid': 6000,
    'gtex_expression_gene': 5000,
    'depmap_gene_dependency': 2500,
    'depmap_gene_effect': 2500,
    'disease_to_genes': 5000,
    'compound_to_targets': 4000,
    'geneset_to_genes': 5000,
    'gene_interaction_pair': 5000,
}
D2_LIMITS = {
    'd2_verify_supported_answer': 20000,
    'd2_correct_wrong_answer': 12000,
    'd2_select_from_distractors': 12000,
}

EXACT_GENE_TASKS = {
    'gwas_trait_to_gene', 'gene_record_to_symbol', 'protein_atlas_gene_profile',
    'mirna_to_target_gene', 'gtex_expression_gene', 'depmap_gene_dependency', 'depmap_gene_effect'
}
EXACT_CASE_INSENSITIVE_TASKS = {'gene_to_disorder', 'drug_interaction_level'}
RSID_TASKS = {'variant_record_to_rsid'}
MIM_TASKS = {'omim_record_to_mim'}
SET_TASKS = {'disease_to_genes', 'compound_to_targets', 'geneset_to_genes', 'gene_interaction_pair'}


def clean(x: object) -> str:
    if x is None:
        return ''
    return re.sub(r'\s+', ' ', str(x)).strip()


def get_msg(obj: dict, role: str) -> str:
    for m in obj.get('messages', []):
        if m.get('role') == role:
            return m.get('content', '')
    return ''


def normalize_token(s: str) -> str:
    return clean(s).strip(' .;:,').upper()


def split_answer(ans: str) -> list[str]:
    ans = clean(ans)
    if ' -- ' in ans:
        parts = ans.split(' -- ')
    else:
        parts = re.split(r'[,;|]\s*', ans)
    return [clean(p).strip(' .;:') for p in parts if clean(p).strip(' .;:')]


def infer_verifier(obj: dict) -> dict | None:
    task = obj.get('task_type', '')
    base_task = obj.get('base_task_type', '')
    ans = clean(obj.get('answer'))
    md = obj.get('metadata') or {}
    if not ans:
        return None
    if task == 'd2_verify_supported_answer':
        return {'type': 'boolean_yes_no', 'gold': ans.upper(), 'positive_values': ['YES'], 'negative_values': ['NO'], 'normalize': ['strip', 'uppercase']}
    if task == 'd2_select_from_distractors':
        letter = clean(md.get('correct_letter'))
        correct = clean(md.get('correct_answer'))
        if not letter or not correct:
            return None
        return {'type': 'multiple_choice', 'gold_letter': letter, 'gold_answer': correct, 'normalize': ['strip', 'uppercase_first_letter'], 'accept': [letter, f'{letter}. {correct}', correct]}
    if task == 'd2_correct_wrong_answer':
        base_task = base_task or md.get('base_task_type', '')
        return verifier_for_base(base_task, ans)
    return verifier_for_base(task, ans)


def verifier_for_base(task: str, ans: str) -> dict | None:
    if task in EXACT_GENE_TASKS:
        return {'type': 'exact_symbol', 'gold': normalize_token(ans), 'normalize': ['strip', 'uppercase', 'remove_trailing_punctuation']}
    if task in EXACT_CASE_INSENSITIVE_TASKS:
        return {'type': 'exact_text_ci', 'gold': clean(ans).lower(), 'normalize': ['strip', 'lowercase', 'remove_trailing_punctuation']}
    if task in RSID_TASKS:
        m = re.search(r'rs\d+', ans, re.I)
        if not m:
            return None
        return {'type': 'rsid', 'gold': m.group(0).lower(), 'pattern': r'rs\d+', 'normalize': ['strip', 'lowercase']}
    if task in MIM_TASKS:
        m = re.search(r'\d{6}', ans)
        if not m:
            return None
        return {'type': 'mim_number', 'gold': m.group(0), 'pattern': r'\d{6}', 'normalize': ['extract_6_digit_mim']}
    if task in SET_TASKS:
        items = [normalize_token(x) for x in split_answer(ans)]
        items = list(dict.fromkeys([x for x in items if x]))
        if not items:
            return None
        return {'type': 'set_exact_or_subset', 'gold_items': items, 'min_recall': 1.0, 'normalize': ['split_commas_semicolons_pipes_or_pair_dash', 'uppercase', 'strip_punctuation']}
    return None


def make_sft(obj: dict, verifier: dict, source_dataset: str) -> dict:
    user = get_msg(obj, 'user')
    assistant = get_msg(obj, 'assistant')
    return {
        'dataset': 'D3_verifiable_v1',
        'split_family': 'sft',
        'source_dataset': source_dataset,
        'task_type': obj.get('task_type'),
        'base_task_type': obj.get('base_task_type', obj.get('task_type')),
        'source': obj.get('source', ''),
        'source_file': obj.get('source_file', ''),
        'source_row_id': obj.get('source_row_id'),
        'answer': clean(obj.get('answer')),
        'gold_answer': clean(obj.get('answer')),
        'verifier': verifier,
        'metadata': obj.get('metadata') or {},
        'quality_flags': list(dict.fromkeys((obj.get('quality_flags') or []) + ['verifiable_answer'])),
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': assistant},
        ],
    }


def make_rl(obj: dict, verifier: dict, source_dataset: str) -> dict:
    user = get_msg(obj, 'user')
    prompt = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': user},
    ]
    return {
        'dataset': 'D3_verifiable_v1',
        'split_family': 'rl',
        'source_dataset': source_dataset,
        'task_type': obj.get('task_type'),
        'base_task_type': obj.get('base_task_type', obj.get('task_type')),
        'source': obj.get('source', ''),
        'source_file': obj.get('source_file', ''),
        'source_row_id': obj.get('source_row_id'),
        'prompt': prompt,
        'gold_answer': clean(obj.get('answer')),
        'verifier': verifier,
        'reward_spec': {'format_reward': 'requires exactly one FINAL ANSWER line', 'answer_reward': verifier},
        'metadata': obj.get('metadata') or {},
        'quality_flags': list(dict.fromkeys((obj.get('quality_flags') or []) + ['rl_ready', 'verifiable_answer'])),
    }


def load_limited(path: Path, limits: dict[str, int], rng: random.Random, source_dataset: str) -> list[tuple[dict, str]]:
    buckets = {k: [] for k in limits}
    with path.open(encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            task = obj.get('task_type')
            if task in buckets:
                buckets[task].append(obj)
    selected = []
    for task, rows in buckets.items():
        rng.shuffle(rows)
        selected.extend((o, source_dataset) for o in rows[:limits[task]])
    rng.shuffle(selected)
    return selected


def dedup_sft(records: list[dict]) -> list[dict]:
    seen = set(); out = []
    for r in records:
        user = get_msg(r, 'user')
        key = hashlib.sha1((user + '\0' + r['gold_answer']).encode('utf-8', 'ignore')).hexdigest()
        if key in seen:
            continue
        seen.add(key); out.append(r)
    return out


def write_jsonl(path: Path, data: list[dict]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--d1', type=Path, default=DEFAULT_D1)
    ap.add_argument('--d2', type=Path, default=DEFAULT_D2)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    selected = load_limited(args.d1, D1_LIMITS, rng, 'D1_datalake_v2')
    selected += load_limited(args.d2, D2_LIMITS, rng, 'D2_contrast_v1')
    rng.shuffle(selected)

    sft = []
    rl = []
    skipped = Counter()
    for obj, source_dataset in selected:
        verifier = infer_verifier(obj)
        if verifier is None:
            skipped[obj.get('task_type', '')] += 1
            continue
        sft.append(make_sft(obj, verifier, source_dataset))
        rl.append(make_rl(obj, verifier, source_dataset))

    sft = dedup_sft(sft)
    # Keep RL aligned with SFT by prompt+gold dedup separately.
    seen = set(); rl_dedup = []
    for r in rl:
        user = next(m['content'] for m in r['prompt'] if m['role'] == 'user')
        key = hashlib.sha1((user + '\0' + r['gold_answer']).encode('utf-8', 'ignore')).hexdigest()
        if key in seen:
            continue
        seen.add(key); rl_dedup.append(r)
    rl = rl_dedup
    rng.shuffle(sft); rng.shuffle(rl)

    sft_split = int(len(sft) * TRAIN_RATIO)
    rl_split = int(len(rl) * TRAIN_RATIO)
    write_jsonl(args.out / 'd3_verifiable_sft_all.jsonl', sft)
    write_jsonl(args.out / 'd3_verifiable_sft_train.jsonl', sft[:sft_split])
    write_jsonl(args.out / 'd3_verifiable_sft_val.jsonl', sft[sft_split:])
    write_jsonl(args.out / 'd3_verifiable_rl_all.jsonl', rl)
    write_jsonl(args.out / 'd3_verifiable_rl_train.jsonl', rl[:rl_split])
    write_jsonl(args.out / 'd3_verifiable_rl_val.jsonl', rl[rl_split:])

    summary = {
        'dataset': 'D3_verifiable_v1',
        'description': 'Verifiable-answer data for SFT and RL/GRPO reward development. It is derived from high-confidence D1 and contrastive D2 samples and includes machine-checkable verifier specs.',
        'seed': args.seed,
        'sft_total': len(sft),
        'sft_train': sft_split,
        'sft_val': len(sft) - sft_split,
        'rl_total': len(rl),
        'rl_train': rl_split,
        'rl_val': len(rl) - rl_split,
        'source_dataset_counts': dict(Counter(r['source_dataset'] for r in sft).most_common()),
        'task_type_counts': dict(Counter(r['task_type'] for r in sft).most_common()),
        'base_task_type_counts': dict(Counter(r['base_task_type'] for r in sft).most_common()),
        'verifier_type_counts': dict(Counter(r['verifier']['type'] for r in sft).most_common()),
        'skipped_counts': dict(skipped.most_common()),
        'construction_policy': {
            'correct_answer_source': 'Copied from D1/D2 gold answer fields, never LLM-generated.',
            'verifiability': 'Each item has a verifier object with exact, set, regex, boolean, or multiple-choice matching rules.',
            'rl_usage': 'Use d3_verifiable_rl_*.jsonl for prompt-only RL/GRPO data with reward_spec.',
            'sft_usage': 'Use d3_verifiable_sft_*.jsonl for supervised warm-up with the same verifier metadata.',
        },
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    readme = '\n'.join([
        '# D3 Verifiable Dataset', '',
        'D3 is the verifiable-answer layer for later SFT evaluation and RL/GRPO reward design.', '',
        '## Files', '',
        '- `d3_verifiable_sft_all.jsonl`, `d3_verifiable_sft_train.jsonl`, `d3_verifiable_sft_val.jsonl`',
        '- `d3_verifiable_rl_all.jsonl`, `d3_verifiable_rl_train.jsonl`, `d3_verifiable_rl_val.jsonl`',
        '- `summary.json`', '',
        '## Key Idea', '',
        'Every item includes `gold_answer` and a `verifier` object. The verifier can be exact symbol matching, case-insensitive text matching, rsID extraction, MIM number extraction, set matching, YES/NO matching, or multiple-choice matching.', '',
        'Correct answers are copied from D1/D2 structured labels. No labels are generated by an LLM.', ''
    ])
    (args.out / 'README.md').write_text(readme, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:8000])

if __name__ == '__main__':
    main()
