from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/root/autodl-tmp/Biomni-main")
OUT = ROOT / "data/sft_targeted_100k_v2"
SEED = 20260428

SPECS = [
    ("D1", ROOT / "data/sft_d1_datalake_v2/d1_datalake_train.jsonl", 45000),
    ("D2", ROOT / "data/sft_d2_contrast_v1/d2_contrast_train.jsonl", 4000),
    ("D3", ROOT / "data/sft_d3_verifiable_v1/d3_verifiable_sft_train.jsonl", 4000),
    ("D4", ROOT / "data/sft_d4_rare_disease_v2/d4_rare_disease_train.jsonl", 10000),
    ("D5", ROOT / "data/sft_d5_gwas_variant_v2/d5_gwas_variant_train.jsonl", 20000),
    ("D6", ROOT / "data/sft_d6_gwas_causal_gene_v1/d6_gwas_causal_gene_train.jsonl", 15000),
]

BAD_TEMPLATE_PHRASES = (
    "Key evidence",
    "same-task distractors",
    "supported answer",
    "matches the supported",
)


def extract_final_answer(text: str) -> str:
    text = (text or "").strip()
    matches = re.findall(r"FINAL ANSWER:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        answer = matches[-1].strip()
    else:
        answer = text
    answer = re.sub(r"^\s*(?:\[ANSWER\])\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s*(?:\[/ANSWER\])\s*$", "", answer, flags=re.IGNORECASE)
    return answer.strip()


def clean_messages(obj: dict) -> dict | None:
    messages = obj.get("messages") or []
    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    assistant = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
    if not user or not assistant:
        return None
    answer = extract_final_answer(assistant)
    if not answer:
        return None
    cleaned = dict(obj)
    cleaned["messages"] = [
        {
            "role": "system",
            "content": system
            or "You are Biomni, a biomedical assistant. Answer concisely and follow the requested output format.",
        },
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": f"FINAL ANSWER: {answer}"},
    ]
    cleaned["cleaning_policy"] = "assistant_final_answer_only"
    return cleaned


def stable_key(obj: dict) -> str:
    messages = obj.get("messages") or []
    user = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
    assistant = "\n".join(m.get("content", "") for m in messages if m.get("role") == "assistant")
    return hashlib.sha1((user + "\0" + assistant).encode("utf-8", "ignore")).hexdigest()


def family_task(obj: dict) -> str:
    return obj.get("task_type") or obj.get("base_task_type") or obj.get("source_task_type") or "unknown"


def normalize_schema(obj: dict, mixture_part: str) -> dict:
    return {
        "messages": obj["messages"],
        "mixture_dataset": "sft_targeted_100k_v2",
        "mixture_part": mixture_part,
        "task_type": family_task(obj),
        "source_dataset": str(obj.get("source_dataset") or obj.get("source_file") or obj.get("source") or "unknown"),
    }


def sample_stream(path: Path, n: int, seed: int) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    heap = []
    seen = set()
    idx = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cleaned = clean_messages(obj)
            if cleaned is None:
                continue
            key = stable_key(cleaned)
            if key in seen:
                continue
            seen.add(key)
            score = int(hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest(), 16)
            entry = (-score, idx, cleaned)
            idx += 1
            if len(heap) < n:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
    rows = [entry[2] for entry in heap]
    random.Random(seed).shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def audit(rows: list[dict]) -> dict:
    phrase_counts = Counter()
    assistant_lines_without_final = 0
    final_answers = Counter()
    task_counts = Counter()
    for obj in rows:
        row_text = json.dumps(obj.get("messages", []), ensure_ascii=False)
        for phrase in BAD_TEMPLATE_PHRASES:
            phrase_counts[phrase] += row_text.count(phrase)
        task_counts[family_task(obj)] += 1
        for msg in obj.get("messages", []):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                final_answers[extract_final_answer(content)[:120]] += 1
                assistant_lines_without_final += sum(
                    1 for line in content.splitlines() if line and not line.startswith("FINAL ANSWER:")
                )
    return {
        "rows": len(rows),
        "template_phrase_counts": dict(phrase_counts),
        "assistant_lines_without_final_answer": assistant_lines_without_final,
        "top_final_answers": dict(final_answers.most_common(30)),
        "task_counts_top": dict(task_counts.most_common(50)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = {
        "dataset": "sft_targeted_100k_v2",
        "seed": args.seed,
        "policy": "D1-heavy mixture with calibrated D4/D5 v2 and reduced D6. Assistant messages are single FINAL ANSWER lines.",
        "parts": {},
    }
    global_seen = set()
    for index, (name, path, target_n) in enumerate(SPECS):
        rows = sample_stream(path, target_n, args.seed + index)
        part_rows = []
        for row in rows:
            key = stable_key(row)
            if key in global_seen:
                continue
            global_seen.add(key)
            new_row = normalize_schema(row, name)
            part_rows.append(new_row)
            all_rows.append(new_row)
        out_path = args.out / f"{name.lower()}_targeted_train.jsonl"
        write_jsonl(out_path, part_rows)
        summary["parts"][name] = {
            "source": str(path),
            "target": target_n,
            "actual": len(part_rows),
            "file": str(out_path),
            "audit": audit(part_rows),
        }

    random.Random(args.seed).shuffle(all_rows)
    write_jsonl(args.out / "sft_targeted_100k_train_messages.jsonl", all_rows)
    write_jsonl(args.out / "sft_targeted_100k_train.jsonl", all_rows)
    summary["total_samples"] = len(all_rows)
    summary["part_counts"] = dict(Counter(row.get("mixture_part") for row in all_rows).most_common())
    summary["audit"] = audit(all_rows)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        "# SFT Targeted 100K V2\n\n"
        "A short-validation SFT mixture using calibrated D4/D5 v2 and a smaller D6 share.\n\n"
        "- D1: 45K direct datalake samples\n"
        "- D2: 4K contrast samples\n"
        "- D3: 4K verifiable samples\n"
        "- D4: 10K rare-disease targeted v2 samples\n"
        "- D5: 20K GWAS variant-prioritization targeted v2 samples\n"
        "- D6: 15K GWAS causal-gene targeted samples\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "total": len(all_rows), "part_counts": summary["part_counts"], "audit": summary["audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
