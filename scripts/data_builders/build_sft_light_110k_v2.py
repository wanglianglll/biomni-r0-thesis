from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/root/autodl-tmp/Biomni-main")
OUT = ROOT / "data/sft_light_110k_v2"
SEED = 20260428

# V2 deliberately shifts weight back to direct datalake tasks. D2/D3 are still
# useful for robustness, but too much contrast/template data made Llama learn the
# explanation scaffold rather than the biomedical answer behavior.
SPECS = [
    ("D1", ROOT / "data/sft_d1_datalake_v2/d1_datalake_train.jsonl", 70000),
    ("D2", ROOT / "data/sft_d2_contrast_v1/d2_contrast_train.jsonl", 20000),
    ("D3", ROOT / "data/sft_d3_verifiable_v1/d3_verifiable_sft_train.jsonl", 20000),
]

BAD_TEMPLATE_PHRASES = (
    "Key evidence",
    "same-task distractors",
    "supported answer",
    "matches the supported",
)


def key_for(obj: dict) -> str:
    msgs = obj.get("messages") or []
    user = "\n".join(m.get("content", "") for m in msgs if m.get("role") == "user")
    assistant = "\n".join(m.get("content", "") for m in msgs if m.get("role") == "assistant")
    return hashlib.sha1((user + "\0" + assistant).encode("utf-8", "ignore")).hexdigest()


def family_task(obj: dict) -> str:
    return obj.get("base_task_type") or obj.get("task_type") or "unknown"


def source_key(obj: dict) -> str:
    return obj.get("source_file") or obj.get("source_dataset") or obj.get("source") or "unknown"


def extract_final_answer(text: str) -> str:
    text = (text or "").strip()
    matches = re.findall(r"FINAL ANSWER:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        answer = matches[-1].strip()
    else:
        answer = text
    answer = answer.strip()
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
            "content": system or "You are Biomni, a biomedical assistant. Answer concisely and follow the requested output format.",
        },
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": f"FINAL ANSWER: {answer}"},
    ]
    cleaned["cleaning_policy"] = "assistant_final_answer_only"
    return cleaned


def sample_stream(path: Path, n: int, seed: int) -> list[dict]:
    heap = []
    seen = set()
    idx = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cleaned = clean_messages(obj)
            if cleaned is not None:
                key = key_for(cleaned)
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
    assistant_lines_without_final_answer = 0
    final_answers = Counter()
    for o in rows:
        row_text = json.dumps(o.get("messages", []), ensure_ascii=False)
        for phrase in BAD_TEMPLATE_PHRASES:
            phrase_counts[phrase] += row_text.count(phrase)
        for m in o.get("messages", []):
            if m.get("role") == "assistant":
                content = m.get("content", "")
                final_answers[extract_final_answer(content)[:120]] += 1
                assistant_lines_without_final_answer += sum(
                    1 for line in content.splitlines() if line and not line.startswith("FINAL ANSWER:")
                )
    return {
        "template_phrase_counts": dict(phrase_counts),
        "assistant_lines_without_final_answer": assistant_lines_without_final_answer,
        "top_final_answers": dict(final_answers.most_common(30)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = {
        "dataset": "sft_light_110k_v2",
        "seed": args.seed,
        "parts": {},
        "policy": (
            "D1-heavy lightweight mixture. Assistant messages are reduced to exactly "
            "one FINAL ANSWER line to avoid explanation-template overfitting."
        ),
    }
    for part_index, (name, path, target_n) in enumerate(SPECS):
        sample = sample_stream(path, target_n, args.seed + part_index)
        part_rows = []
        for obj in sample:
            new_obj = dict(obj)
            new_obj["mixture_dataset"] = "sft_light_110k_v2"
            new_obj["mixture_part"] = name
            part_rows.append(new_obj)
            all_rows.append(new_obj)
        outp = args.out / f"{name.lower()}_light_train.jsonl"
        write_jsonl(outp, part_rows)
        summary["parts"][name] = {
            "source": str(path),
            "target": target_n,
            "actual": len(part_rows),
            "file": str(outp),
            "task_counts": dict(Counter(family_task(o) for o in part_rows).most_common()),
            "source_counts_top": dict(Counter(source_key(o) for o in part_rows).most_common(20)),
            "audit": audit(part_rows),
        }

    random.Random(args.seed).shuffle(all_rows)
    write_jsonl(args.out / "sft_light_110k_train_messages.jsonl", all_rows)
    write_jsonl(args.out / "sft_light_110k_train.jsonl", all_rows)
    summary["total_samples"] = len(all_rows)
    summary["total_task_counts"] = dict(Counter(family_task(o) for o in all_rows).most_common())
    summary["part_counts"] = dict(Counter(o.get("mixture_part") for o in all_rows).most_common())
    summary["audit"] = audit(all_rows)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = """# SFT Light 110K V2

A cleaner lightweight SFT mixture for 8B/9B models.

- D1: 70K direct datalake-style samples
- D2: 20K contrast/correction samples
- D3: 20K verifiable samples
- Total: 110K

V2 removes explanation scaffolds from assistant messages. Every assistant
message is reduced to a single `FINAL ANSWER:` line, which should reduce
`Key evidence` / `supported answer` / distractor-template overfitting.
"""
    (args.out / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"out": str(args.out), "total": len(all_rows), "part_counts": summary["part_counts"], "audit": summary["audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
