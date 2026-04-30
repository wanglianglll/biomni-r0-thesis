from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path("/root/autodl-tmp/Biomni-main")
DATA_LAKE = ROOT / "data/biomni_data/data_lake"
OUT = ROOT / "data/sft_d5_gwas_variant_v1"
SEED = 20260428

SYSTEM_PROMPT = (
    "You are Biomni, a biomedical assistant. Answer concisely and follow the requested output format."
)


def norm_text(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_rsids(value: object) -> list[str]:
    seen = set()
    out = []
    for match in re.findall(r"rs\d+", str(value or ""), flags=re.IGNORECASE):
        rsid = match.lower()
        if rsid not in seen:
            seen.add(rsid)
            out.append(rsid)
    return out


def extract_eval_variant_exclusions(paths: Iterable[Path]) -> tuple[set[str], set[str], set[str]]:
    traits = set()
    candidate_rsids = set()
    answer_rsids = set()
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "task_name" not in df.columns:
            continue
        sub = df[df["task_name"].eq("gwas_variant_prioritization")]
        for row in sub.itertuples(index=False):
            prompt = getattr(row, "prompt", "")
            match = re.search(r"GWAS phenotype:\s*(.+?)\nVariants:", prompt, flags=re.DOTALL)
            if match:
                traits.add(norm_text(match.group(1)))
            candidate_rsids.update(extract_rsids(prompt))
            answer_rsids.update(extract_rsids(getattr(row, "answer", "")))
    return traits, candidate_rsids, answer_rsids


def stable_key(obj: dict) -> str:
    messages = obj.get("messages") or []
    text = "\n".join(m.get("content", "") for m in messages)
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def choose_primary_rsid(row: pd.Series) -> str | None:
    risk = extract_rsids(row.get("STRONGEST SNP-RISK ALLELE"))
    if risk:
        return risk[0]
    snps = extract_rsids(row.get("SNPS"))
    if snps:
        return snps[0]
    current = row.get("SNP_ID_CURRENT")
    if pd.notna(current):
        try:
            return f"rs{int(float(current))}".lower()
        except Exception:
            return None
    return None


def score_row(row: pd.Series) -> float | None:
    mlog = row.get("PVALUE_MLOG")
    if pd.notna(mlog):
        try:
            return float(mlog)
        except Exception:
            pass
    pvalue = row.get("P-VALUE")
    if pd.notna(pvalue):
        try:
            p = float(pvalue)
            if p > 0:
                return -1.0 * __import__("math").log10(p)
        except Exception:
            return None
    return None


def read_gwas_groups(
    path: Path,
    excluded_traits: set[str],
    excluded_rsids: set[str],
) -> dict[str, list[dict]]:
    df = pd.read_pickle(path)
    by_trait_rsid: dict[str, dict[str, dict]] = defaultdict(dict)
    for _, row in df.iterrows():
        trait = str(row.get("DISEASE/TRAIT") or "").strip()
        trait_norm = norm_text(trait)
        if not trait or trait_norm in excluded_traits:
            continue
        rsid = choose_primary_rsid(row)
        if not rsid or rsid in excluded_rsids:
            continue
        score = score_row(row)
        if score is None:
            continue
        if str(row.get("CNV") or "").upper() == "Y":
            continue
        record = {
            "trait": trait,
            "trait_norm": trait_norm,
            "rsid": rsid,
            "score": score,
            "p_value": float(row.get("P-VALUE")) if pd.notna(row.get("P-VALUE")) else None,
            "study": str(row.get("STUDY") or ""),
            "pubmedid": str(row.get("PUBMEDID") or ""),
            "mapped_gene": str(row.get("MAPPED_GENE") or ""),
            "context": str(row.get("CONTEXT") or ""),
        }
        old = by_trait_rsid[trait_norm].get(rsid)
        if old is None or record["score"] > old["score"]:
            by_trait_rsid[trait_norm][rsid] = record
    groups = {}
    for trait_norm, rs_map in by_trait_rsid.items():
        variants = sorted(rs_map.values(), key=lambda x: (-x["score"], x["rsid"]))
        if len(variants) >= 4:
            groups[trait_norm] = variants
    return groups


def make_user_prompt(trait: str, candidates: list[str]) -> str:
    return (
        "Your task is to identify the most promising variant associated wtih a given GWAS phenotype for futher examination.\n"
        "From the list, prioritize the top associated variant (matching one of the given variant).\n"
        f"GWAS phenotype: {trait}\n"
        f"Variants: {', '.join(candidates)}"
    )


def make_row(trait: str, positive: dict, candidates: list[str], variant_index: int) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt(trait, candidates)},
            {"role": "assistant", "content": f"FINAL ANSWER: {positive['rsid']}"},
        ],
        "dataset": "sft_d5_gwas_variant_v1",
        "task_type": "D5-gwas-variant-prioritization",
        "source_dataset": "biomni_data_lake/gwas_catalog.pkl",
        "source_policy": "Within a GWAS trait, choose the candidate rsID with the strongest association score; distractors are weaker same-trait variants.",
        "gwas_trait": trait,
        "answer_rsid": positive["rsid"],
        "answer_score": positive["score"],
        "answer_p_value": positive["p_value"],
        "candidate_count": len(candidates),
        "variant_index": variant_index,
    }


def build_rows(groups: dict[str, list[dict]], max_samples: int, seed: int, variants_per_trait: int) -> list[dict]:
    rng = random.Random(seed)
    trait_keys = list(groups)
    rng.shuffle(trait_keys)
    rows = []
    seen = set()
    for trait_norm in trait_keys:
        variants = groups[trait_norm]
        top_limit = min(25, max(1, len(variants) // 3))
        positive_pool = variants[:top_limit]
        for variant_index in range(variants_per_trait):
            positive = rng.choice(positive_pool)
            weaker = [v for v in variants if v["score"] < positive["score"] and v["rsid"] != positive["rsid"]]
            if len(weaker) < 3:
                continue
            candidate_n = min(len(weaker) + 1, rng.randint(5, 11))
            distractors = rng.sample(weaker, candidate_n - 1)
            candidates = [v["rsid"] for v in distractors] + [positive["rsid"]]
            rng.shuffle(candidates)
            row = make_row(positive["trait"], positive, candidates, variant_index)
            key = stable_key(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(rows) >= max_samples:
                return rows
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def audit(
    rows: list[dict],
    eval_traits: set[str],
    eval_candidate_rsids: set[str],
    eval_answer_rsids: set[str],
) -> dict:
    answers = Counter()
    traits = Counter()
    candidate_sizes = Counter()
    phrase_counts = Counter()
    all_candidate_rsids = set()
    bad_phrases = ("Key evidence", "same-task distractors", "supported answer")
    for row in rows:
        text = json.dumps(row["messages"], ensure_ascii=False)
        for phrase in bad_phrases:
            phrase_counts[phrase] += text.count(phrase)
        answers[row["answer_rsid"]] += 1
        traits[norm_text(row["gwas_trait"])] += 1
        candidate_sizes[row["candidate_count"]] += 1
        all_candidate_rsids.update(extract_rsids(row["messages"][1]["content"]))
    return {
        "rows": len(rows),
        "unique_traits": len(traits),
        "unique_answer_rsids": len(answers),
        "unique_candidate_rsids": len(all_candidate_rsids),
        "top_answer_rsids": dict(answers.most_common(20)),
        "top_traits": dict(traits.most_common(20)),
        "candidate_count_distribution": dict(sorted(candidate_sizes.items())),
        "template_phrase_counts": dict(phrase_counts),
        "eval_trait_overlap_count": len(set(traits) & eval_traits),
        "eval_trait_overlap": sorted(set(traits) & eval_traits)[:50],
        "eval_candidate_rsid_overlap_count": len(all_candidate_rsids & eval_candidate_rsids),
        "eval_candidate_rsid_overlap": sorted(all_candidate_rsids & eval_candidate_rsids)[:50],
        "eval_answer_rsid_overlap_count": len(set(answers) & eval_answer_rsids),
        "eval_answer_rsid_overlap": sorted(set(answers) & eval_answer_rsids)[:50],
    }


def validate_rows(rows: list[dict]) -> list[tuple[int, str]]:
    problems = []
    for i, row in enumerate(rows, 1):
        messages = row.get("messages") or []
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            problems.append((i, "roles"))
            continue
        user = messages[1].get("content", "")
        assistant = messages[2].get("content", "")
        candidates = extract_rsids(user)
        answer = row.get("answer_rsid")
        if not assistant.startswith("FINAL ANSWER: rs"):
            problems.append((i, "assistant_format"))
        if answer not in candidates:
            problems.append((i, "answer_not_in_candidates"))
        if len(candidates) < 4:
            problems.append((i, "candidate_count"))
        if len(set(candidates)) != len(candidates):
            problems.append((i, "duplicate_candidates"))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-samples", type=int, default=30000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--variants-per-trait", type=int, default=2)
    parser.add_argument("--allow-eval-overlap", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    eval_traits, eval_candidate_rsids, eval_answer_rsids = extract_eval_variant_exclusions(
        [ROOT / "data/biomni_eval1_dataset.parquet"]
    )
    excluded_traits = set() if args.allow_eval_overlap else eval_traits
    excluded_rsids = set() if args.allow_eval_overlap else eval_candidate_rsids
    groups = read_gwas_groups(DATA_LAKE / "gwas_catalog.pkl", excluded_traits, excluded_rsids)
    rows = build_rows(groups, args.max_samples, args.seed, args.variants_per_trait)
    random.Random(args.seed).shuffle(rows)
    val_n = max(1, int(len(rows) * args.val_ratio)) if rows else 0
    val_rows = rows[:val_n]
    train_rows = rows[val_n:]

    write_jsonl(args.out / "d5_gwas_variant_all.jsonl", rows)
    write_jsonl(args.out / "d5_gwas_variant_train.jsonl", train_rows)
    write_jsonl(args.out / "d5_gwas_variant_val.jsonl", val_rows)

    problems = validate_rows(rows)
    summary = {
        "dataset": "sft_d5_gwas_variant_v1",
        "seed": args.seed,
        "max_samples": args.max_samples,
        "variants_per_trait": args.variants_per_trait,
        "files": {
            "all": str(args.out / "d5_gwas_variant_all.jsonl"),
            "train": str(args.out / "d5_gwas_variant_train.jsonl"),
            "val": str(args.out / "d5_gwas_variant_val.jsonl"),
        },
        "source_files": [str(DATA_LAKE / "gwas_catalog.pkl")],
        "policy": {
            "positive_examples": "For each GWAS trait, answer is the strongest-scoring rsID among the sampled candidate list.",
            "candidate_rsids": "Distractors are lower-scoring variants from the same GWAS trait.",
            "answer_format": "Single FINAL ANSWER line containing only one rsID.",
            "eval_overlap_excluded": not args.allow_eval_overlap,
        },
        "counts": {"all": len(rows), "train": len(train_rows), "val": len(val_rows)},
        "source_stats": {
            "eligible_traits_after_filter": len(groups),
            "eval_traits": len(eval_traits),
            "eval_candidate_rsids": len(eval_candidate_rsids),
            "eval_answer_rsids": len(eval_answer_rsids),
        },
        "audit": audit(rows, eval_traits, eval_candidate_rsids, eval_answer_rsids),
        "validation_problem_count": len(problems),
        "validation_problem_examples": problems[:20],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        "# SFT D5 GWAS Variant V1\n\n"
        "Targeted SFT data for GWAS variant prioritization.\n\n"
        "- Input: GWAS phenotype plus candidate rsIDs\n"
        "- Output: `FINAL ANSWER: rs...`\n"
        "- Source: Biomni data lake `gwas_catalog.pkl`\n"
        "- Construction: within each trait, choose the strongest association among sampled candidate rsIDs\n"
        "- Default guard: excludes traits and rsIDs appearing in `data/biomni_eval1_dataset.parquet` GWAS variant examples\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "counts": summary["counts"], "audit": summary["audit"], "validation_problem_count": len(problems)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
