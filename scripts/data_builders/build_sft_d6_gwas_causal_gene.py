from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path("/root/autodl-tmp/Biomni-main")
DATA_LAKE = ROOT / "data/biomni_data/data_lake"
OUT = ROOT / "data/sft_d6_gwas_causal_gene_v1"
SEED = 20260428

SYSTEM_PROMPT = (
    "You are Biomni, a biomedical assistant. Answer concisely and follow the requested output format."
)


def norm_text(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_gene_list_from_prompt(prompt: str) -> set[str]:
    return {g.strip() for g in re.findall(r"\{([^{}]+)\}", prompt or "") if g.strip()}


def extract_eval_causal_gene_exclusions(paths: Iterable[Path]) -> tuple[set[str], set[str], set[str]]:
    traits = set()
    candidate_genes = set()
    answer_genes = set()
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "task_name" not in df.columns:
            continue
        sub = df[df["task_name"].eq("gwas_causal_gene_gwas_catalog")]
        for row in sub.itertuples(index=False):
            prompt = getattr(row, "prompt", "")
            match = re.search(r"GWAS phenotype:\s*(.+?)\nGenes in locus:", prompt, flags=re.DOTALL)
            if match:
                traits.add(norm_text(match.group(1)))
            candidate_genes.update(parse_gene_list_from_prompt(prompt))
            answer = str(getattr(row, "answer", "") or "").strip()
            if answer:
                answer_genes.add(answer)
    return traits, candidate_genes, answer_genes


def stable_key(obj: dict) -> str:
    messages = obj.get("messages") or []
    text = "\n".join(m.get("content", "") for m in messages)
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def parse_single_gene(value: object, valid_symbols: set[str]) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    if any(sep in text for sep in [" - ", ",", ";", "/", " x "]):
        return None
    text = text.strip()
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", text) and text in valid_symbols:
        return text
    upper = text.upper()
    if re.fullmatch(r"[A-Z][A-Z0-9_.-]*", upper) and upper in valid_symbols:
        return upper
    return None


def parse_chr(value: object) -> str | None:
    text = str(value or "").strip().replace("chr", "")
    if text in {str(i) for i in range(1, 23)} or text in {"X", "Y"}:
        return text
    return None


def parse_pos(value: object) -> int | None:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    first = re.split(r"[;,\sxX-]+", text)[0]
    try:
        pos = int(float(first))
    except Exception:
        return None
    return pos if pos > 0 else None


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
                return -math.log10(p)
        except Exception:
            return None
    return None


def read_gene_index(path: Path, excluded_candidate_genes: set[str]) -> tuple[dict[str, list[dict]], set[str]]:
    df = pd.read_parquet(
        path,
        columns=["chr", "gene_start", "gene_end", "gene_name", "gene_type", "transcript_is_canonical"],
    )
    df = df[df["gene_name"].notna()]
    df = df[df["gene_type"].eq("protein_coding")]
    df = df[df["transcript_is_canonical"].fillna(0).astype(float).eq(1.0)]
    by_chr: dict[str, list[dict]] = defaultdict(list)
    valid_symbols = set()
    for row in df.itertuples(index=False):
        symbol = str(row.gene_name).strip()
        chrom = parse_chr(row.chr)
        if not symbol or not chrom or symbol in excluded_candidate_genes:
            continue
        start = int(row.gene_start)
        end = int(row.gene_end)
        by_chr[chrom].append({"symbol": symbol, "start": start, "end": end, "mid": (start + end) // 2})
        valid_symbols.add(symbol)
    for chrom in by_chr:
        by_chr[chrom].sort(key=lambda g: g["mid"])
    return by_chr, valid_symbols


def locus_candidates(
    genes_by_chr: dict[str, list[dict]],
    chrom: str,
    pos: int,
    answer_gene: str,
    min_candidates: int,
    max_candidates: int,
) -> list[str] | None:
    genes = genes_by_chr.get(chrom) or []
    if not genes:
        return None
    for window in [100_000, 250_000, 500_000, 1_000_000, 2_000_000]:
        candidates = [g for g in genes if abs(g["mid"] - pos) <= window or (g["start"] <= pos <= g["end"])]
        symbols = []
        seen = set()
        for gene in sorted(candidates, key=lambda g: (abs(g["mid"] - pos), g["symbol"])):
            symbol = gene["symbol"]
            if symbol not in seen:
                symbols.append(symbol)
                seen.add(symbol)
        if answer_gene in seen and len(symbols) >= min_candidates:
            if len(symbols) > max_candidates:
                keep = symbols[: max_candidates - 1]
                if answer_gene not in keep:
                    keep.append(answer_gene)
                symbols = sorted(set(keep), key=lambda s: (abs(next(g["mid"] for g in genes if g["symbol"] == s) - pos), s))
            return symbols
    return None


def make_user_prompt(trait: str, candidates: list[str]) -> str:
    gene_list = ",".join(f"{{{gene}}}" for gene in candidates)
    return (
        "Your task is to identify likely causal genes within a locus for a given GWAS phenotype. "
        "From the list, provide only the likely causal gene (matching one of the given genes).\n"
        "Identify the causal gene.\n"
        f"GWAS phenotype: {trait}\n"
        f"Genes in locus: {gene_list}"
    )


def make_row(row: pd.Series, answer_gene: str, candidates: list[str], score: float) -> dict:
    trait = str(row.get("DISEASE/TRAIT") or "").strip()
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt(trait, candidates)},
            {"role": "assistant", "content": f"FINAL ANSWER: {answer_gene}"},
        ],
        "dataset": "sft_d6_gwas_causal_gene_v1",
        "task_type": "D6-gwas-causal-gene",
        "source_dataset": "biomni_data_lake/gwas_catalog.pkl+gene_info.parquet",
        "source_policy": "Answer is single MAPPED_GENE from GWAS Catalog; candidate locus genes are canonical protein-coding genes near the association position.",
        "gwas_trait": trait,
        "answer_gene": answer_gene,
        "answer_score": score,
        "chromosome": parse_chr(row.get("CHR_ID")),
        "position": parse_pos(row.get("CHR_POS")),
        "candidate_count": len(candidates),
        "mapped_gene": str(row.get("MAPPED_GENE") or ""),
        "reported_genes": str(row.get("REPORTED GENE(S)") or ""),
        "snps": str(row.get("SNPS") or ""),
    }


def build_rows(
    gwas_path: Path,
    genes_by_chr: dict[str, list[dict]],
    valid_symbols: set[str],
    excluded_traits: set[str],
    excluded_answer_genes: set[str],
    max_samples: int,
    seed: int,
    min_candidates: int,
    max_candidates: int,
    max_per_answer: int,
    max_per_trait: int,
) -> list[dict]:
    rng = random.Random(seed)
    df = pd.read_pickle(gwas_path)
    order = list(df.index)
    rng.shuffle(order)
    rows = []
    seen = set()
    seen_prompts = set()
    answer_counts = Counter()
    trait_counts = Counter()
    for idx in order:
        row = df.loc[idx]
        trait = str(row.get("DISEASE/TRAIT") or "").strip()
        trait_norm = norm_text(trait)
        if not trait or trait_norm in excluded_traits or trait_counts[trait_norm] >= max_per_trait:
            continue
        answer_gene = parse_single_gene(row.get("MAPPED_GENE"), valid_symbols)
        if not answer_gene or answer_gene in excluded_answer_genes or answer_counts[answer_gene] >= max_per_answer:
            continue
        chrom = parse_chr(row.get("CHR_ID"))
        pos = parse_pos(row.get("CHR_POS"))
        score = score_row(row)
        if not chrom or not pos or score is None:
            continue
        candidates = locus_candidates(genes_by_chr, chrom, pos, answer_gene, min_candidates, max_candidates)
        if not candidates or answer_gene not in candidates:
            continue
        rng.shuffle(candidates)
        out = make_row(row, answer_gene, candidates, score)
        key = stable_key(out)
        user_prompt = out["messages"][1]["content"]
        if key in seen or user_prompt in seen_prompts:
            continue
        seen.add(key)
        seen_prompts.add(user_prompt)
        rows.append(out)
        answer_counts[answer_gene] += 1
        trait_counts[trait_norm] += 1
        if len(rows) >= max_samples:
            return rows
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def audit(rows: list[dict], eval_traits: set[str], eval_candidate_genes: set[str], eval_answer_genes: set[str]) -> dict:
    answers = Counter()
    traits = Counter()
    candidate_sizes = Counter()
    all_candidate_genes = set()
    phrase_counts = Counter()
    for row in rows:
        text = json.dumps(row["messages"], ensure_ascii=False)
        for phrase in ("Key evidence", "same-task distractors", "supported answer"):
            phrase_counts[phrase] += text.count(phrase)
        answers[row["answer_gene"]] += 1
        traits[norm_text(row["gwas_trait"])] += 1
        candidate_sizes[row["candidate_count"]] += 1
        all_candidate_genes.update(parse_gene_list_from_prompt(row["messages"][1]["content"]))
    return {
        "rows": len(rows),
        "unique_traits": len(traits),
        "unique_answer_genes": len(answers),
        "unique_candidate_genes": len(all_candidate_genes),
        "top_answer_genes": dict(answers.most_common(20)),
        "top_traits": dict(traits.most_common(20)),
        "candidate_count_distribution": dict(sorted(candidate_sizes.items())),
        "template_phrase_counts": dict(phrase_counts),
        "eval_trait_overlap_count": len(set(traits) & eval_traits),
        "eval_trait_overlap": sorted(set(traits) & eval_traits)[:50],
        "eval_candidate_gene_overlap_count": len(all_candidate_genes & eval_candidate_genes),
        "eval_candidate_gene_overlap": sorted(all_candidate_genes & eval_candidate_genes)[:50],
        "eval_answer_gene_overlap_count": len(set(answers) & eval_answer_genes),
        "eval_answer_gene_overlap": sorted(set(answers) & eval_answer_genes)[:50],
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
        candidates = parse_gene_list_from_prompt(user)
        answer = row.get("answer_gene")
        if assistant != f"FINAL ANSWER: {answer}":
            problems.append((i, "assistant_format"))
        if answer not in candidates:
            problems.append((i, "answer_not_in_candidates"))
        if len(candidates) < 4:
            problems.append((i, "candidate_count"))
        if len(candidates) != row.get("candidate_count"):
            problems.append((i, "candidate_count_mismatch"))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-samples", type=int, default=30000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-candidates", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--max-per-answer", type=int, default=20)
    parser.add_argument("--max-per-trait", type=int, default=4)
    parser.add_argument("--allow-eval-overlap", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    eval_traits, eval_candidate_genes, eval_answer_genes = extract_eval_causal_gene_exclusions(
        [ROOT / "data/biomni_eval1_dataset.parquet"]
    )
    excluded_traits = set() if args.allow_eval_overlap else eval_traits
    excluded_candidate_genes = set() if args.allow_eval_overlap else eval_candidate_genes
    excluded_answer_genes = set() if args.allow_eval_overlap else eval_answer_genes

    genes_by_chr, valid_symbols = read_gene_index(DATA_LAKE / "gene_info.parquet", excluded_candidate_genes)
    rows = build_rows(
        gwas_path=DATA_LAKE / "gwas_catalog.pkl",
        genes_by_chr=genes_by_chr,
        valid_symbols=valid_symbols,
        excluded_traits=excluded_traits,
        excluded_answer_genes=excluded_answer_genes,
        max_samples=args.max_samples,
        seed=args.seed,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
        max_per_answer=args.max_per_answer,
        max_per_trait=args.max_per_trait,
    )
    random.Random(args.seed).shuffle(rows)
    val_n = max(1, int(len(rows) * args.val_ratio)) if rows else 0
    val_rows = rows[:val_n]
    train_rows = rows[val_n:]

    write_jsonl(args.out / "d6_gwas_causal_gene_all.jsonl", rows)
    write_jsonl(args.out / "d6_gwas_causal_gene_train.jsonl", train_rows)
    write_jsonl(args.out / "d6_gwas_causal_gene_val.jsonl", val_rows)

    problems = validate_rows(rows)
    summary = {
        "dataset": "sft_d6_gwas_causal_gene_v1",
        "seed": args.seed,
        "max_samples": args.max_samples,
        "files": {
            "all": str(args.out / "d6_gwas_causal_gene_all.jsonl"),
            "train": str(args.out / "d6_gwas_causal_gene_train.jsonl"),
            "val": str(args.out / "d6_gwas_causal_gene_val.jsonl"),
        },
        "source_files": [str(DATA_LAKE / "gwas_catalog.pkl"), str(DATA_LAKE / "gene_info.parquet")],
        "policy": {
            "positive_examples": "Single-gene GWAS Catalog MAPPED_GENE records with valid chromosome and position.",
            "candidate_genes": "Canonical protein-coding genes near the GWAS position, filtered to remove eval candidate genes by default.",
            "answer_format": "Single FINAL ANSWER line containing only one gene symbol.",
            "eval_overlap_excluded": not args.allow_eval_overlap,
            "max_per_answer": args.max_per_answer,
            "max_per_trait": args.max_per_trait,
        },
        "counts": {"all": len(rows), "train": len(train_rows), "val": len(val_rows)},
        "source_stats": {
            "valid_gene_symbols_after_filter": len(valid_symbols),
            "eval_traits": len(eval_traits),
            "eval_candidate_genes": len(eval_candidate_genes),
            "eval_answer_genes": len(eval_answer_genes),
        },
        "audit": audit(rows, eval_traits, eval_candidate_genes, eval_answer_genes),
        "validation_problem_count": len(problems),
        "validation_problem_examples": problems[:20],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        "# SFT D6 GWAS Causal Gene V1\n\n"
        "Targeted SFT data for GWAS causal-gene selection.\n\n"
        "- Input: GWAS phenotype plus locus gene list\n"
        "- Output: `FINAL ANSWER: GENE`\n"
        "- Sources: Biomni data lake `gwas_catalog.pkl` and `gene_info.parquet`\n"
        "- Construction: answer is a single GWAS Catalog `MAPPED_GENE`; candidates are nearby canonical protein-coding genes\n"
        "- Default guard: excludes traits, candidate genes, and answer genes from `data/biomni_eval1_dataset.parquet` GWAS Catalog examples\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "counts": summary["counts"], "audit": summary["audit"], "validation_problem_count": len(problems)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
