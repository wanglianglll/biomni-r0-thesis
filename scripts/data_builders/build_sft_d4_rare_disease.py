from __future__ import annotations

import argparse
import ast
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
OUT = ROOT / "data/sft_d4_rare_disease_v1"
SEED = 20260428

SYSTEM_PROMPT = (
    "You are Biomni, a biomedical assistant. Answer concisely and follow the requested output format."
)


def norm_name(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_omim_phenotypes(value: object) -> list[tuple[str, str]]:
    if pd.isna(value) or not str(value).strip():
        return []
    out = []
    for part in str(value).split(";"):
        part = part.strip()
        match = re.search(r"(.+?),\s*(\d{6})\s*\(", part)
        if not match:
            continue
        raw_name = match.group(1).strip()
        # OMIM braces usually denote susceptibility and question marks denote
        # provisional entries. They are poor supervision for diagnosis-style D4.
        if raw_name.startswith(("{", "[", "?")):
            continue
        name = raw_name.strip("{}[]")
        if re.search(r"\bsusceptibility\b|\bqtl\b|quantitative trait", name, flags=re.IGNORECASE):
            continue
        if name:
            out.append((name, match.group(2)))
    return out


def split_gene_ids(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    pieces = re.split(r"[,;|\s]+", str(value).strip())
    return [p for p in pieces if re.fullmatch(r"ENSG\d{11}", p)]


def format_hpo_id(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    if text.startswith("HP:"):
        return text
    if re.fullmatch(r"\d+", text):
        return f"HP:{int(text):07d}"
    return None


def read_hpo_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("id: HP:"):
            ids.add(line.split("id:", 1)[1].strip())
    return ids


def read_gene_symbol_to_ensembl(path: Path) -> dict[str, str]:
    df = pd.read_parquet(path, columns=["gene_id", "gene_name", "transcript_is_canonical", "gene_type"])
    df = df[df["gene_name"].notna()]
    if "gene_type" in df.columns:
        protein = df[df["gene_type"].eq("protein_coding")]
        if len(protein):
            df = protein
    if "transcript_is_canonical" in df.columns:
        canonical = df[df["transcript_is_canonical"].fillna(0).astype(float).eq(1.0)]
        if len(canonical):
            df = canonical
    mapping = {}
    for row in df.itertuples(index=False):
        symbol = str(row.gene_name).upper()
        gene_id = str(row.gene_id)
        if re.fullmatch(r"ENSG\d{11}", gene_id):
            mapping.setdefault(symbol, gene_id)
    return mapping


def read_omim_entries(path: Path, symbol_to_ensembl: dict[str, str]) -> dict[str, list[dict]]:
    df = pd.read_parquet(path)
    by_norm = defaultdict(list)
    for _, data in df.iterrows():
        symbol = str(data.get("Approved Gene Symbol") or "").strip().upper()
        ensembl_ids = split_gene_ids(data.get("Ensembl Gene ID"))
        if not ensembl_ids and symbol:
            mapped = symbol_to_ensembl.get(symbol)
            if mapped:
                ensembl_ids = [mapped]
        for disease_name, omim_id in parse_omim_phenotypes(data.get("Phenotypes")):
            by_norm[norm_name(disease_name)].append(
                {
                    "disease_name": disease_name,
                    "omim_id": omim_id,
                    "gene_symbol": symbol,
                    "ensembl_ids": ensembl_ids,
                    "omim_mim_number": str(int(data["MIM Number"])) if pd.notna(data.get("MIM Number")) else "",
                }
            )
    return by_norm


def disease_side(row: object) -> tuple[str, str, str] | None:
    if getattr(row, "x_type") == "disease":
        return str(getattr(row, "x_id")), str(getattr(row, "x_name")), str(getattr(row, "x_source"))
    if getattr(row, "y_type") == "disease":
        return str(getattr(row, "y_id")), str(getattr(row, "y_name")), str(getattr(row, "y_source"))
    return None


def phenotype_side(row: object) -> str | None:
    if getattr(row, "x_type") == "effect/phenotype" and getattr(row, "x_source") == "HPO":
        return format_hpo_id(getattr(row, "x_id"))
    if getattr(row, "y_type") == "effect/phenotype" and getattr(row, "y_source") == "HPO":
        return format_hpo_id(getattr(row, "y_id"))
    return None


def gene_side(row: object) -> str | None:
    if getattr(row, "x_type") == "gene/protein":
        return str(getattr(row, "x_name") or "").upper()
    if getattr(row, "y_type") == "gene/protein":
        return str(getattr(row, "y_name") or "").upper()
    return None


def read_kg_disease_maps(path: Path, valid_hpo_ids: set[str]) -> dict[str, dict]:
    columns = ["relation", "x_id", "x_type", "x_name", "x_source", "y_id", "y_type", "y_name", "y_source"]
    diseases: dict[str, dict] = {}
    for chunk in pd.read_csv(path, usecols=columns, chunksize=300000, low_memory=False):
        sub = chunk[chunk["relation"].isin(["disease_phenotype_positive", "disease_protein"])]
        for row in sub.itertuples(index=False):
            disease = disease_side(row)
            if disease is None:
                continue
            disease_id, disease_name, disease_source = disease
            record = diseases.setdefault(
                disease_id,
                {
                    "disease_id": disease_id,
                    "disease_name": disease_name,
                    "disease_source": disease_source,
                    "hpo_ids": set(),
                    "gene_symbols": set(),
                },
            )
            if getattr(row, "relation") == "disease_phenotype_positive":
                hpo_id = phenotype_side(row)
                if hpo_id and (not valid_hpo_ids or hpo_id in valid_hpo_ids):
                    record["hpo_ids"].add(hpo_id)
            elif getattr(row, "relation") == "disease_protein":
                symbol = gene_side(row)
                if symbol:
                    record["gene_symbols"].add(symbol)
    return diseases


def read_eval_rare_omim_ids(paths: Iterable[Path]) -> set[str]:
    ids = set()
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "task_name" not in df.columns:
            continue
        sub = df[df["task_name"].eq("rare_disease_diagnosis")]
        for answer in sub.get("answer", []):
            try:
                obj = json.loads(answer)
            except Exception:
                continue
            omim_id = str(obj.get("OMIM_ID") or "").strip()
            if re.fullmatch(r"\d{6}", omim_id):
                ids.add(omim_id)
    return ids


def stable_key(obj: dict) -> str:
    messages = obj.get("messages") or []
    text = "\n".join(m.get("content", "") for m in messages)
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def make_user_prompt(hpo_ids: list[str], candidate_genes: list[str]) -> str:
    return (
        "Task: given a patient's phenotypes and a list of candidate genes, diagnose the rare disease that the patient has.\n"
        f"Phenotypes: {', '.join(hpo_ids)}\n"
        f"Candidate genes: {candidate_genes}\n\n"
        "Output format: {'disease_name': XXX, 'OMIM_ID': XXX}"
    )


def make_row(
    disease: dict,
    omim_entry: dict,
    hpo_ids: list[str],
    candidate_genes: list[str],
    positive_gene: str,
    variant_index: int,
) -> dict:
    answer = {"disease_name": omim_entry["disease_name"], "OMIM_ID": omim_entry["omim_id"]}
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt(hpo_ids, candidate_genes)},
            {"role": "assistant", "content": "FINAL ANSWER: " + json.dumps(answer, ensure_ascii=False)},
        ],
        "dataset": "sft_d4_rare_disease_v1",
        "task_type": "D4-rare-disease",
        "source_dataset": "biomni_data_lake/kg.csv+omim.parquet+gene_info.parquet+hp.obo",
        "source_policy": "exact disease-name match between KG disease node and OMIM phenotype; HPO terms from KG disease_phenotype_positive edges",
        "disease_id": disease["disease_id"],
        "kg_disease_name": disease["disease_name"],
        "kg_disease_source": disease["disease_source"],
        "omim_disease_name": omim_entry["disease_name"],
        "omim_id": omim_entry["omim_id"],
        "positive_gene": positive_gene,
        "candidate_gene_count": len(candidate_genes),
        "hpo_count": len(hpo_ids),
        "variant_index": variant_index,
    }


def build_rows(
    diseases: dict[str, dict],
    omim_by_norm: dict[str, list[dict]],
    symbol_to_ensembl: dict[str, str],
    max_samples: int,
    seed: int,
    exclude_omim_ids: set[str],
    variants_per_pair: int,
) -> list[dict]:
    rng = random.Random(seed)
    all_ensembl = sorted(set(symbol_to_ensembl.values()))
    rows = []
    seen = set()

    candidate_pairs = []
    for disease in diseases.values():
        hpo_ids = sorted(disease["hpo_ids"])
        if len(hpo_ids) < 3:
            continue
        entries = omim_by_norm.get(norm_name(disease["disease_name"]), [])
        if not entries:
            continue
        for entry in entries:
            if entry["omim_id"] in exclude_omim_ids:
                continue
            positive_genes = list(entry["ensembl_ids"])
            if not positive_genes and entry["gene_symbol"]:
                positive_genes = [symbol_to_ensembl.get(entry["gene_symbol"], "")]
            for gene_id in positive_genes:
                if re.fullmatch(r"ENSG\d{11}", gene_id):
                    candidate_pairs.append((disease, entry, gene_id))

    rng.shuffle(candidate_pairs)
    for disease, entry, positive_gene in candidate_pairs:
        hpo_pool = sorted(disease["hpo_ids"])
        for variant_index in range(variants_per_pair):
            hpo_n = min(len(hpo_pool), rng.randint(3, 8))
            hpo_ids = sorted(rng.sample(hpo_pool, hpo_n))
            if variant_index % 3 == 0:
                candidate_genes = [positive_gene]
            else:
                distractor_n = rng.randint(2, 6)
                distractors = []
                while len(distractors) < distractor_n:
                    gene = rng.choice(all_ensembl)
                    if gene != positive_gene and gene not in distractors:
                        distractors.append(gene)
                candidate_genes = distractors + [positive_gene]
                rng.shuffle(candidate_genes)
            row = make_row(disease, entry, hpo_ids, candidate_genes, positive_gene, variant_index)
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


def audit(rows: list[dict], eval_all_omim_ids: set[str], eval_test_omim_ids: set[str]) -> dict:
    answers = Counter()
    disease_sources = Counter()
    candidate_sizes = Counter()
    hpo_sizes = Counter()
    omim_ids = Counter()
    bad_template_phrases = ("Key evidence", "same-task distractors", "supported answer")
    phrase_counts = Counter()
    for row in rows:
        text = json.dumps(row["messages"], ensure_ascii=False)
        for phrase in bad_template_phrases:
            phrase_counts[phrase] += text.count(phrase)
        answer = row["messages"][2]["content"].replace("FINAL ANSWER: ", "", 1)
        answers[answer] += 1
        disease_sources[row.get("kg_disease_source", "unknown")] += 1
        candidate_sizes[row.get("candidate_gene_count")] += 1
        hpo_sizes[row.get("hpo_count")] += 1
        omim_ids[row.get("omim_id")] += 1
    return {
        "rows": len(rows),
        "unique_final_answers": len(answers),
        "unique_omim_ids": len(omim_ids),
        "top_final_answers": dict(answers.most_common(20)),
        "disease_source_counts": dict(disease_sources.most_common()),
        "candidate_gene_count_distribution": dict(sorted(candidate_sizes.items())),
        "hpo_count_distribution": dict(sorted(hpo_sizes.items())),
        "template_phrase_counts": dict(phrase_counts),
        "eval_all_omim_overlap_count": len(set(omim_ids) & eval_all_omim_ids),
        "eval_all_omim_overlap_ids": sorted(set(omim_ids) & eval_all_omim_ids),
        "eval_test_omim_overlap_count": len(set(omim_ids) & eval_test_omim_ids),
        "eval_test_omim_overlap_ids": sorted(set(omim_ids) & eval_test_omim_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--variants-per-pair", type=int, default=6)
    parser.add_argument("--allow-eval-omim", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    valid_hpo_ids = read_hpo_ids(DATA_LAKE / "hp.obo")
    symbol_to_ensembl = read_gene_symbol_to_ensembl(DATA_LAKE / "gene_info.parquet")
    omim_by_norm = read_omim_entries(DATA_LAKE / "omim.parquet", symbol_to_ensembl)
    diseases = read_kg_disease_maps(DATA_LAKE / "kg.csv", valid_hpo_ids)
    eval_all_omim_ids = read_eval_rare_omim_ids([ROOT / "data/biomni_eval1_dataset.parquet"])
    eval_test_omim_ids = read_eval_rare_omim_ids([ROOT / "data/sft/eval_test_split.parquet"])
    exclude_omim_ids = set() if args.allow_eval_omim else eval_all_omim_ids

    rows = build_rows(
        diseases=diseases,
        omim_by_norm=omim_by_norm,
        symbol_to_ensembl=symbol_to_ensembl,
        max_samples=args.max_samples,
        seed=args.seed,
        exclude_omim_ids=exclude_omim_ids,
        variants_per_pair=args.variants_per_pair,
    )
    random.Random(args.seed).shuffle(rows)
    val_n = max(1, int(len(rows) * args.val_ratio)) if rows else 0
    val_rows = rows[:val_n]
    train_rows = rows[val_n:]

    write_jsonl(args.out / "d4_rare_disease_all.jsonl", rows)
    write_jsonl(args.out / "d4_rare_disease_train.jsonl", train_rows)
    write_jsonl(args.out / "d4_rare_disease_val.jsonl", val_rows)

    summary = {
        "dataset": "sft_d4_rare_disease_v1",
        "seed": args.seed,
        "max_samples": args.max_samples,
        "variants_per_pair": args.variants_per_pair,
        "files": {
            "all": str(args.out / "d4_rare_disease_all.jsonl"),
            "train": str(args.out / "d4_rare_disease_train.jsonl"),
            "val": str(args.out / "d4_rare_disease_val.jsonl"),
        },
        "source_files": [
            str(DATA_LAKE / "kg.csv"),
            str(DATA_LAKE / "omim.parquet"),
            str(DATA_LAKE / "gene_info.parquet"),
            str(DATA_LAKE / "hp.obo"),
        ],
        "policy": {
            "positive_examples": "KG disease nodes with HPO phenotype edges, exact matched to OMIM phenotype names with OMIM gene/Ensembl evidence.",
            "candidate_genes": "Positive Ensembl gene from OMIM alone for one third of variants; otherwise mixed with random Ensembl distractors.",
            "answer_format": "Single FINAL ANSWER line containing JSON with disease_name and OMIM_ID.",
            "eval_omim_excluded": not args.allow_eval_omim,
        },
        "counts": {"all": len(rows), "train": len(train_rows), "val": len(val_rows)},
        "audit": audit(rows, eval_all_omim_ids, eval_test_omim_ids),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        "# SFT D4 Rare Disease V1\n\n"
        "Targeted SFT data for rare-disease diagnosis.\n\n"
        "- Input: HPO phenotype IDs plus candidate Ensembl gene IDs\n"
        "- Output: `FINAL ANSWER: {\"disease_name\": ..., \"OMIM_ID\": ...}`\n"
        "- Sources: Biomni data lake `kg.csv`, `omim.parquet`, `gene_info.parquet`, `hp.obo`\n"
        "- Construction: exact KG disease-name to OMIM phenotype-name match, with KG HPO phenotype edges and disease-gene edges\n"
        "- Default guard: OMIM IDs appearing in `data/biomni_eval1_dataset.parquet` rare-disease examples are excluded\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "counts": summary["counts"], "audit": summary["audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
