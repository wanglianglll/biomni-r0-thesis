#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_LAKE = PROJECT_ROOT / 'data' / 'biomni_data' / 'data_lake'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'datalake_sft'
RANDOM_SEED = 42
TRAIN_RATIO = 0.98

SYSTEM_PROMPT = (
    'You are Biomni, a biomedical reasoning assistant. '
    'Use the provided structured evidence to answer carefully and concisely. '
    'When possible, explain the key evidence first and then give a final answer.'
)


def clean_text(value) -> str:
    if value is None:
        return ''
    if isinstance(value, float) and pd.isna(value):
        return ''
    text = str(value)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def make_sample(task_type: str, source: str, user: str, assistant: str, metadata: dict) -> dict:
    return {
        'task_type': task_type,
        'source': source,
        'metadata': metadata,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': assistant},
        ],
    }


def build_gwas_samples(limit: int = 8000):
    df = pd.read_pickle(DATA_LAKE / 'gwas_catalog.pkl')
    df = df[['DISEASE/TRAIT', 'SNPS', 'MAPPED_GENE', 'CHR_ID', 'CHR_POS', 'CONTEXT', 'P-VALUE', 'OR or BETA', 'REPORTED GENE(S)']].copy()
    df = df.dropna(subset=['DISEASE/TRAIT', 'SNPS', 'MAPPED_GENE'])
    df['DISEASE/TRAIT'] = df['DISEASE/TRAIT'].map(clean_text)
    df['SNPS'] = df['SNPS'].map(clean_text)
    df['MAPPED_GENE'] = df['MAPPED_GENE'].map(clean_text)
    df = df[df['MAPPED_GENE'].str.match(r'^[A-Za-z0-9._-]+$')]
    df = df[df['SNPS'].str.match(r'^rs\d+$')]
    df = df[df['DISEASE/TRAIT'].str.len() > 3]
    df = df.drop_duplicates(subset=['DISEASE/TRAIT', 'SNPS', 'MAPPED_GENE'])
    if len(df) > limit:
        df = df.sample(limit, random_state=RANDOM_SEED)
    samples = []
    for row in df.itertuples(index=False):
        trait = clean_text(row[0])
        snp = clean_text(row[1])
        gene = clean_text(row[2])
        chrom = clean_text(row[3])
        pos = clean_text(row[4])
        context = clean_text(row[5])
        pval = clean_text(row[6])
        beta = clean_text(row[7])
        reported = clean_text(row[8])
        user = (
            f'For the GWAS trait "{trait}", analyze the strongest associated variant and infer the most likely mapped gene.\n'
            f'Evidence:\n- SNP: {snp}\n- Chromosome: {chrom}\n- Position: {pos}\n- Variant context: {context}\n'
            f'- Reported gene(s): {reported or "N/A"}\n- P-value: {pval}\n- Effect size: {beta or "N/A"}\n\n'
            'Return the most likely mapped gene and a short rationale.'
        )
        assistant = (
            f'Reasoning:\n- The trait is {trait}.\n- The lead variant is {snp} on chromosome {chrom} at position {pos}.\n'
            f'- The catalog maps this association to {gene}.\n- The variant context is {context or "not specified"}, '
            f'and the reported gene field is {reported or gene}.\n\nFINAL ANSWER: {gene}'
        )
        samples.append(make_sample('gwas_trait_to_gene', 'gwas_catalog', user, assistant, {'trait': trait, 'snp': snp, 'answer': gene}))
    return samples


def build_disgenet_samples(limit_pairs: int = 6000, limit_diseases: int = 4000):
    df = pd.read_parquet(DATA_LAKE / 'DisGeNET.parquet').dropna().copy()
    samples = []
    disease_rows = []
    gene_pairs = []
    for row in df.itertuples(index=False):
        disorder = clean_text(row[0])
        genes_raw = row[1]
        try:
            genes = [clean_text(g) for g in ast.literal_eval(genes_raw)]
        except Exception:
            continue
        genes = [g for g in genes if g and re.match(r'^[A-Za-z0-9._-]+$', g)]
        genes = list(dict.fromkeys(genes))
        if not disorder or not genes:
            continue
        disease_rows.append((disorder, genes))
        for gene in genes[:8]:
            gene_pairs.append((gene, disorder, genes))
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(disease_rows)
    rng.shuffle(gene_pairs)
    for disorder, genes in disease_rows[:limit_diseases]:
        answer = ', '.join(genes[:8])
        user = f'List candidate genes associated with the disorder "{disorder}" based on the available biomedical knowledge base.'
        assistant = (
            f'Reasoning:\n- The disorder is {disorder}.\n- The knowledge base links this disorder to the following candidate genes: {answer}.\n'
            '- These genes can be used as a starting point for downstream validation.\n\n'
            f'FINAL ANSWER: {answer}'
        )
        samples.append(make_sample('disease_to_genes', 'DisGeNET', user, assistant, {'disorder': disorder, 'answer_genes': genes[:8]}))
    for gene, disorder, genes in gene_pairs[:limit_pairs]:
        user = f'A biomedical knowledge base links gene {gene} to at least one disorder. Name one associated disorder and mention the broader candidate gene set for that disorder.'
        assistant = (
            f'Reasoning:\n- Gene {gene} appears in the candidate set for {disorder}.\n'
            f'- Other genes linked to the same disorder include: {", ".join(genes[:8])}.\n\n'
            f'FINAL ANSWER: {disorder}'
        )
        samples.append(make_sample('gene_to_disorder', 'DisGeNET', user, assistant, {'gene': gene, 'answer_disorder': disorder}))
    return samples


def build_omim_samples(limit: int = 5000):
    df = pd.read_parquet(DATA_LAKE / 'omim.parquet').copy()
    keep_cols = ['MIM Number', 'Approved Gene Symbol', 'Gene Name', 'Phenotypes', 'Chromosome', 'Cyto Location']
    df = df[keep_cols].dropna(subset=['MIM Number'])
    df['Approved Gene Symbol'] = df['Approved Gene Symbol'].map(clean_text)
    df['Gene Name'] = df['Gene Name'].map(clean_text)
    df['Phenotypes'] = df['Phenotypes'].map(clean_text)
    df['Chromosome'] = df['Chromosome'].map(clean_text)
    df['Cyto Location'] = df['Cyto Location'].map(clean_text)
    df = df[(df['Approved Gene Symbol'] != '') | (df['Gene Name'] != '')]
    df = df.drop_duplicates(subset=['MIM Number', 'Approved Gene Symbol', 'Gene Name'])
    if len(df) > limit:
        df = df.sample(limit, random_state=RANDOM_SEED)
    samples = []
    for row in df.itertuples(index=False):
        mim, symbol, gene_name, phenotypes, chrom, cyto = row
        entity = symbol or gene_name
        phenotype_text = phenotypes or 'Phenotype information is not explicitly listed.'
        user = (
            f'Provide a compact OMIM-style summary for the gene or locus "{entity}".\n'
            f'Known fields:\n- MIM number: {int(mim) if pd.notna(mim) else mim}\n- Chromosome: {chrom or "N/A"}\n- Cytogenetic location: {cyto or "N/A"}\n- Phenotypes: {phenotype_text}'
        )
        assistant = (
            f'Reasoning:\n- The focal gene/locus is {entity}.\n- Its OMIM identifier is {int(mim) if pd.notna(mim) else mim}.\n'
            f'- The locus is on {chrom or "unknown chromosome"} at {cyto or "unknown cytoband"}.\n'
            f'- Reported phenotype summary: {phenotype_text}.\n\n'
            f'FINAL ANSWER: MIM {int(mim) if pd.notna(mim) else mim}'
        )
        samples.append(make_sample('omim_gene_summary', 'omim', user, assistant, {'entity': entity, 'mim_number': int(mim) if pd.notna(mim) else None}))
    return samples


def build_gene_info_samples(limit: int = 8000):
    df = pd.read_parquet(DATA_LAKE / 'gene_info.parquet').copy()
    df = df[['gene_id', 'gene_name', 'chr', 'gene_start', 'gene_end', 'strand', 'gene_type', 'transcript_is_canonical']]
    df = df.dropna(subset=['gene_id', 'chr', 'gene_start', 'gene_end'])
    df['gene_name'] = df['gene_name'].map(clean_text)
    df = df[df['gene_name'] != '']
    df = df.sort_values(['gene_id', 'transcript_is_canonical'], ascending=[True, False]).drop_duplicates(subset=['gene_id'])
    if len(df) > limit:
        df = df.sample(limit, random_state=RANDOM_SEED)
    samples = []
    for row in df.itertuples(index=False):
        gene_id, gene_name, chrom, start, end, strand, gene_type, canonical = row
        strand_text = '+' if str(strand) == '1' else '-' if str(strand) == '-1' else str(strand)
        user = (
            f'Given the Ensembl gene record below, summarize the genomic location of {gene_name}.\n'
            f'Evidence:\n- Ensembl gene ID: {gene_id}\n- Chromosome: {chrom}\n- Start: {int(start)}\n- End: {int(end)}\n'
            f'- Strand: {strand_text}\n- Gene type: {gene_type}\n- Canonical transcript available: {bool(canonical)}'
        )
        assistant = (
            f'Reasoning:\n- {gene_name} corresponds to Ensembl gene ID {gene_id}.\n'
            f'- It is located on chromosome {chrom} from {int(start)} to {int(end)} on the {strand_text} strand.\n'
            f'- The gene type is {gene_type}.\n\n'
            f'FINAL ANSWER: {gene_name} is a {gene_type} gene on chr{chrom}:{int(start)}-{int(end)} ({strand_text} strand).'
        )
        samples.append(make_sample('gene_coordinate_summary', 'gene_info', user, assistant, {'gene_id': gene_id, 'gene_name': gene_name}))
    return samples


def build_proteinatlas_samples(limit: int = 5000):
    df = pd.read_csv(DATA_LAKE / 'proteinatlas.tsv', sep='\t', low_memory=False)
    cols = ['Gene', 'Ensembl', 'Gene description', 'Protein class', 'RNA tissue specificity', 'RNA tissue distribution', 'RNA single cell type specificity', 'Subcellular location', 'Disease involvement']
    df = df[cols].copy().dropna(subset=['Gene'])
    for col in cols:
        df[col] = df[col].map(clean_text)
    df = df[(df['RNA tissue specificity'] != '') | (df['Subcellular location'] != '') | (df['Disease involvement'] != '')]
    df = df.drop_duplicates(subset=['Gene'])
    if len(df) > limit:
        df = df.sample(limit, random_state=RANDOM_SEED)
    samples = []
    for row in df.itertuples(index=False):
        gene, ensembl, desc, protein_class, tissue_spec, tissue_dist, sc_spec, subcell, disease_inv = row
        user = (
            f'Summarize the Human Protein Atlas profile for gene {gene}.\n'
            f'Evidence:\n- Ensembl: {ensembl or "N/A"}\n- Description: {desc or "N/A"}\n- Protein class: {protein_class or "N/A"}\n'
            f'- RNA tissue specificity: {tissue_spec or "N/A"}\n- RNA tissue distribution: {tissue_dist or "N/A"}\n'
            f'- Single-cell specificity: {sc_spec or "N/A"}\n- Subcellular location: {subcell or "N/A"}\n- Disease involvement: {disease_inv or "N/A"}'
        )
        assistant = (
            f'Reasoning:\n- {gene} ({ensembl or "no Ensembl ID provided"}) is described as {desc or "a gene with limited annotation"}.\n'
            f'- Protein class: {protein_class or "not specified"}.\n'
            f'- RNA expression pattern: {tissue_spec or "unknown tissue specificity"}; distribution: {tissue_dist or "unknown"}.\n'
            f'- Single-cell specificity: {sc_spec or "not specified"}.\n'
            f'- Subcellular location: {subcell or "not specified"}.\n'
            f'- Disease involvement: {disease_inv or "not specified"}.\n\n'
            f'FINAL ANSWER: {gene} shows {tissue_spec or "unspecified"} expression with {tissue_dist or "unspecified distribution"}.'
        )
        samples.append(make_sample('protein_expression_summary', 'proteinatlas', user, assistant, {'gene': gene, 'ensembl': ensembl}))
    return samples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)

    all_samples = []
    builders = [
        build_gwas_samples,
        build_disgenet_samples,
        build_omim_samples,
        build_gene_info_samples,
        build_proteinatlas_samples,
    ]
    for builder in builders:
        samples = builder()
        print(f'{builder.__name__}: {len(samples)} samples')
        all_samples.extend(samples)

    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    def write_jsonl(path: Path, rows):
        with path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    write_jsonl(OUTPUT_DIR / 'datalake_all.jsonl', all_samples)
    write_jsonl(OUTPUT_DIR / 'datalake_train.jsonl', train_samples)
    write_jsonl(OUTPUT_DIR / 'datalake_val.jsonl', val_samples)

    summary = {
        'random_seed': RANDOM_SEED,
        'train_ratio': TRAIN_RATIO,
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'task_type_counts': dict(Counter(sample['task_type'] for sample in all_samples)),
        'source_counts': dict(Counter(sample['source'] for sample in all_samples)),
        'output_files': {
            'all': str(OUTPUT_DIR / 'datalake_all.jsonl'),
            'train': str(OUTPUT_DIR / 'datalake_train.jsonl'),
            'val': str(OUTPUT_DIR / 'datalake_val.jsonl'),
        },
    }
    (OUTPUT_DIR / 'README.md').write_text(
        '# Datalake SFT Dataset\n\n'
        'This directory contains instruction-style ChatML data generated from the Biomni datalake.\n\n'
        '- `datalake_all.jsonl`: all generated samples\n'
        '- `datalake_train.jsonl`: train split\n'
        '- `datalake_val.jsonl`: validation split\n'
        '- `summary.json`: dataset statistics\n\n'
        'The current version focuses on GWAS, disease-gene associations, OMIM summaries, gene coordinate summaries, and Human Protein Atlas expression summaries.\n',
        encoding='utf-8'
    )
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
