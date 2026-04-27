# Unified BiomniEval1 Results

Generated on 2026-04-27 from two AutoDL instances.

## Overall

| Run | Model | Variant | Accuracy | Correct/Total | Timestamp |
|---|---|---:|---:|---:|---|
| `qwen3.5_base_fixed` | Qwen3.5-9B | Base | 0.3279 | 142/433 | 20260426_205934 |
| `llama3.1_base_fixed` | Meta-Llama-3.1-8B-Instruct | Base | 0.3025 | 131/433 | 20260426_210856 |
| `qwen2.5_sft` | /root/autodl-tmp/Biomni-main/scripts/output/qwen2.5_sft | SFT | 0.2702 | 117/433 | 20260318_212358 |
| `mistral_base` | /root/autodl-tmp/Biomni-main/models/Mistral-7B-Instruct-v0.3 | Base | 0.2217 | 96/433 | 20260318_202828 |
| `qwen2.5_base` | /root/autodl-tmp/Biomni-main/models/Qwen2.5-7B-Instruct/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 | Base | 0.1732 | 75/433 | 20260318_141734 |
| `mistral_sft` | /root/autodl-tmp/Biomni-main/scripts/output/mistral_sft | SFT | 0.1178 | 51/433 | 20260318_222034 |

## Task Accuracy

| Run | crispr_delivery | gwas_causal_gene_gwas_catalog | gwas_causal_gene_opentargets | gwas_causal_gene_pharmaprojects | gwas_variant_prioritization | lab_bench_dbqa | lab_bench_seqqa | patient_gene_detection | rare_disease_diagnosis | screen_gene_retrieval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mistral_base` | 0.0000 | 0.2000 | 0.6200 | 0.4800 | 0.0930 | 0.1800 | 0.1200 | 0.0000 | 0.0000 | 0.2400 |
| `qwen2.5_base` | 0.0000 | 0.1400 | 0.5600 | 0.4000 | 0.0465 | 0.0600 | 0.0400 | 0.0000 | 0.0000 | 0.2600 |
| `mistral_sft` | 0.2000 | 0.0200 | 0.2000 | 0.1600 | 0.4884 | 0.1800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `qwen2.5_sft` | 0.3000 | 0.2800 | 0.7600 | 0.6000 | 0.3023 | 0.0200 | 0.0600 | 0.0200 | 0.0000 | 0.2800 |
| `qwen3.5_base_fixed` | 0.1000 | 0.4800 | 0.6600 | 0.5400 | 0.1628 | 0.2400 | 0.3000 | 0.0600 | 0.0000 | 0.4000 |
| `llama3.1_base_fixed` | 0.1000 | 0.2000 | 0.6800 | 0.5000 | 0.1860 | 0.3200 | 0.3000 | 0.1200 | 0.0000 | 0.3200 |

## Notes

- `qwen3.5_base_fixed` uses the corrected Qwen3.5 chat-template and task-specific extraction pipeline.
- `llama3.1_base_fixed` uses the same corrected extraction pipeline on the second instance.
- Raw summary/detail JSON files are archived under `raw/`.
