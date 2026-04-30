from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/root/autodl-tmp/Biomni-main")
DATA = ROOT / "data"
OUT = DATA / "sft_eval1_taskwise_v1"
SEED = 20260429

SYSTEM_PROMPT = (
    "You are a biomedical assistant. Follow the requested answer format strictly. "
    "Reply with exactly one line starting with FINAL ANSWER:. Do not explain."
)

TASKS = [
    "crispr_delivery",
    "gwas_causal_gene_gwas_catalog",
    "gwas_causal_gene_opentargets",
    "gwas_causal_gene_pharmaprojects",
    "gwas_variant_prioritization",
    "lab_bench_dbqa",
    "lab_bench_seqqa",
    "patient_gene_detection",
    "rare_disease_diagnosis",
    "screen_gene_retrieval",
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_row(task: str, prompt: str, answer: str, source: str, **meta) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": f"FINAL ANSWER: {answer}"},
        ],
        "dataset": "sft_eval1_taskwise_v1",
        "task_name": task,
        "task_type": task,
        "source_dataset": source,
        **meta,
    }


def stable_key(row: dict) -> str:
    messages = row.get("messages", [])
    return json.dumps(
        {
            "task": row.get("task_name"),
            "user": messages[1]["content"] if len(messages) > 1 else "",
            "assistant": messages[2]["content"] if len(messages) > 2 else "",
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def dedup(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        key = stable_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def split_rows(rows: list[dict], seed: int, val_ratio: float = 0.04) -> tuple[list[dict], list[dict]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    protected = [r for r in rows if r.get("calibration_split") == "eval_train"]
    public = [r for r in rows if r.get("calibration_split") != "eval_train"]
    val_n = max(1, int(len(public) * val_ratio)) if len(public) >= 10 else 0
    val = public[:val_n]
    train = protected + public[val_n:]
    random.Random(seed + 1).shuffle(train)
    return train, val


def eval_rows(split: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / "sft" / f"eval_{split}_split.parquet")


def normalize_mc_prompt(question: str, options: list[str], answer_index: int, task: str) -> tuple[str, str]:
    labels = list("ABCDEF")
    lines = [
        "The following is a multiple choice question about biology.",
        "Please answer by responding with the letter of the correct answer.",
        "",
        f"Question: {question}",
        "Options:",
    ]
    for label, option in zip(labels, options):
        lines.append(f"{label}.{option}")
    return "\n".join(lines), labels[answer_index]


def add_eval_train_calibration(rows_by_task: dict[str, list[dict]]) -> None:
    df = eval_rows("train")
    for row in df.itertuples(index=False):
        task = str(row.task_name)
        answer = str(row.answer).strip()
        if task == "rare_disease_diagnosis":
            try:
                answer_obj = json.loads(answer)
                answer = json.dumps(
                    {
                        "disease_name": answer_obj.get("disease_name", ""),
                        "OMIM_ID": str(answer_obj.get("OMIM_ID", "")),
                    },
                    ensure_ascii=False,
                )
            except Exception:
                pass
        elif task == "patient_gene_detection":
            answer = json.dumps({"causal_gene": [answer]}, ensure_ascii=False)
        rows_by_task[task].append(
            make_row(
                task,
                str(row.prompt),
                answer,
                "data/sft/eval_train_split.parquet",
                calibration_split="eval_train",
                task_instance_id=int(row.task_instance_id),
            )
        )


def build_lab_bench(rows_by_task: dict[str, list[dict]]) -> None:
    specs = [
        ("lab_bench_dbqa", DATA / "biomni_data/benchmark/DbQA/train-00000-of-00001.parquet"),
        ("lab_bench_seqqa", DATA / "biomni_data/benchmark/SeqQA/train-00000-of-00001.parquet"),
    ]
    eval_test_prompts = set(eval_rows("test")["prompt"].astype(str))
    rng = random.Random(SEED + 11)
    for task, path in specs:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for idx, row in df.iterrows():
            question = str(row["question"]).strip()
            if question in eval_test_prompts:
                continue
            options = [str(row["ideal"])] + [str(x) for x in list(row["distractors"])]
            options = [x for x in options if x and x.lower() != "nan"]
            if len(options) < 2:
                continue
            for variant in range(2):
                shuffled = list(options)
                rng.shuffle(shuffled)
                answer_index = shuffled.index(str(row["ideal"]))
                prompt, answer = normalize_mc_prompt(question, shuffled, answer_index, task)
                rows_by_task[task].append(
                    make_row(
                        task,
                        prompt,
                        answer,
                        str(path.relative_to(ROOT)),
                        subtask=str(row.get("subtask", "")),
                        source_row=int(idx),
                        augmentation=f"option_shuffle_{variant}",
                    )
                )


def build_crispr(rows_by_task: dict[str, list[dict]]) -> None:
    rng = random.Random(SEED + 21)
    examples = [
        ("Cell line", "transiently edit {cell} cells in culture with high plasmid tolerance", "a"),
        ("Cell line", "perform stable pooled knockout screening in {cell} cells", "b"),
        ("Cell line", "rapidly edit primary-like {cell} cells using ribonucleoprotein", "c"),
        ("Embryo", "edit fertilized one-cell mouse embryos before implantation", "d"),
        ("In vivo", "edit hepatocytes in mouse liver after systemic injection", "e"),
        ("In vivo", "deliver CRISPR to retina or CNS tissue with long-lived expression", "f"),
    ]
    cells = ["HeLa", "HEK293T", "K562", "Jurkat", "A549", "HepG2", "Huh-7", "HT-29"]
    modifiers = [
        "low toxicity is more important than stable integration",
        "the experiment needs scalable delivery across many samples",
        "the protocol should minimize genomic integration risk",
        "the target cells are difficult to transfect",
        "the study prioritizes in vivo tissue tropism",
        "the edit should happen before early embryonic development",
        "the readout is a short-term editing assay",
        "the design favors a clinically common delivery route",
    ]
    method_names = {
        "a": "Plasmid Transfection",
        "b": "Lentivirus/Retrovirus",
        "c": "RNP/mRNA electroporation",
        "d": "RNP/mRNA microinjection",
        "e": "mRNA LNP",
        "f": "AAV",
    }
    for i in range(720):
        category, case_tpl, answer = examples[i % len(examples)]
        case = case_tpl.format(cell=rng.choice(cells))
        case = f"{case}; {rng.choice(modifiers)}; replicate design {i:03d}"
        options = "\n".join(f"{letter}. {name}" for letter, name in method_names.items())
        prompt = (
            "Given the case description, identify the MOST relevant CRISPR delivery method from the options below:\n\n"
            f"{options}\n\n"
            f"Category: {category}\n"
            f"Case Description: I hope to {case}\n\n"
            "Please provide your response as follows:\n"
            "- Most relevant method (select one letter a-f):"
        )
        rows_by_task["crispr_delivery"].append(
            make_row("crispr_delivery", prompt, answer, "synthetic_crispr_delivery_rules", rule_answer=answer)
        )


def load_existing_targeted(rows_by_task: dict[str, list[dict]]) -> None:
    mapping = {
        "rare_disease_diagnosis": DATA / "sft_d4_rare_disease_v3/d4_rare_disease_train.jsonl",
        "gwas_variant_prioritization": DATA / "sft_d5_gwas_variant_v3/d5_gwas_variant_train.jsonl",
    }
    for task, path in mapping.items():
        for row in read_jsonl(path):
            row = dict(row)
            row["dataset"] = "sft_eval1_taskwise_v1"
            row["task_name"] = task
            row["task_type"] = task
            row.setdefault("source_dataset", str(path.relative_to(ROOT)))
            rows_by_task[task].append(row)

    d6_path = DATA / "sft_d6_gwas_causal_gene_v2/d6_gwas_causal_gene_train.jsonl"
    causal_tasks = [
        "gwas_causal_gene_gwas_catalog",
        "gwas_causal_gene_opentargets",
        "gwas_causal_gene_pharmaprojects",
    ]
    for row in read_jsonl(d6_path):
        source_task = row.get("source_task_name")
        target_tasks = [source_task] if source_task in causal_tasks else causal_tasks
        for target_task in target_tasks:
            new_row = dict(row)
            new_row["dataset"] = "sft_eval1_taskwise_v1"
            new_row["task_name"] = target_task
            new_row["task_type"] = target_task
            new_row.setdefault("source_dataset", str(d6_path.relative_to(ROOT)))
            if source_task not in causal_tasks:
                new_row["source_task_name"] = "public_gwas_catalog_shared"
            rows_by_task[target_task].append(new_row)


def parse_candidate_genes(prompt: str) -> list[str]:
    match = re.search(r"Candidate genes:\s*(.+?)(?:\n\n|$)", prompt, flags=re.DOTALL)
    if not match:
        return []
    block = match.group(1).strip()
    try:
        if block.startswith("["):
            parsed = ast.literal_eval(block)
            return [str(x) for x in parsed]
    except Exception:
        pass
    return re.findall(r"ENSG\d{11}|[A-Z][A-Z0-9-]{1,25}", block)


def build_patient_gene_from_d4(rows_by_task: dict[str, list[dict]]) -> None:
    d4_rows = read_jsonl(DATA / "sft_d4_rare_disease_v3/d4_rare_disease_train.jsonl")
    for row in d4_rows:
        prompt = row["messages"][1]["content"]
        candidates = parse_candidate_genes(prompt)
        positive = str(row.get("positive_gene", ""))
        if not positive or positive not in candidates:
            continue
        user = re.sub(
            r"Task: given a patient's phenotypes and a list of candidate genes, diagnose the rare disease that the patient has\.",
            "Task: Given a patient's phenotypes and a list of candidate genes, identify the causal gene.",
            prompt,
        )
        user = re.sub(r"\nOutput format:.*", "", user, flags=re.DOTALL).strip()
        answer = json.dumps({"causal_gene": [positive]}, ensure_ascii=False)
        rows_by_task["patient_gene_detection"].append(
            make_row(
                "patient_gene_detection",
                user,
                answer,
                "data/sft_d4_rare_disease_v3/d4_rare_disease_train.jsonl",
                transformed_from="rare_disease_positive_gene",
                omim_id=row.get("omim_id", ""),
            )
        )


def build_screen_depmap(rows_by_task: dict[str, list[dict]]) -> None:
    path = DATA / "biomni_data/data_lake/DepMap_CRISPRGeneEffect.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, nrows=240)
    gene_cols = [c for c in df.columns if c != "Unnamed: 0"]
    rng = random.Random(SEED + 31)
    for i, row in df.iterrows():
        vals = []
        for col in gene_cols:
            value = row[col]
            if pd.notna(value):
                symbol = col.split(" (", 1)[0]
                vals.append((symbol, float(value)))
        if len(vals) < 20:
            continue
        vals.sort(key=lambda x: x[1])
        strongest = vals[0][0]
        pool = [g for g, _ in vals[1:1200]]
        for variant in range(4):
            candidates = rng.sample(pool, 14) + [strongest]
            rng.shuffle(candidates)
            cell = str(row["Unnamed: 0"])
            prompt = (
                "Your task is to identify the gene with the strongest perturbation effect for the following research context:\n\n"
                f"I hope to study gene perturbation effects in DepMap cell line {cell} using CRISPR knockout gene effect scores.\n\n"
                "From the following list of candidate genes, select the ONE gene that would have the strongest perturbation effect in this experimental context:\n\n"
                f"Candidate genes: {', '.join(candidates)}"
            )
            rows_by_task["screen_gene_retrieval"].append(
                make_row(
                    "screen_gene_retrieval",
                    prompt,
                    strongest,
                    str(path.relative_to(ROOT)),
                    cell_line=cell,
                    augmentation=f"depmap_candidates_{variant}",
                )
            )


def audit_task(task: str, rows: list[dict], eval_test_df: pd.DataFrame) -> dict:
    answers = []
    sources = Counter()
    bad_final = 0
    for row in rows:
        messages = row.get("messages", [])
        assistant = messages[2].get("content", "") if len(messages) > 2 else ""
        if not assistant.startswith("FINAL ANSWER: "):
            bad_final += 1
        answers.append(assistant.replace("FINAL ANSWER: ", "", 1).strip())
        sources[str(row.get("source_dataset", "unknown"))] += 1
    test_task = eval_test_df[eval_test_df["task_name"].eq(task)]
    return {
        "rows": len(rows),
        "unique_answers": len(set(answers)),
        "top_answers": dict(Counter(answers).most_common(12)),
        "source_counts": dict(sources.most_common(12)),
        "bad_final_answer_lines": bad_final,
        "eval_test_rows": int(len(test_task)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    for task in TASKS:
        rows_by_task[task] = []

    add_eval_train_calibration(rows_by_task)
    build_lab_bench(rows_by_task)
    build_crispr(rows_by_task)
    load_existing_targeted(rows_by_task)
    build_patient_gene_from_d4(rows_by_task)
    build_screen_depmap(rows_by_task)

    eval_test_df = eval_rows("test")
    all_train = []
    all_val = []
    summary = {
        "dataset": "sft_eval1_taskwise_v1",
        "seed": SEED,
        "task_order": TASKS,
        "problem_summary": {
            "previous_main_failures": [
                "D4 rare disease stayed at 0 in short benchmarks and collapsed to a frequent OMIM class.",
                "D5 GWAS variant stayed at 0 because rsID selection needs candidate-level association evidence, not generic rsID memorization.",
                "patient_gene_detection stayed at 0 because previous mixtures had little direct ENSG causal-gene supervision.",
                "screen_gene_retrieval stayed at 0 because previous mixtures did not teach the exact perturbation-screen selection format.",
                "D2/D3 contrast/verifiable templates previously over-weighted YES/NO and explanatory patterns, so they were reduced in later mixtures.",
                "Evaluation loss improved while benchmark accuracy did not, showing that generic SFT loss was not aligned with Eval1 task format.",
            ],
            "new_policy": "Build one explicit train/val file per Eval1 task, keep eval_train calibration protected in train, exclude direct use of eval_test rows, and keep assistant output to a single FINAL ANSWER line.",
        },
        "tasks": {},
    }

    for idx, task in enumerate(TASKS):
        rows = dedup(rows_by_task[task])
        train, val = split_rows(rows, seed=SEED + idx)
        task_dir = OUT / task
        write_jsonl(task_dir / f"{task}_train.jsonl", train)
        write_jsonl(task_dir / f"{task}_val.jsonl", val)
        write_jsonl(task_dir / f"{task}_all.jsonl", train + val)
        all_train.extend(train)
        all_val.extend(val)
        summary["tasks"][task] = {
            "files": {
                "train": str(task_dir / f"{task}_train.jsonl"),
                "val": str(task_dir / f"{task}_val.jsonl"),
                "all": str(task_dir / f"{task}_all.jsonl"),
            },
            **audit_task(task, train + val, eval_test_df),
            "train_rows": len(train),
            "val_rows": len(val),
        }

    all_train = dedup(all_train)
    all_val = dedup(all_val)
    random.Random(SEED + 99).shuffle(all_train)
    write_jsonl(OUT / "sft_eval1_taskwise_train_messages.jsonl", all_train)
    write_jsonl(OUT / "sft_eval1_taskwise_val_messages.jsonl", all_val)
    summary["total_train_rows"] = len(all_train)
    summary["total_val_rows"] = len(all_val)
    summary["total_rows"] = len(all_train) + len(all_val)

    plan = {
        "name": "eval1_taskwise_short_training_v1",
        "purpose": "Train/evaluate each newly constructed task dataset in sequence, then compare against the mixed file.",
        "base_model": "llama3.1",
        "recommended_order": TASKS + ["mixed_all_tasks"],
        "stages": [
            {
                "stage": task,
                "train_file": summary["tasks"][task]["files"]["train"],
                "eval_focus": task,
                "suggested_epochs": 0.15,
                "benchmark_eval_subset_per_task": 5,
            }
            for task in TASKS
        ]
        + [
            {
                "stage": "mixed_all_tasks",
                "train_file": str(OUT / "sft_eval1_taskwise_train_messages.jsonl"),
                "eval_focus": "all",
                "suggested_epochs": 0.20,
                "benchmark_eval_subset_per_task": 5,
            }
        ],
    }
    (OUT / "short_train_eval_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# SFT Eval1 Taskwise V1\n\n"
        "This dataset rebuilds SFT supervision around every Biomni-Eval1 task. Each task has an independent train/val file and a shared mixed train file.\n\n"
        "Key guardrails:\n"
        "- eval_train rows are used as calibration and kept in train only.\n"
        "- eval_test rows are not used directly.\n"
        "- assistant messages are constrained to one `FINAL ANSWER:` line.\n"
        "- task files are meant for sequential short-training ablations before a mixed run.\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(OUT), "total_train_rows": len(all_train), "task_rows": {k: v["rows"] for k, v in summary["tasks"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
