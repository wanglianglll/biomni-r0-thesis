from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPT_RESULTS_DIR = PROJECT_ROOT / "scripts" / "results"
SCRIPT_OUTPUT_DIR = PROJECT_ROOT / "scripts" / "output"


@dataclass(frozen=True)
class LocalModelSpec:
    family: str
    label: str
    trust_remote_code: bool
    base_model_dir: str
    sft_model_dir: str
    base_results_dir: str
    sft_results_dir: str


LOCAL_MODELS = {
    "qwen2.5": LocalModelSpec(
        family="qwen2.5",
        label="Qwen2.5-7B-Instruct",
        trust_remote_code=True,
        base_model_dir="/root/autodl-tmp/Biomni-main/models/Qwen2.5-7B-Instruct/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        sft_model_dir="/root/autodl-tmp/Biomni-main/scripts/output/qwen2.5_sft",
        base_results_dir=str(SCRIPT_RESULTS_DIR / "qwen2.5_base"),
        sft_results_dir=str(RESULTS_DIR / "qwen2.5_sft"),
    ),
    "qwen3.5": LocalModelSpec(
        family="qwen3.5",
        label="Qwen3.5-9B",
        trust_remote_code=True,
        base_model_dir="/root/autodl-tmp/Biomni-main/models/Qwen3.5-9B",
        sft_model_dir="/root/autodl-tmp/Biomni-main/models/Qwen3.5-9B",
        base_results_dir=str(SCRIPT_RESULTS_DIR / "qwen3.5_base"),
        sft_results_dir=str(RESULTS_DIR / "qwen3.5_sft"),
    ),
    "qwen3.5-27b": LocalModelSpec(
        family="qwen3.5-27b",
        label="Qwen3.5-27B",
        trust_remote_code=True,
        base_model_dir="/root/autodl-tmp/Biomni-main/models/Qwen3.5-27B",
        sft_model_dir="/root/autodl-tmp/Biomni-main/scripts/output/qwen3.5-27b_qlora_sft",
        base_results_dir=str(SCRIPT_RESULTS_DIR / "qwen3.5-27b_base"),
        sft_results_dir=str(RESULTS_DIR / "qwen3.5-27b_sft"),
    ),
    "llama3.1": LocalModelSpec(
        family="llama3.1",
        label="Meta-Llama-3.1-8B-Instruct",
        trust_remote_code=False,
        base_model_dir="/root/autodl-tmp/Biomni-main/models/Meta-Llama-3.1-8B-Instruct",
        sft_model_dir="/root/autodl-tmp/Biomni-main/scripts/output/20260428_123042_llama3.1_qlora_sft",
        base_results_dir=str(SCRIPT_RESULTS_DIR / "llama3.1_base"),
        sft_results_dir=str(RESULTS_DIR / "llama3.1_sft"),
    ),
    "mistral": LocalModelSpec(
        family="mistral",
        label="Mistral-7B-Instruct-v0.3",
        trust_remote_code=False,
        base_model_dir="/root/autodl-tmp/Biomni-main/models/Mistral-7B-Instruct-v0.3",
        sft_model_dir="/root/autodl-tmp/Biomni-main/scripts/output/mistral_sft",
        base_results_dir=str(SCRIPT_RESULTS_DIR / "mistral_base"),
        sft_results_dir=str(RESULTS_DIR / "mistral_sft"),
    ),
}


def get_local_model_spec(model_family: str) -> LocalModelSpec:
    try:
        return LOCAL_MODELS[model_family]
    except KeyError as exc:
        supported = ", ".join(sorted(LOCAL_MODELS))
        raise ValueError(f"Unsupported local model family: {model_family}. Supported: {supported}") from exc
