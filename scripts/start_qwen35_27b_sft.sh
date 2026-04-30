#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export CUDA_MODULE_LOADING=LAZY
export HF_PARALLEL_LOADING_WORKERS=1
export HF_ENABLE_PARALLEL_LOADING=false
cd /root/autodl-tmp/Biomni-main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate biomni
python scripts/run_train.py --config configs/train_qwen35_27b_qlora_eval1_taskwise_safe.json
