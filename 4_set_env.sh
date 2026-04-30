# 如果你的模型不在默认 output/qwen2.5_sft，设置实际路径：
export LOCAL_SFT_MODEL_DIR="/scripts/output/qwen2.5_sft"

# 如果需要 Deepseek API（注意：不要把真实密钥推到 git）
export CUSTOM_MODEL_BASE_URL=https://api.deepseek.com/v1
export CUSTOM_MODEL_API_KEY=sk-6bad291cda504d12a931a3ece8f999b0

export CONDA_BASE=$(conda info --base)
export ENV_PYTHON="$CONDA_BASE/envs/biomni/bin/python"