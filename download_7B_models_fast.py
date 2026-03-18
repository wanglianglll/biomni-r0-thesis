from huggingface_hub import snapshot_download
import os

# 下载目录
target_dir = "./models/Mistral-7B-Instruct-v0.3"
os.makedirs(target_dir, exist_ok=True)

print('==== Downloading mistralai/Mistral-7B-Instruct-v0.3 ====')
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    max_workers=12,
    resume_download=True,
    # endpoint参数一般只用于公开模型，如国内网络可选如"https://hf-mirror.com"
    # endpoint="https://hf-mirror.com"  # 如需镜像可加上（但Instruct模型一般可直接拉）
)
print("Download complete.")
