"""
scripts/run_local_infer.py

本地推理封装：加载 SFT/基础模型并提供 generate_text 接口。

默认模型路径：output/qwen2.5_sft
可通过环境变量 LOCAL_SFT_MODEL_DIR 覆盖。

注意：
- Qwen 系列通常需要 trust_remote_code=True。
- 对于大模型，建议把模型放在支持的 GPU 机器并使用 fastapi_server.py 服务化。
"""
import os
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR_DEFAULT = os.getenv("LOCAL_SFT_MODEL_DIR", "output/qwen2.5_sft")
BASE_MODEL_FALLBACK = "/root/autodl-tmp/Biomni-main/models/Qwen2.5-7B-Instruct/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

class LocalInfer:
    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None):
        self.model_dir = model_dir or MODEL_DIR_DEFAULT
        # device 默认：cuda（如果有）或 cpu
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        # 先尝试从本地目录加载，否则回退到 base model（可能会从远程下载）
        model_source = self.model_dir if os.path.isdir(self.model_dir) else BASE_MODEL_FALLBACK
        print(f"[LocalInfer] Loading model from: {model_source} (device hint={self.device})")
        try:
            # 注意：对于大模型建议在命令行使用 accelerate / device_map 等配置。
            # 这里尝试使用 device_map="auto" 当 GPU 可用时，让 transformers 做分配（需要 accelerate 支持）。
            self.tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
            if torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    trust_remote_code=True,
                    device_map="auto",    # 依赖 accelerate/transformers 支持
                    torch_dtype="auto",
                )
                print("[LocalInfer] Model loaded with device_map='auto'.")
            else:
                # CPU 加载：尽量节省内存（low_cpu_mem_usage）
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                self.model.to(self.device)
                print("[LocalInfer] Model loaded on CPU.")
        except Exception as e:
            raise RuntimeError(f"[LocalInfer] 模型加载失败: {e}")
    def format_qwen_prompt(user_prompt: str) -> str:
        """
        若输入非 Qwen 格式，则自动封装为 Qwen 官方对话模板。
        若已是完整 Qwen prompt（含 <|im_start|> 等），则原样返回。
        """
        if "<|im_start|>" in user_prompt and "<|im_end|>" in user_prompt:
            return user_prompt
        # 否则自动包裹
        return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: Optional[bool] = None,
        stop_tokens: Optional[List[str]] = None,
    ) -> str:
        """
        生成文本并返回字符串（只返回模型生成的后缀，不包含原始 prompt）。
        参数名与 Gradio 前端保持一致。
        """
        do_sample = (temperature > 0.0) if do_sample is None else do_sample
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # 把 inputs 放到适当设备（若 model 使用 device_map='auto'，transformers ���自动迁移输入）
        try:
            if hasattr(self.model, "device") and isinstance(self.model.device, torch.device):
                input_ids = input_ids.to(self.model.device)
                attention_mask = attention_mask.to(self.model.device)
            else:
                # device_map='auto' or sharded model: 让 generate 自行处理（通常 acceptable）
                pass
        except Exception:
            pass

        # gen_kwargs = dict(
        #     max_new_tokens=max_new_tokens,
        #     temperature=temperature,
        #     top_p=top_p,
        #     do_sample=do_sample,
        #     eos_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else None,
        #     pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else None,
        # )
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            # 不传 eos_token_id/pad_token_id！
            # 若需要 pad_token_id 可加，但一般不需要
            # WARNING: eos_token_id/pad_token_id 不要传，否则模型会提前输出 stop token 导致空输出
        )
        try:
            with torch.no_grad():
                out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            # out[0] 是包含输入 + 生成的 token id 序列。我们只返回新生成的部分：
            generated_ids = out[0][input_ids.shape[-1]:] if out is not None and len(out) > 0 else []
            if isinstance(generated_ids, torch.Tensor) and generated_ids.numel() > 0:
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                # 若无法切片（某些模型/配置），退回到全部 decode 并剔除 prompt 前缀（尝试）
                full = self.tokenizer.decode(out[0], skip_special_tokens=True)
                if full.startswith(prompt):
                    text = full[len(prompt):].lstrip()
                else:
                    text = full
            # 处理 stop_tokens（简单截断）
            if stop_tokens:
                for t in stop_tokens:
                    idx = text.find(t)
                    if idx != -1:
                        text = text[:idx]
            return text.strip()
        except Exception as e:
            return f"[本地推理失败] {e}"

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Hello world")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()
    infer = LocalInfer(model_dir=args.model_dir)
    print(infer.generate_text(prompt=args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p))