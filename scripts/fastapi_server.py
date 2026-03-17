"""
scripts/fastapi_server.py

FastAPI 服务：启动时加载模型一次，暴露 /generate 接口供前端（Gradio）或其他客户端调用。

启动示例（在项目根目录运行）：
  pip install -r requirements.txt
  uvicorn scripts.fastapi_server:app --host 0.0.0.0 --port 8000 --workers 1

注意：
- 建议使用 --workers 1 来避免模型在多个 worker 中重复加载（占用多份显存）。
"""
from typing import Optional, List
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse

# 相对导入 LocalInfer（确保从项目根运行 uvicorn）
from scripts.run_local_infer import LocalInfer

def format_qwen_prompt(user_prompt: str) -> str:
    if "<|im_start|>" in user_prompt and "<|im_end|>" in user_prompt:
        return user_prompt
    return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

app = FastAPI(title="Local SFT Inference Service")

MODEL_DIR = os.getenv("LOCAL_SFT_MODEL_DIR", "output/sft_qwen2_7b")
# 你可以通过环境变量 USE_CACHE（示例）或其他方式控制行为

infer: Optional[LocalInfer] = None

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = None
    stop_tokens: Optional[List[str]] = None

@app.on_event("startup")
def startup_event():
    global infer
    # 只加载一次模型
    try:
        infer = LocalInfer(model_dir=MODEL_DIR)
        print("[fastapi_server] Model loaded.")
    except Exception as e:
        # 如果模型加载失败，保留 infer 为 None 并在请求时返回错误
        infer = None
        print(f"[fastapi_server] Model load failed: {e}")

@app.post("/generate")
def generate(req: GenerateRequest):
    global infer
    if infer is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})
    # 智能自动格式化 prompt
    prompt = format_qwen_prompt(req.prompt)
    try:
        text = infer.generate_text(
            prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.do_sample,
            stop_tokens=getattr(req, "stop_tokens", None),
        )
        return {"text": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Local inference failed: {e}"})