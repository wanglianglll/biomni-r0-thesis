"""
app.py

Gradio Web 原型（浏览器界面），默认通过本地 FastAPI 服务调用本地推理（http://127.0.0.1:8000/generate）。
也支持“直接导入” LocalInfer（如果你希望在同一进程中加载模型并运行，请将 USE_DIRECT_IMPORT 环境变量设为 "1"）。

运行步骤（推荐）：
1) 启动本地服务（推荐，模型只加载一次）：
   python -m uvicorn scripts.fastapi_server:app --host 0.0.0.0 --port 8000 --workers 1
2) 启动前端：
   python app.py
3) 浏览器打开 http://localhost:7860

如果想在单个进程内运行（小模型或 CPU），设置环境变量：
  export USE_DIRECT_IMPORT=1
然后直接运行： python app.py
"""
import os
import requests
from typing import List, Tuple
import gradio as gr

# 配置
FASTAPI_URL = os.getenv("LOCAL_INFER_URL", "http://127.0.0.1:6006/generate")
USE_DIRECT_IMPORT = os.getenv("USE_DIRECT_IMPORT", "0") == "1"

# 如果选择直接导入模式，则导入 LocalInfer 并实例化（会在此进程加载模型）
local_infer = None
if USE_DIRECT_IMPORT:
    try:
        from scripts.run_local_infer import LocalInfer
        local_infer = LocalInfer()
        print("[app.py] 使用直接导入模式加载模型（注意：在此进程将加载模型）。")
    except Exception as e:
        local_infer = None
        print(f"[app.py] 直接导入 LocalInfer 失败: {e}. 将回退到 HTTP 模式。")
        USE_DIRECT_IMPORT = False

def call_local_model_http(prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
    try:
        payload = {
            "prompt": prompt,
            "temperature": float(temperature),
            "top_new_tokens": int(max_tokens),  # 兼容性字段名
            "max_new_tokens": int(max_tokens),
            "top_p": float(top_p),
        }
        # timeout 根据模型大小调整（单位秒）
        r = requests.post(FASTAPI_URL, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data.get("text", "[服务返回格式异常] " + str(data))
    except Exception as e:
        return f"[本地服务调用失败] {e}"

def call_local_model_direct(prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
    if local_infer is None:
        return "[本地直接导入未初始化]"
    try:
        return local_infer.generate_text(prompt=prompt, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p)
    except Exception as e:
        return f"[本地直接调用失败] {e}"

def call_deepseek_api(prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
    """
    Deepseek API gpt兼容方式
    """
    import json
    DEEPSEEK_API_KEY = os.environ.get("CUSTOM_MODEL_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    ENDPOINT = os.environ.get("CUSTOM_MODEL_BASE_URL") or os.environ.get("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/v1/chat/completions")
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",  # 或 deepseek-coder
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens)
    }
    try:
        r = requests.post(ENDPOINT, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        # deepseek v1 兼容gpt格式
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Deepseek API 调用失败] {e}"

def respond(choice, prompt, temperature, top_p, max_tokens, history):
    if not prompt.strip():
        return history, ""
    if choice == "Local (direct import)":
        reply = call_local_model_direct(prompt, temperature, top_p, max_tokens)
    elif choice == "Local (HTTP service)":
        reply = call_local_model_http(prompt, temperature, top_p, max_tokens)
    else:
        reply = call_deepseek_api(prompt, temperature, top_p, max_tokens)

    # 新 history 列表
    new_history = history + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply}
    ]
    return new_history, ""

with gr.Blocks() as demo:
    gr.Markdown("## Biomni-R0 thesis — 本地推理 Web 原型")
    with gr.Row():
        model_choice = gr.Dropdown(
            choices=["Local (HTTP service)", "Local (direct import)", "Deepseek API"],
            value="Local (HTTP service)" if not USE_DIRECT_IMPORT else "Local (direct import)",
            label="模型/后端选择",
        )
        temp = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Temperature")
        top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Top-p")
        max_tokens = gr.Slider(16, 2048, value=512, step=1, label="Max tokens")
    chatbot = gr.Chatbot(label="Chat")
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="在此输入 prompt 并回车或点击发送...")
        send = gr.Button("Send")

    history_state = gr.State([])

    send.click(fn=respond, inputs=[model_choice, txt, temp, top_p, max_tokens, history_state], outputs=[chatbot, txt])

if __name__ == "__main__":
    # 推荐启用 queue 防止并发冲突
    demo.queue().launch(server_name="127.0.0.1", server_port=6008, share=False)