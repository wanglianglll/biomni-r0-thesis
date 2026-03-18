import os
import gradio as gr
import requests
from typing import List, Dict, Any

FASTAPI_URL = os.getenv("LOCAL_INFER_URL", "http://127.0.0.1:6006/generate")
USE_DIRECT_IMPORT = os.getenv("USE_DIRECT_IMPORT", "0") == "1"
local_infer = None
if USE_DIRECT_IMPORT:
    try:
        from scripts.run_local_infer import LocalInfer
        local_infer = LocalInfer()
    except Exception as e:
        local_infer = None
        USE_DIRECT_IMPORT = False

# 多轮对话历史格式化
def build_context(history: List[Dict[str, Any]], user_prompt: str) -> str:
    context = ""
    for msg in history:
        if msg["role"] == "user":
            context += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "assistant":
            context += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    context += f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    return context

def call_local_model_http(prompt, temperature, top_p, max_tokens):
    try:
        payload = {
            "prompt": prompt,
            "temperature": float(temperature),
            "max_new_tokens": int(max_tokens),
            "top_p": float(top_p),
        }
        r = requests.post(FASTAPI_URL, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data.get("text", "")
    except Exception as e:
        return f"[本地服务调用失败] {e}"

def call_local_model_direct(prompt, temperature, top_p, max_tokens):
    if local_infer is None:
        return "[本地直接导入未初始化]"
    try:
        return local_infer.generate_text(prompt=prompt, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p)
    except Exception as e:
        return f"[本地直接调用失败] {e}"

def call_deepseek_api(prompt, temperature, top_p, max_tokens):
    import json
    DEEPSEEK_API_KEY = os.environ.get("CUSTOM_MODEL_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens)
    }
    try:
        r = requests.post(ENDPOINT, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Deepseek API 调用失败] {e}"

# 核心历史维护
def respond(choice, prompt, temperature, top_p, max_tokens, history):
    if not prompt.strip():
        return history, "", history
    full_prompt = build_context(history, prompt)
    if choice == "Local (direct import)":
        reply = call_local_model_direct(full_prompt, temperature, top_p, max_tokens)
    elif choice == "Local (HTTP service)":
        reply = call_local_model_http(full_prompt, temperature, top_p, max_tokens)
    else:
        reply = call_deepseek_api(full_prompt, temperature, top_p, max_tokens)
    # 新增历史
    new_history = history + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply}
    ]
    return new_history, "", new_history

def clear_history():
    return [], "", []

with gr.Blocks() as demo:
    gr.Markdown("## Biomni-R0 thesis — 本地推理 Web 原型\n多轮聊天历史（对话窗口保留），支持上下文记忆")
    with gr.Row():
        model_choice = gr.Dropdown(
            choices=["Local (HTTP service)", "Local (direct import)", "Deepseek API"],
            value="Local (HTTP service)" if not USE_DIRECT_IMPORT else "Local (direct import)",
            label="模型/后端选择",
        )
        temp = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Temperature")
        top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Top-p")
        max_tokens = gr.Slider(16, 2048, value=512, step=1, label="Max tokens")
    chatbot = gr.Chatbot(label="Chat", value=[])
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="在此输入 prompt 并回车或点击发送...")
        send = gr.Button("Send")
        clear = gr.Button("清空历史")

    history_state = gr.State([])

    # outputs 三个，保持历史+文本框+状态同步
    send.click(fn=respond, inputs=[model_choice, txt, temp, top_p, max_tokens, history_state], outputs=[chatbot, txt, history_state])
    clear.click(fn=clear_history, inputs=[], outputs=[chatbot, txt, history_state])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=6008, share=False)
