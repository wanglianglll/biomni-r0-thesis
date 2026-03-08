import os
import requests

url = "https://api.deepseek.com/v1/chat/completions"
api_key = os.getenv("CUSTOM_MODEL_API_KEY")

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": "一句话告诉我：什么是强化学习？"}
    ]
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

resp = requests.post(url, json=payload, headers=headers)
print(resp.status_code)
print(resp.text)
