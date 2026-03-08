import os
from dotenv import load_dotenv

load_dotenv(".env", override=True)
print("PY ENV LLM_SOURCE =", os.getenv("LLM_SOURCE"))
print("PY ENV CUSTOM_MODEL_BASE_URL =", os.getenv("CUSTOM_MODEL_BASE_URL"))
print("PY ENV CUSTOM_MODEL_API_KEY =", (os.getenv("CUSTOM_MODEL_API_KEY") or "")[:6], "...")

from biomni.agent import A1
from biomni.config import default_config

# Biomni 的数据库/索引部分默认用 Anthropic，我们先关闭这个逻辑
default_config.llm = "custom"
default_config.source = "Custom"

# 让 Agent 的推理也走我们自定义 API（DeepSeek）
agent = A1(
    llm="deepseek-chat",
    source="Custom",
    base_url=os.getenv("CUSTOM_MODEL_BASE_URL"),
    api_key=os.getenv("CUSTOM_MODEL_API_KEY")
)

res = agent.go("请用一句话告诉我强化学习是什么？")
print(res)
