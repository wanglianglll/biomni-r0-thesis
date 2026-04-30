from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = PROJECT_ROOT / ".autodl_msg_token"
API_URL = "https://www.autodl.com/api/v1/wechat/message/send"


def get_autodl_token() -> str:
    token = os.environ.get("AUTODL_TOKEN") or os.environ.get("AUTODL_API_TOKEN")
    if token:
        return token.strip()
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    return ""


def send_autodl_message(title: str, name: str = "Biomni training", content: str = "") -> bool:
    token = get_autodl_token()
    if not token:
        return False
    payload = json.dumps({"title": title[:80], "name": name[:80], "content": content[:1800]}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Authorization": token, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", "replace")
        return 200 <= getattr(resp, "status", 200) < 300 and "error" not in body.lower()
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
