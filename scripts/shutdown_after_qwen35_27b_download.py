#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp/Biomni-main")
LOG = ROOT / "logs/shutdown_after_qwen35_27b_download.log"
CHECK = ROOT / "scripts/check_qwen35_27b_ready.py"
SHUTDOWN = ROOT / "scripts/autodl_shutdown.py"
MARKER = ROOT / "logs/qwen35_27b_download_ready_shutdown.marker"


def write(msg: str) -> None:
    line = f"[{datetime.now().strftime('%F %T')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def download_running() -> bool:
    proc = run(["bash", "-lc", "pgrep -af '/tmp/download_qwen35_27b_curl.sh|curl .*Qwen3.5-27B' || true"])
    return bool(proc.stdout.strip())


def ready() -> bool:
    proc = run(["/root/miniconda3/envs/biomni/bin/python", str(CHECK)])
    try:
        report = json.loads(proc.stdout)
    except Exception:
        report = {"parse_error": proc.stdout[-500:], "stderr": proc.stderr[-500:]}
    write("readiness=" + json.dumps(report, ensure_ascii=False)[:1200])
    return proc.returncode == 0


def shutdown() -> None:
    MARKER.write_text(datetime.now().isoformat(), encoding="utf-8")
    write("download complete; invoking AutoDL shutdown")
    if SHUTDOWN.exists():
        subprocess.Popen(["/root/miniconda3/envs/biomni/bin/python", str(SHUTDOWN)], cwd=ROOT)
    else:
        subprocess.Popen(["/usr/bin/shutdown", "-h", "now"])


def main() -> int:
    write("watchdog started")
    while True:
        if ready():
            shutdown()
            return 0
        if not download_running():
            write("download process is not running and model is not ready; not shutting down")
            return 2
        time.sleep(300)


if __name__ == "__main__":
    raise SystemExit(main())
