#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoDL-friendly shutdown helper.")
    parser.add_argument("--delay-seconds", type=int, default=10)
    parser.add_argument("--message", default="Biomni task finished; shutting down the AutoDL instance.")
    args = parser.parse_args()

    try:
        from scripts.autodl_notify import send_autodl_message

        send_autodl_message("Biomni instance shutting down", name="AutoDL shutdown", content=args.message)
    except Exception:
        pass

    if args.delay_seconds > 0:
        time.sleep(args.delay_seconds)

    # AutoDL's own save-money guide recommends the absolute /usr/bin/shutdown
    # path. Using /usr/sbin/shutdown is not portable on their minimized images.
    subprocess.Popen("/usr/bin/shutdown -h now", shell=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
