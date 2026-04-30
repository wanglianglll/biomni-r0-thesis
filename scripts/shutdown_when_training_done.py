#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from scripts.autodl_notify import send_autodl_message

def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--interval', type=int, default=300)
    ap.add_argument('--shutdown', action='store_true')
    args = ap.parse_args()
    log = Path(args.log)
    send_autodl_message('Biomni shutdown watchdog armed', 'training watchdog', f'Watching pid={args.pid}\nlog={log}')
    while alive(args.pid):
        time.sleep(args.interval)
    text = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
    if 'Training finished.' in text or 'Training finished' in text:
        send_autodl_message('Biomni training finished', 'training watchdog', f'Training pid={args.pid} finished. Shutdown={args.shutdown}\nlog={log}')
        if args.shutdown:
            send_autodl_message('Biomni instance shutdown', 'training watchdog', 'Training finished; executing /usr/bin/shutdown -h now')
            os.system('/usr/bin/shutdown -h now')
    else:
        send_autodl_message('Biomni training stopped unexpectedly', 'training watchdog', f'pid={args.pid} exited but finish marker was not found. Instance will NOT be shut down.\nlog={log}')

if __name__ == '__main__':
    main()
