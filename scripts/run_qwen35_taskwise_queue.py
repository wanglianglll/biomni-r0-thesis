#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path('/root/autodl-tmp/Biomni-main')
CONFIG_DIR = ROOT / 'configs/eval1_taskwise_short_v1_qwen35_lora'
LOG_DIR = ROOT / 'logs'
SUMMARY_PATH = LOG_DIR / 'qwen35_lora_taskwise_queue_summary.json'
QUEUE_LOG = LOG_DIR / f'qwen35_lora_taskwise_queue_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
PID_PATH = LOG_DIR / 'qwen35_lora_taskwise_queue.pid'

TASKS = [
    'rare_disease_diagnosis',
    'gwas_variant_prioritization',
    'patient_gene_detection',
    'screen_gene_retrieval',
    'gwas_causal_gene_gwas_catalog',
    'gwas_causal_gene_opentargets',
    'gwas_causal_gene_pharmaprojects',
    'lab_bench_dbqa',
    'lab_bench_seqqa',
    'crispr_delivery',
]


def log(msg: str) -> None:
    text = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(text, flush=True)
    with QUEUE_LOG.open('a', encoding='utf-8') as f:
        f.write(text + '\n')


def run_cmd(cmd: list[str] | str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=isinstance(cmd, str), cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def proc_lines(pattern: str) -> list[str]:
    res = run_cmd(f"pgrep -af {json.dumps(pattern)} || true")
    return [line for line in res.stdout.splitlines() if line.strip() and 'pgrep -af' not in line]


def current_task_process(task: str) -> list[str]:
    cfg = f"configs/eval1_taskwise_short_v1_qwen35_lora/train_qwen35_lora_{task}.json"
    return proc_lines(f"run_train.py --config {cfg}")


def find_outputs_for_task(task: str) -> list[Path]:
    cfg_suffix = f"configs/eval1_taskwise_short_v1_qwen35_lora/train_qwen35_lora_{task}.json"
    out = []
    for meta in (ROOT / 'scripts/output').glob('*qwen3.5_lora_sft/*_metadata.json'):
        try:
            data = json.loads(meta.read_text(encoding='utf-8'))
        except Exception:
            continue
        if str(data.get('config_path', '')).endswith(cfg_suffix):
            out.append(meta.parent)
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def latest_finished_output(task: str) -> Path | None:
    for out in find_outputs_for_task(task):
        status_path = out / 'live_status.json'
        try:
            status = json.loads(status_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if status.get('status') == 'finished':
            return out
    return None


def summarize_output(task: str, out: Path | None) -> dict:
    item = {'task': task, 'output_dir': str(out) if out else None, 'status': 'missing'}
    if not out:
        return item
    status_path = out / 'live_status.json'
    bench_path = out / 'benchmark_latest.json'
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding='utf-8'))
            item.update({
                'status': status.get('status'),
                'global_step': status.get('global_step'),
                'max_steps': status.get('max_steps'),
                'seconds_per_step': status.get('seconds_per_step'),
                'train_loss': (status.get('latest_log') or {}).get('train_loss'),
            })
        except Exception as exc:
            item['status_error'] = str(exc)
    if bench_path.exists():
        try:
            bench = json.loads(bench_path.read_text(encoding='utf-8'))
            item.update({
                'benchmark_accuracy': bench.get('accuracy'),
                'benchmark_examples': bench.get('num_examples'),
                'template_issue_rate': bench.get('template_issue_rate'),
                'task_summary': bench.get('task_summary'),
            })
        except Exception as exc:
            item['benchmark_error'] = str(exc)
    return item


def write_summary(records: list[dict], queue_status: str) -> None:
    payload = {
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'queue_status': queue_status,
        'tasks': records,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def wait_for_task(task: str) -> Path | None:
    last_report = 0.0
    while True:
        lines = current_task_process(task)
        if not lines:
            return latest_finished_output(task)
        if time.time() - last_report > 300:
            log(f"waiting task={task}; process={lines[0]}")
            out = find_outputs_for_task(task)
            if out:
                log(f"latest output for {task}: {out[0]}")
            last_report = time.time()
        time.sleep(60)


def start_task(task: str) -> subprocess.Popen:
    cfg = CONFIG_DIR / f'train_qwen35_lora_{task}.json'
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    task_log = LOG_DIR / f'taskwise_qwen35_lora_{task}_queue_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    f = task_log.open('w', encoding='utf-8')
    cmd = ['/root/miniconda3/envs/biomni/bin/python', 'scripts/run_train.py', '--config', str(cfg.relative_to(ROOT))]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    log(f"started task={task}; pid={proc.pid}; log={task_log}")
    return proc


def shutdown_instance() -> None:
    log('all taskwise short trainings completed; initiating shutdown')
    script = ROOT / 'scripts/autodl_shutdown.py'
    if script.exists():
        subprocess.Popen(['/root/miniconda3/envs/biomni/bin/python', str(script)], cwd=ROOT, start_new_session=True)
    else:
        subprocess.Popen(['/usr/bin/shutdown', '-h', 'now'], cwd=ROOT, start_new_session=True)


def main() -> None:
    PID_PATH.write_text(str(os.getpid()), encoding='utf-8')
    records: list[dict] = []
    write_summary(records, 'running')
    log('queue started')
    for task in TASKS:
        finished = latest_finished_output(task)
        if finished:
            rec = summarize_output(task, finished)
            records.append(rec)
            write_summary(records, f'skipped_finished:{task}')
            log(f"skip finished task={task}; acc={rec.get('benchmark_accuracy')}; output={finished}")
            continue
        running = current_task_process(task)
        if running:
            log(f"detected running task={task}; waiting existing process")
            out = wait_for_task(task)
        else:
            start_task(task)
            out = wait_for_task(task)
        rec = summarize_output(task, out)
        records.append(rec)
        write_summary(records, f'finished:{task}')
        log(f"finished task={task}; status={rec.get('status')}; acc={rec.get('benchmark_accuracy')}; output={rec.get('output_dir')}")
    write_summary(records, 'completed')
    shutdown_instance()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        log(f'queue failed: {type(exc).__name__}: {exc}')
        write_summary([], f'failed:{type(exc).__name__}')
        raise
