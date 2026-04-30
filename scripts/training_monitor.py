#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "scripts" / "output"

app = FastAPI(title="Biomni Training Monitor")


def latest_run_dir(output_root: Path = OUTPUT_ROOT) -> Path | None:
    if not output_root.exists():
        return None
    dirs = [p for p in output_root.iterdir() if p.is_dir()]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl_tail(path: Path, limit: int = 30) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(x) for x in lines[-limit:] if x.strip()]
    except Exception:
        return []


def gpu_smi() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        ).strip()
        if not out:
            return {"available": False}
        name, used, free, total, util, power, temp = [x.strip() for x in out.split(",")[:7]]
        return {
            "available": True,
            "name": name,
            "memory_used_mb": float(used),
            "memory_free_mb": float(free),
            "memory_total_mb": float(total),
            "utilization_gpu_percent": float(util),
            "power_draw_w": float(power),
            "temperature_c": float(temp),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def read_run_payload() -> tuple[Path | None, dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    run_dir = latest_run_dir()
    if run_dir is None:
        return None, {}, None, []
    live = read_json(run_dir / "live_status.json") or {}
    meta_files = sorted(run_dir.glob("*_metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    metadata = read_json(meta_files[0]) if meta_files else None
    events = read_jsonl_tail(run_dir / "live_events.jsonl", 30)
    return run_dir, live, metadata, events


@app.get("/api/status")
def api_status():
    run_dir, live, metadata, events = read_run_payload()
    if run_dir is None:
        return JSONResponse({"status": "no_run", "message": "No run directory found", "gpu_smi": gpu_smi()})
    return JSONResponse(
        {
            "run_dir": str(run_dir),
            "live": live,
            "metadata": metadata,
            "events_tail": events,
            "gpu_smi": gpu_smi(),
        }
    )


@app.get("/api/benchmark")
def api_benchmark():
    run_dir = latest_run_dir()
    if run_dir is None:
        return JSONResponse({"status": "no_run", "message": "No run directory found"})
    latest = read_json(run_dir / "benchmark_latest.json")
    events = read_jsonl_tail(run_dir / "benchmark_events.jsonl", 50)
    return JSONResponse(
        {
            "run_dir": str(run_dir),
            "has_benchmark": latest is not None,
            "latest": latest,
            "events_tail": events,
        }
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Biomni Training Monitor</title>
<style>
:root{--bg:#0f172a;--card:#111827;--muted:#94a3b8;--text:#e5e7eb;--line:#334155;--good:#22c55e;--warn:#f59e0b;--bad:#ef4444;--cyan:#06b6d4}
body{margin:0;font-family:ui-sans-serif,system-ui,"Segoe UI",sans-serif;background:radial-gradient(circle at 20% 0%,#1e293b,var(--bg));color:var(--text)}
main{max-width:1280px;margin:0 auto;padding:28px}
h1{margin:0 0 8px;font-size:30px}.sub{color:var(--muted);margin-bottom:22px}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.three{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.card{background:rgba(17,24,39,.88);border:1px solid rgba(148,163,184,.18);border-radius:18px;padding:16px;box-shadow:0 18px 50px rgba(0,0,0,.22)}
.label{color:var(--muted);font-size:13px}.value{font-size:26px;font-weight:750;margin-top:5px}.small{font-size:13px;color:var(--muted)}
.progress{height:16px;background:#020617;border-radius:999px;overflow:hidden;border:1px solid var(--line)}.bar{height:100%;width:0%;background:linear-gradient(90deg,var(--good),var(--cyan));transition:width .5s ease}
pre{white-space:pre-wrap;overflow:auto;max-height:360px;background:#020617;border-radius:14px;padding:14px;border:1px solid var(--line);color:#cbd5e1}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:8px;text-align:left}th{color:#cbd5e1}.pill{display:inline-block;border-radius:999px;padding:3px 9px;background:#020617;border:1px solid var(--line)}
.good{color:var(--good)}.warn{color:var(--warn)}.bad{color:var(--bad)}
canvas{width:100%;height:180px;background:#020617;border:1px solid var(--line);border-radius:14px;margin-top:10px}
@media(max-width:900px){.grid,.two,.three{grid-template-columns:1fr}}
</style>
</head>
<body>
<main>
<h1>Biomni Training Monitor</h1>
<div class="sub">Auto-refresh training progress, GPU state, and periodic benchmark checks. Keep your SSH tunnel open and refresh this page when needed.</div>

<div class="card" style="margin-bottom:14px">
  <div class="label">Training Progress</div>
  <div class="progress"><div id="bar" class="bar"></div></div>
  <div id="progressText" class="sub" style="margin:8px 0 0"></div>
</div>

<div class="grid">
  <div class="card"><div class="label">Status</div><div id="status" class="value">-</div></div>
  <div class="card"><div class="label">Step</div><div id="step" class="value">-</div></div>
  <div class="card"><div class="label">Speed</div><div id="speed" class="value">-</div></div>
  <div class="card"><div class="label">ETA</div><div id="eta" class="value">-</div></div>
</div>

<div class="card" style="margin-top:14px">
  <div class="label">Benchmark Live</div>
  <div class="three" style="margin-top:10px">
    <div><div class="small">Latest Step</div><div id="benchStep" class="value">waiting</div></div>
    <div><div class="small">Accuracy</div><div id="benchAcc" class="value">-</div></div>
    <div><div class="small">Template Issue Rate</div><div id="benchIssue" class="value">-</div></div>
  </div>
  <canvas id="benchChart" width="900" height="210"></canvas>
  <div id="benchNote" class="small" style="margin-top:8px">No benchmark yet. It will appear after the first configured benchmark step.</div>
  <table id="taskTable"><thead><tr><th>Task</th><th>Accuracy</th><th>Correct</th><th>Total</th></tr></thead><tbody></tbody></table>
</div>

<div class="two">
  <div class="card"><div class="label">GPU</div><pre id="gpu">loading...</pre></div>
  <div class="card"><div class="label">Latest Training Log</div><pre id="log">loading...</pre></div>
</div>
<div class="card" style="margin-top:14px"><div class="label">Run</div><pre id="run">loading...</pre></div>
</main>
<script>
function fmtSec(x){if(x===null||x===undefined)return'unknown';x=Math.round(Number(x));let h=Math.floor(x/3600),m=Math.floor((x%3600)/60),s=x%60;return h?`${h}h ${m}m ${s}s`:(m?`${m}m ${s}s`:`${s}s`)}
function pctClass(x){return x>=0.4?'good':(x>=0.2?'warn':'bad')}
function drawChart(events){
  const c=document.getElementById('benchChart'),ctx=c.getContext('2d'),w=c.width,h=c.height;
  ctx.clearRect(0,0,w,h); ctx.fillStyle='#020617'; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle='#334155'; ctx.lineWidth=1;
  for(let i=1;i<5;i++){let y=h*i/5;ctx.beginPath();ctx.moveTo(35,y);ctx.lineTo(w-12,y);ctx.stroke()}
  ctx.fillStyle='#94a3b8';ctx.font='12px sans-serif';ctx.fillText('acc / issue',10,16);
  if(!events||events.length===0){ctx.fillText('waiting for first benchmark...',35,h/2);return}
  const xs=events.map(e=>Number(e.global_step||0)), minX=Math.min(...xs), maxX=Math.max(...xs);
  function x(i){return events.length===1?w/2:35+(w-50)*i/(events.length-1)}
  function y(v){return h-20-(h-42)*Math.max(0,Math.min(1,v))}
  function line(key,color){
    ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();
    events.forEach((e,i)=>{let yy=y(Number(e[key]||0)); if(i===0)ctx.moveTo(x(i),yy);else ctx.lineTo(x(i),yy)});
    ctx.stroke();
    events.forEach((e,i)=>{ctx.fillStyle=color;ctx.beginPath();ctx.arc(x(i),y(Number(e[key]||0)),4,0,Math.PI*2);ctx.fill()});
  }
  line('accuracy','#22c55e'); line('template_issue_rate','#ef4444');
  ctx.fillStyle='#cbd5e1';ctx.fillText(`steps ${minX} - ${maxX}`,35,h-5);
}
async function refresh(){
  const [statusRes, benchRes]=await Promise.all([fetch('/api/status?ts='+Date.now()),fetch('/api/benchmark?ts='+Date.now())]);
  const data=await statusRes.json(); const bench=await benchRes.json();
  const live=data.live||{}, gpu=data.gpu_smi||live.gpu||{}, pct=live.progress_percent||0;
  document.getElementById('bar').style.width=pct+'%';
  document.getElementById('progressText').textContent=`${pct}% · updated ${live.updated_at||'-'}`;
  document.getElementById('status').textContent=live.status||data.status||'-';
  document.getElementById('step').textContent=`${live.global_step||0} / ${live.max_steps||'?'}`;
  document.getElementById('speed').textContent=live.seconds_per_step?`${live.seconds_per_step}s/step`:'-';
  document.getElementById('eta').textContent=fmtSec(live.eta_seconds);
  document.getElementById('gpu').textContent=JSON.stringify(gpu,null,2);
  document.getElementById('log').textContent=JSON.stringify(live.latest_log||{},null,2);
  document.getElementById('run').textContent=JSON.stringify({run_dir:data.run_dir,run_stem:live.run_stem,metadata:data.metadata||live.metadata},null,2);
  const latest=bench.latest||{}, events=bench.events_tail||[];
  if(bench.has_benchmark){
    document.getElementById('benchStep').textContent=latest.global_step||'-';
    document.getElementById('benchAcc').innerHTML=`<span class="${pctClass(Number(latest.accuracy||0))}">${((latest.accuracy||0)*100).toFixed(1)}%</span>`;
    const issue=Number(latest.template_issue_rate||0);
    document.getElementById('benchIssue').innerHTML=`<span class="${issue>0.15?'bad':(issue>0.05?'warn':'good')}">${(issue*100).toFixed(1)}%</span>`;
    document.getElementById('benchNote').textContent=`Composite ${(latest.composite_score||0).toFixed(4)} · examples ${latest.num_examples||0} · updated ${latest.updated_at||'-'}`;
    const tbody=document.querySelector('#taskTable tbody'); tbody.innerHTML='';
    Object.entries(latest.task_summary||{}).forEach(([task,s])=>{
      const tr=document.createElement('tr');
      tr.innerHTML=`<td>${task}</td><td><span class="pill ${pctClass(Number(s.accuracy||0))}">${((s.accuracy||0)*100).toFixed(1)}%</span></td><td>${s.correct}</td><td>${s.total}</td>`;
      tbody.appendChild(tr);
    });
  }else{
    document.getElementById('benchStep').textContent='waiting';
    document.getElementById('benchAcc').textContent='-';
    document.getElementById('benchIssue').textContent='-';
    document.getElementById('benchNote').textContent='No benchmark yet. It will appear after the first configured benchmark step.';
    document.querySelector('#taskTable tbody').innerHTML='';
  }
  drawChart(events);
}
refresh(); setInterval(refresh,5000);
</script>
</body>
</html>"""
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
