#!/usr/bin/env python3
"""Focused uncoordinated pipeline validation for one context/workload."""
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/phaseguard/context_validation"
sys.path.insert(0, str(REPO / "src"))
from common import measure, models, thermal  # noqa: E402
from phaseguard.context_validation import (SyntheticRandomLoad, capture_state,
        classify_clean, distribution_metrics, state_delta)  # noqa: E402
from phaseguard.cpu_task_manager import CPUTaskManager  # noqa: E402
from phaseguard.metrics import append_jsonl, environment_metadata, percentile  # noqa: E402
from phaseguard.phase_monitor import PhaseMonitor  # noqa: E402
from phaseguard.request_pipeline import run_closed_loop  # noqa: E402


class NoopManager:
    def submit(self, request_id: str, queries: int, chunk: int, seed: int) -> str:
        return request_id

    def wait(self, task_id: str) -> dict[str, object]:
        now = time.perf_counter()
        return {"task_id": task_id, "request_id": task_id, "worker_id": -1,
                "submitted": now, "started": now, "ended": now,
                "queries": 0, "chunks": 0, "checksum": 0.0, "error": None}


def main() -> None:
    try: sys.stdout.reconfigure(line_buffering=True)
    except AttributeError: pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", type=int, required=True)
    ap.add_argument("--context", type=int, choices=(2048, 4096), required=True)
    ap.add_argument("--workload", choices=("none", "hnsw4", "random"), required=True)
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--retrieval-queries", type=int, default=4096)
    ap.add_argument("--index", default="experiments/phaseguard/index/hnsw_100k_d384.faiss")
    ap.add_argument("--ef-search", type=int, default=128)
    ap.add_argument("--idle-seconds", type=float, default=30)
    ap.add_argument("--mem-limit-gb", type=float, default=12)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke: args.requests, args.output_tokens, args.idle_seconds = 2, 8, 1
    if args.requests < 2 or args.concurrency < 1 or args.idle_seconds <= 0:
        raise SystemExit("invalid request configuration")
    run_key = f"p{args.attempt:02d}_ctx{args.context}_{args.workload}"
    runs_path = OUT / "raw/pipeline_runs.jsonl"
    if runs_path.exists() and any(json.loads(x).get("run_key") == run_key for x in runs_path.read_text().splitlines()):
        print(f"[resume] {run_key}"); return
    index = Path(args.index); index = index if index.is_absolute() else REPO / index
    power_ok = thermal.assert_power(args.allow_battery); measure.set_mem_limit_gb(args.mem_limit_gb)
    model, _ = models.load_model(args.model); mx.eval(model.parameters()); measure.global_warmup(model)
    attempt_id = uuid.uuid4().hex
    manifest = {"attempt_id": attempt_id, "run_key": run_key,
                "command": " ".join(shlex.quote(x) for x in sys.argv),
                "started": datetime.now().isoformat(timespec="seconds"),
                "arguments": vars(args), "environment": environment_metadata(REPO), "power_ok": power_ok}
    (OUT / "logs" / f"{run_key}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    with CPUTaskManager(str(index), 4, args.ef_search, 10) as manager:
        pids = manager.worker_pids(); manager.set_permits(4 if args.workload == "hnsw4" else 0)
        before_idle = capture_state(pids); time.sleep(args.idle_seconds); after_idle = capture_state(pids)
        idle_delta = state_delta(before_idle, after_idle)
        synthetic: SyntheticRandomLoad | None = None
        if args.workload == "random": synthetic = SyntheticRandomLoad().start()
        state_before = capture_state(pids); start = time.perf_counter()
        pipeline_manager: Any = manager if args.workload == "hnsw4" else NoopManager()
        rows = run_closed_loop(model, pipeline_manager, PhaseMonitor(), args.concurrency,
                               max(1, math.ceil(args.requests / args.concurrency)), args.context,
                               args.output_tokens, args.retrieval_queries, 1,
                               args.seed + args.attempt * 1_000_000)
        duration = time.perf_counter() - start; state_after = capture_state(pids)
        if synthetic: synthetic.stop()
        time.sleep(args.idle_seconds)
        after_post_idle = capture_state(pids)
        post_idle_delta = state_delta(state_after, after_post_idle)
    prefills = [float(row["prefill_ms"]) for row in rows]
    queues = [float(row["gpu_queue_ms"]) for row in rows]
    tpot = [float(value) for row in rows for value in row["tpot_intervals_ms"]]
    e2e = [float(row["end_to_end_ms"]) for row in rows]
    metrics = distribution_metrics(prefills); delta = state_delta(state_before, state_after)
    flags = classify_clean(delta, metrics, 4 * 1024**3)
    summary: dict[str, Any] = {"attempt_id": attempt_id, "run_key": run_key,
        "attempt": args.attempt, "context": args.context, "workload": args.workload,
        "policy": "uncoordinated", "requests": len(rows), "concurrency": args.concurrency,
        "output_tokens": args.output_tokens, "duration_s": duration,
        "request_throughput_s": len(rows) / duration, "prefill_median_ms": metrics["median_ms"],
        "prefill_p95_ms": metrics["p95_ms"], "prefill_p99_ms": metrics["p99_ms"],
        "gpu_queue_median_ms": percentile(queues, 50), "gpu_queue_p95_ms": percentile(queues, 95),
        "queue_inclusive_median_ms": percentile([p + q for p, q in zip(prefills, queues)], 50),
        "queue_inclusive_p95_ms": percentile([p + q for p, q in zip(prefills, queues)], 95),
        "tpot_median_ms": percentile(tpot, 50), "tpot_p95_ms": percentile(tpot, 95),
        "end_to_end_median_ms": percentile(e2e, 50), "end_to_end_p95_ms": percentile(e2e, 95),
        "idle_before_pageout_rate_s": idle_delta["pageouts_delta"] / args.idle_seconds,
        "idle_after_pageout_rate_s": post_idle_delta["pageouts_delta"] / args.idle_seconds,
        "power_clean": power_ok, "temperature_c": None, "gpu_frequency_mhz": None,
        **metrics, **delta, **flags, "status": "ok"}
    append_jsonl(runs_path, summary)
    for row in rows:
        row.update({"attempt_id": attempt_id, "run_key": run_key, "context": args.context,
                    "workload": args.workload, "policy": "uncoordinated"})
        append_jsonl(OUT / "raw/pipeline_requests.jsonl", row)
    print(json.dumps({k: summary[k] for k in ("run_key", "requests", "prefill_median_ms",
          "prefill_p95_ms", "gpu_queue_p95_ms", "tpot_p95_ms", "clean")}, indent=2))


if __name__ == "__main__": main()
