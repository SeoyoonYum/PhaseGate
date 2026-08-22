#!/usr/bin/env python3
"""Run one resumable closed-loop PhaseGuard policy evaluation."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from common import measure, models, thermal  # noqa: E402
from phaseguard.cpu_task_manager import CPUTaskManager  # noqa: E402
from phaseguard.metrics import append_jsonl, environment_metadata, percentile, process_rss_bytes, vm_snapshot  # noqa: E402
from phaseguard.phase_monitor import GPUPhase, PhaseMonitor  # noqa: E402
from phaseguard.policies import (BandwidthThresholdPolicy, PolicyController, ProfilePolicy,
                                 SerializedPolicy, StaticPhasePolicy, UncoordinatedPolicy,
                                 load_profile)  # noqa: E402
from phaseguard.request_pipeline import run_closed_loop  # noqa: E402


def cpu_sampler(pids: list[int], stop: threading.Event, output: list[float]) -> None:
    targets = ",".join(str(pid) for pid in pids)
    while not stop.wait(0.25):
        raw = subprocess.run(["ps", "-o", "%cpu=", "-p", targets], capture_output=True,
                             text=True, timeout=5).stdout
        values = [float(x) for x in raw.split()]
        if values: output.append(sum(values))


def make_policy(args: argparse.Namespace, profile_rows: list[object], baseline: float):
    if args.policy == "uncoordinated": return UncoordinatedPolicy(args.max_workers)
    if args.policy == "serialized": return SerializedPolicy(args.max_workers)
    if args.policy == "static": return StaticPhasePolicy(args.max_workers, args.decode_workers)
    if args.policy == "phaseguard":
        return ProfilePolicy(args.max_workers, profile_rows, args.model, "faiss-hnsw", baseline, args.slo)
    if args.policy == "bandwidth":
        safe = [p.logical_qps for p in profile_rows if p.phase == "DECODE" and
                p.context == args.context and p.p95_ms <= baseline * args.slo]
        threshold = max(safe, default=0.0)
        return BandwidthThresholdPolicy(args.max_workers, profile_rows, threshold)
    raise ValueError(args.policy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="experiments/phaseguard/index/hnsw_100k_d384.faiss")
    ap.add_argument("--profile", default="experiments/phaseguard/processed/profile_primary.csv")
    ap.add_argument("--policy", choices=("uncoordinated", "serialized", "static", "phaseguard", "bandwidth"), required=True)
    ap.add_argument("--decode-workers", type=int, default=1)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--requests-per-client", type=int, default=8)
    ap.add_argument("--retrieval-queries", type=int, default=256)
    ap.add_argument("--retrieval-chunk", type=int, default=1)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--slo", type=float, default=1.15)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--mem-limit-gb", type=float, default=12.0)
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.output_tokens, args.requests_per_client, args.retrieval_queries = 8, 1, 8
    if not 0 <= args.decode_workers <= args.max_workers: raise SystemExit("invalid decode worker count")
    for name in ("index", "profile"):
        value = Path(getattr(args, name)); setattr(args, name, value if value.is_absolute() else REPO / value)
    run_key = (f"{args.tag}_{args.policy}_dw{args.decode_workers}_c{args.concurrency}_"
               f"ctx{args.context}_slo{args.slo:.2f}_r{args.repeat}")
    run_path = REPO / "experiments/phaseguard/raw/runs.jsonl"
    if run_path.exists() and not args.force:
        for line in run_path.read_text().splitlines():
            if json.loads(line).get("run_key") == run_key:
                print(f"[resume] already complete: {run_key}"); return
    power_ok = thermal.assert_power(args.allow_battery)
    profile_rows = load_profile(args.profile)
    candidates = [p for p in profile_rows if p.model == args.model and p.context == args.context and
                  p.phase == "DECODE" and p.workers == 0]
    if not candidates: raise SystemExit("profile lacks matching uncontended baseline")
    baseline = candidates[0].p95_ms
    policy = make_policy(args, profile_rows, baseline)
    measure.set_mem_limit_gb(args.mem_limit_gb)
    model, _ = models.load_model(args.model); mx.eval(model.parameters()); measure.global_warmup(model)
    index_meta = json.loads(args.index.with_suffix(args.index.suffix + ".json").read_text())
    start_vm = vm_snapshot(); run_id = uuid.uuid4().hex
    with CPUTaskManager(str(args.index), args.max_workers, args.ef_search, args.top_k) as manager:
        monitor = PhaseMonitor(); controller = PolicyController(policy, manager, args.context)
        monitor.set_listener(controller); controller(GPUPhase.IDLE, None)
        cpu_values: list[float] = []; sampler_stop = threading.Event()
        sampler = threading.Thread(target=cpu_sampler,
                                   args=([os.getpid()] + manager.worker_pids(), sampler_stop, cpu_values),
                                   daemon=True); sampler.start()
        wall_start = time.perf_counter()
        rows = run_closed_loop(model, manager, monitor, args.concurrency,
                               args.requests_per_client, args.context, args.output_tokens,
                               args.retrieval_queries, args.retrieval_chunk,
                               args.seed + args.repeat * 1_000_000)
        wall_end = time.perf_counter(); sampler_stop.set(); sampler.join(timeout=5)
        events = monitor.events()
    end_vm = vm_snapshot()
    duration = wall_end - wall_start
    intervals = [float(x) for row in rows for x in row["tpot_intervals_ms"]]
    e2e = [float(row["end_to_end_ms"]) for row in rows]
    slo_ms = baseline * args.slo
    summary = {"run_id": run_id, "run_key": run_key, "tag": args.tag, "status": "ok", "policy": policy.name,
               "policy_arg": args.policy, "decode_workers": args.decode_workers,
               "model": args.model, "context": args.context, "output_tokens": args.output_tokens,
               "concurrency": args.concurrency, "requests": len(rows), "repeat": args.repeat,
               "slo_multiplier": args.slo, "slo_ms": slo_ms, "baseline_p95_tpot_ms": baseline,
               "max_workers": args.max_workers, "ef_search": args.ef_search,
               "index_vectors": index_meta["vectors"], "duration_s": duration,
               "request_throughput_s": len(rows) / duration, "tokens_s": len(rows) * args.output_tokens / duration,
               "retrieval_qps": sum(int(row["retrieval_queries"]) for row in rows) / duration,
               "p50_request_ms": percentile(e2e, 50), "p95_request_ms": percentile(e2e, 95),
               "p99_request_ms": percentile(e2e, 99), "p50_tpot_ms": percentile(intervals, 50),
               "p95_tpot_ms": percentile(intervals, 95), "p99_tpot_ms": percentile(intervals, 99),
               "token_slo_violation_rate": sum(x > slo_ms for x in intervals) / len(intervals),
               "request_slo_violation_rate": sum(float(r["p95_tpot_ms"]) > slo_ms for r in rows) / len(rows),
               "cpu_util_pct": float(np.mean(cpu_values)) if cpu_values else None,
               "rss_bytes": process_rss_bytes(),
               "headroom_bytes": end_vm.get("headroom_bytes", 0),
               "pageouts_delta": end_vm.get("pageouts", 0) - start_vm.get("pageouts", 0),
               "thermal_spread": max(intervals) / float(np.median(intervals)),
               "power_contaminated": not power_ok, "scheduler_overhead_ms": controller.overhead_s * 1e3,
               "worker_count_changes": controller.changes,
               "contaminated": (not power_ok) or (end_vm.get("pageouts", 0) > start_vm.get("pageouts", 0)),
               "environment": environment_metadata(REPO)}
    request_path = REPO / "experiments/phaseguard/raw/requests.jsonl"
    for row in rows:
        row.update({"run_id": run_id, "run_key": run_key, "tag": args.tag, "policy": policy.name,
                    "model": args.model, "context": args.context, "concurrency": args.concurrency,
                    "slo_ms": slo_ms, "repeat": args.repeat})
        append_jsonl(request_path, row)
    append_jsonl(run_path, summary)
    timeline = {"run_id": run_id, "run_key": run_key, "events": events,
                "retrieval": [{k: r[k] for k in ("request_id", "retrieval_start", "retrieval_end", "retrieval_worker")}
                              for r in rows]}
    timeline_path = REPO / "experiments/phaseguard/raw/timelines" / f"{run_key}.json"
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_path.write_text(json.dumps(timeline, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("run_key", "requests", "request_throughput_s",
          "p95_tpot_ms", "token_slo_violation_rate", "p95_request_ms", "contaminated")}, indent=2))


if __name__ == "__main__": main()
