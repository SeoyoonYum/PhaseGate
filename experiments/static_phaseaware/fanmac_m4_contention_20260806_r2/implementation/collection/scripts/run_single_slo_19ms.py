#!/usr/bin/env python3
"""Single-SLO feasibility block with an always-backlogged HNSW workload.

This intentionally measures one question only: how much *whole steady-window*
HNSW work can proceed while aggregate p95 TPOT stays at or below 19 ms.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/phaseguard/backlog_validation"
sys.path.insert(0, str(REPO / "src"))
from common import measure, models, thermal  # noqa: E402
from phaseguard.context_validation import (AlwaysBackloggedHNSW, capture_state,
    classify_clean, state_delta)  # noqa: E402
from phaseguard.cpu_task_manager import CPUTaskManager  # noqa: E402
from phaseguard.metrics import append_jsonl, environment_metadata, percentile  # noqa: E402
from phaseguard.phase_monitor import GPUPhase, PhaseMonitor  # noqa: E402
from phaseguard.policies import DemandAwareCapPolicy, PolicyController, UncoordinatedPolicy  # noqa: E402
from phaseguard.request_pipeline import GPUTicket, GPUWorker  # noqa: E402


def fan_capable_host() -> tuple[bool, str]:
    raw = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True,
                         text=True, timeout=30).stdout
    model = next((line.split(":", 1)[1].strip() for line in raw.splitlines()
                  if "Model Name:" in line), "unknown")
    return model != "unknown" and "MacBook Air" not in model, model


def prefill_sentinel(model: Any, context: int, reps: int) -> float:
    values = measure.time_prefill(model, context, reps=reps, warmup=1)
    return float(np.median(np.asarray(values, dtype=float)))


def await_sentinel(model: Any, context: int, reference_path: Path, tolerance: float,
                   cooldown_s: float, attempts: int) -> dict[str, Any]:
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    if reference_path.exists():
        reference = float(json.loads(reference_path.read_text())["median_ms"])
    else:
        reference = prefill_sentinel(model, context, 3)
        reference_path.write_text(json.dumps({"context": context, "median_ms": reference}, indent=2) + "\n")
    rows: list[dict[str, float | bool]] = []
    for attempt in range(attempts):
        value = prefill_sentinel(model, context, 2)
        ratio = value / reference
        passed = abs(ratio - 1.0) <= tolerance
        rows.append({"attempt": attempt, "median_ms": value, "ratio": ratio, "passed": passed})
        if passed:
            return {"reference_ms": reference, "passed": True, "attempts": rows}
        if attempt + 1 < attempts:
            time.sleep(cooldown_s)
    return {"reference_ms": reference, "passed": False, "attempts": rows}


def cpu_percent(pids: list[int]) -> float:
    if not pids:
        return 0.0
    raw = subprocess.run(["ps", "-o", "%cpu=", "-p", ",".join(map(str, pids))],
                         capture_output=True, text=True, timeout=5).stdout
    values = []
    for line in raw.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            pass
    return float(sum(values))


def memory_preflight(pids: list[int], idle_seconds: float,
                     minimum_headroom_gb: float) -> dict[str, Any]:
    """Reject a block before load if macOS is already paging or swapping."""
    before = capture_state(pids)
    time.sleep(idle_seconds)
    after = capture_state(pids)
    delta = state_delta(before, after)
    required = int(minimum_headroom_gb * 1024**3)
    headroom = min(int(delta["headroom_before_bytes"]), int(delta["headroom_after_bytes"]))
    passed = (delta["pageouts_delta"] == 0 and delta["swap_used_delta_bytes"] == 0
              and headroom >= required)
    return {"passed": passed, "idle_seconds": idle_seconds,
            "minimum_headroom_gb": minimum_headroom_gb,
            "observed_minimum_headroom_gb": headroom / 1024**3, **delta}


def phase_qps(samples: list[dict[str, Any]], phase: GPUPhase) -> float:
    queries, seconds = 0, 0.0
    for before, after in zip(samples, samples[1:]):
        if before["phase"] != phase.value or after["phase"] != phase.value:
            continue
        elapsed = float(after["timestamp"]) - float(before["timestamp"])
        progress = int(after["completed_queries"]) - int(before["completed_queries"])
        if elapsed > 0 and progress >= 0:
            seconds += elapsed
            queries += progress
    return float(queries / seconds) if seconds else 0.0


def execute_gpu_trace(model: Any, monitor: PhaseMonitor, requests: int, context: int,
                      output_tokens: int) -> list[dict[str, object]]:
    gpu = GPUWorker(model, context, output_tokens, monitor)
    gpu.start()
    tickets: list[GPUTicket] = []
    for sequence in range(requests):
        now = time.perf_counter()
        empty_retrieval: dict[str, object] = {"submitted": now, "started": now,
                                              "ended": now, "queries": 0, "worker_id": -1}
        ticket = GPUTicket(f"single-slo-{sequence:03d}", sequence, now,
                           empty_retrieval, now)
        tickets.append(ticket)
        gpu.submit(ticket)
    for ticket in tickets:
        if not ticket.done.wait(timeout=3600):
            raise TimeoutError(ticket.request_id)
        if ticket.error:
            raise ticket.error
    gpu.close()
    return [ticket.result for ticket in tickets if ticket.result is not None]


def require_arguments(args: argparse.Namespace) -> None:
    if args.max_workers != 4:
        raise SystemExit("this feasibility check fixes max-workers=4")
    if args.mode == "main" and args.policy == "phaseguard" and not 1 <= args.decode_cap <= 4:
        raise SystemExit("PhaseGuard decode cap must be in [1, 4]")
    if args.output_tokens < 16 or args.llm_requests < 2:
        raise SystemExit("use at least 16 output tokens and two LLM requests")
    if args.feeders <= args.max_workers or args.queries_per_task < args.chunk:
        raise SystemExit("feeders must exceed workers and task size must cover a chunk")
    if (args.target_tpot_ms <= 0 or args.warmup_s < 0 or args.sample_ms <= 0
            or args.mem_limit_gb <= 0 or args.min_headroom_gb <= 0
            or args.memory_idle_seconds < 1):
        raise SystemExit("invalid target, warmup, or sampling value")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("cpu-only", "main"), required=True)
    ap.add_argument("--policy", choices=("phaseguard", "uncoordinated"), default="phaseguard")
    ap.add_argument("--decode-cap", type=int, default=1)
    ap.add_argument("--attempt", type=int, required=True)
    ap.add_argument("--trace-seed", type=int, default=20260730)
    ap.add_argument("--target-tpot-ms", type=float, default=19.0)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--feeders", type=int, default=16)
    ap.add_argument("--queries-per-task", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--ef-search", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--cpu-only-seconds", type=float, default=5.0)
    ap.add_argument("--warmup-s", type=float, default=1.0)
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--llm-requests", type=int, default=4)
    ap.add_argument("--sample-ms", type=float, default=5.0)
    ap.add_argument("--sentinel-tolerance", type=float, default=0.05)
    ap.add_argument("--sentinel-cooldown", type=float, default=30.0)
    ap.add_argument("--sentinel-attempts", type=int, default=4)
    ap.add_argument("--index", type=Path, default=Path("experiments/phaseguard/index/hnsw_100k_d384.faiss"))
    ap.add_argument("--sentinel-reference", type=Path,
                    help="defaults to a memory-cap-specific reference")
    ap.add_argument("--mem-limit-gb", type=float, default=6.0,
                    help="MLX allocator ceiling for this memory-controlled pilot")
    ap.add_argument("--min-headroom-gb", type=float, default=6.5,
                    help="minimum OS headroom before model/worker work starts")
    ap.add_argument("--memory-idle-seconds", type=float, default=30.0,
                    help="idle observation window; pageout and swap growth must stay zero")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--allow-fanless-pilot", action="store_true")
    args = ap.parse_args()
    require_arguments(args)
    args.index = args.index if args.index.is_absolute() else REPO / args.index
    if args.sentinel_reference is None:
        cap_label = f"{args.mem_limit_gb:g}".replace(".", "p")
        args.sentinel_reference = (REPO / "experiments/phaseguard/backlog_validation/logs"
                                   / f"single_slo_sentinel_mem{cap_label}gb.json")
    elif not args.sentinel_reference.is_absolute():
        args.sentinel_reference = REPO / args.sentinel_reference
    fan_capable, host_model = fan_capable_host()
    if not fan_capable and not args.allow_fanless_pilot:
        raise SystemExit(f"fan-cooled host required unless --allow-fanless-pilot; detected {host_model}")
    power_clean = thermal.assert_power(args.allow_battery)
    measure.set_mem_limit_gb(args.mem_limit_gb)
    label = (f"cap{args.decode_cap}" if args.policy == "phaseguard" else "uncoordinated")
    run_key = f"single_slo_19ms_{args.mode}_{label}_a{args.attempt:02d}"
    raw_path = OUT / "raw" / ("single_slo_19ms_cpu_only.jsonl" if args.mode == "cpu-only"
                               else "single_slo_19ms_runs.jsonl")
    if raw_path.exists() and any(json.loads(line).get("run_key") == run_key
                                  for line in raw_path.read_text().splitlines() if line.strip()):
        print(f"[resume] {run_key}")
        return
    run_id = uuid.uuid4().hex
    manifest = {"run_id": run_id, "run_key": run_key, "started": datetime.now().isoformat(),
                "arguments": vars(args), "host_model": host_model, "fan_capable": fan_capable,
                "environment": environment_metadata(REPO)}
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs" / f"{run_key}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    preload_preflight = memory_preflight([], args.memory_idle_seconds, args.min_headroom_gb)
    if not preload_preflight["passed"]:
        row = {"run_id": run_id, "run_key": run_key, "status": "skipped",
               "invalid_reason": "preload_memory_preflight", "mode": args.mode,
               "policy_arg": args.policy, "max_decode_cap": args.decode_cap,
               "memory_limit_gb": args.mem_limit_gb,
               "preload_memory_preflight": preload_preflight,
               "host_model": host_model, "fan_capable": fan_capable}
        append_jsonl(raw_path, row)
        print(json.dumps(row, indent=2, default=str))
        return

    if args.mode == "main":
        model, _ = models.load_model(args.model)
        mx.eval(model.parameters())
        measure.global_warmup(model)
        sentinel = await_sentinel(model, args.context, args.sentinel_reference,
                                  args.sentinel_tolerance, args.sentinel_cooldown,
                                  args.sentinel_attempts)
        if not sentinel["passed"]:
            row = {"run_id": run_id, "run_key": run_key, "status": "invalid",
                   "invalid_reason": "sentinel_failed", "sentinel": sentinel}
            append_jsonl(raw_path, row)
            print(json.dumps(row, indent=2))
            return
    else:
        model, sentinel = None, None

    with CPUTaskManager(str(args.index), args.max_workers, args.ef_search, args.top_k) as manager:
        manager.set_permits(args.decode_cap if args.mode == "cpu-only" else args.max_workers)
        pids = manager.worker_pids()
        resident_preflight = memory_preflight(
            pids, args.memory_idle_seconds, args.min_headroom_gb)
        if not resident_preflight["passed"]:
            row = {"run_id": run_id, "run_key": run_key, "status": "skipped",
                   "invalid_reason": "resident_memory_preflight", "mode": args.mode,
                   "policy_arg": args.policy, "max_decode_cap": args.decode_cap,
                   "memory_limit_gb": args.mem_limit_gb,
                   "preload_memory_preflight": preload_preflight,
                   "resident_memory_preflight": resident_preflight,
                   "host_model": host_model, "fan_capable": fan_capable}
            append_jsonl(raw_path, row)
            print(json.dumps(row, indent=2, default=str))
            return
        load = AlwaysBackloggedHNSW(manager, args.feeders, args.queries_per_task,
                                    args.chunk, args.trace_seed).start()
        time.sleep(args.warmup_s)
        if args.mode == "cpu-only":
            before, q0, l0 = capture_state(pids), manager.demand_snapshot(), load.snapshot()
            samples: list[dict[str, Any]] = []
            deadline = time.perf_counter() + args.cpu_only_seconds
            while time.perf_counter() < deadline:
                samples.append({"timestamp": time.perf_counter(), **manager.demand_snapshot(),
                                "cpu_percent": cpu_percent(pids)})
                time.sleep(args.sample_ms / 1000.0)
            after, q1, l1 = capture_state(pids), manager.demand_snapshot(), load.snapshot()
            load.stop()
            duration = args.cpu_only_seconds
            total_qps = (q1["completed_queries"] - q0["completed_queries"]) / duration
            queue_fraction = float(np.mean([s["retrieval_queue_depth"] > 0 for s in samples]))
            delta = state_delta(before, after)
            valid = (queue_fraction >= .95 and delta["pageouts_delta"] == 0 and
                     delta["swap_used_delta_bytes"] == 0)
            row = {"run_id": run_id, "run_key": run_key, "status": "valid" if valid else "invalid",
                   "mode": args.mode, "workers": args.decode_cap, "duration_s": duration,
                   "retrieval_qps": total_qps, "active_workers_mean": float(np.mean(
                       [s["active_retrievals"] for s in samples])),
                   "active_workers_p95": percentile([s["active_retrievals"] for s in samples], 95),
                   "cpu_percent_mean": float(np.mean([s["cpu_percent"] for s in samples])),
                   "queue_nonempty_fraction": queue_fraction,
                   "beginning_backlog": q0["outstanding_tasks"], "ending_backlog": q1["outstanding_tasks"],
                   "arrivals": l1["submitted_tasks"] - l0["submitted_tasks"],
                   "completions": l1["completed_tasks"] - l0["completed_tasks"],
                   "power_clean": power_clean, "memory_limit_gb": args.mem_limit_gb,
                   "preload_memory_preflight": preload_preflight,
                   "resident_memory_preflight": resident_preflight, **delta}
            append_jsonl(raw_path, row)
            print(json.dumps(row, indent=2))
            return

        assert model is not None
        monitor = PhaseMonitor()
        policy = (DemandAwareCapPolicy(args.max_workers, args.decode_cap, manager.demand_snapshot)
                  if args.policy == "phaseguard" else UncoordinatedPolicy(args.max_workers))
        controller = PolicyController(policy, manager, args.context)
        monitor.set_listener(controller)
        controller(GPUPhase.IDLE, None)
        samples = []
        stop = threading.Event()
        def sample_loop() -> None:
            while not stop.is_set():
                phase, request_id, _ = monitor.snapshot()
                samples.append({"timestamp": time.perf_counter(), "phase": phase.value,
                                "request_id": request_id, "cpu_percent": cpu_percent(pids),
                                **manager.demand_snapshot()})
                stop.wait(args.sample_ms / 1000.0)
        before, q0, l0 = capture_state(pids), manager.demand_snapshot(), load.snapshot()
        sampler = threading.Thread(target=sample_loop, daemon=True, name="single-slo-sampler")
        sampler.start()
        started = time.perf_counter()
        rows = execute_gpu_trace(model, monitor, args.llm_requests, args.context, args.output_tokens)
        duration = time.perf_counter() - started
        stop.set(); sampler.join(timeout=10)
        after, q1, l1 = capture_state(pids), manager.demand_snapshot(), load.snapshot()
        load.stop()

    intervals = [float(value) for row in rows for value in row["tpot_intervals_ms"]]
    request_p95 = [float(row["p95_tpot_ms"]) for row in rows]
    first = intervals[:max(1, len(intervals) // 4)]
    last = intervals[-max(1, len(intervals) // 4):]
    drift = float(np.median(last) / np.median(first))
    spread = float(max(intervals) / np.median(intervals))
    delta = state_delta(before, after)
    clean_flags = classify_clean(delta, {"thermal_drift_ratio": drift, "spread_ratio": 1.0},
                                 2 * 1024**3)
    queue_fraction = float(np.mean([sample["retrieval_queue_depth"] > 0 for sample in samples]))
    decode_samples = [sample for sample in samples if sample["phase"] == GPUPhase.DECODE.value]
    cap_binding = float(np.mean([sample["permitted_workers"] <= args.decode_cap
                                 for sample in decode_samples])) if decode_samples else 0.0
    total_qps = (q1["completed_queries"] - q0["completed_queries"]) / duration
    completed_tasks = l1["completed_tasks"] - l0["completed_tasks"]
    latency = ((l1["latency_s"] - l0["latency_s"]) / completed_tasks * 1e3
               if completed_tasks else float("nan"))
    p95_tpot_ms = percentile(intervals, 95)
    slo_pass = p95_tpot_ms <= args.target_tpot_ms
    valid = (sentinel is not None and sentinel["passed"]
             and queue_fraction >= .95 and clean_flags["clean"])
    invalid_reasons = []
    if queue_fraction < .95: invalid_reasons.append("queue_not_saturated")
    if not clean_flags["pageout_delta_zero"]: invalid_reasons.append("pageout")
    if clean_flags["swap_usage_increased"]: invalid_reasons.append("swap")
    if not clean_flags["thermal_clean"]: invalid_reasons.append("thermal_drift")
    row = {"run_id": run_id, "run_key": run_key, "status": "valid" if valid else "invalid",
           "invalid_reason": ",".join(invalid_reasons) if invalid_reasons else None,
           "mode": args.mode, "policy": policy.name, "policy_arg": args.policy,
           "max_decode_cap": args.decode_cap if args.policy == "phaseguard" else 4,
           "target_tpot_ms": args.target_tpot_ms, "model": args.model, "context": args.context,
           "output_tokens": args.output_tokens, "llm_requests": args.llm_requests,
           "memory_limit_gb": args.mem_limit_gb,
           "preload_memory_preflight": preload_preflight,
           "resident_memory_preflight": resident_preflight,
           "duration_s": duration, "p50_tpot_ms": percentile(intervals, 50),
           "p95_tpot_ms": p95_tpot_ms, "slo_pass": slo_pass,
           "request_p95_tpot_ms": percentile(request_p95, 95),
           "p95_ttft_ms": percentile([float(r["ttft_ms"]) for r in rows], 95),
           "application_goodput_rps": len(rows) / duration, "total_retrieval_qps": total_qps,
           "decode_retrieval_qps": phase_qps(samples, GPUPhase.DECODE),
           "prefill_retrieval_qps": phase_qps(samples, GPUPhase.PREFILL),
           "queue_nonempty_fraction": queue_fraction, "beginning_backlog": q0["outstanding_tasks"],
           "ending_backlog": q1["outstanding_tasks"],
           "backlog_slope_tasks_s": (q1["outstanding_tasks"] - q0["outstanding_tasks"]) / duration,
           "arrivals": l1["submitted_tasks"] - l0["submitted_tasks"],
           "completions": completed_tasks, "active_retrieval_workers_mean": float(np.mean(
               [sample["active_retrievals"] for sample in samples])),
           "active_retrieval_workers_p95": percentile([sample["active_retrievals"] for sample in samples], 95),
           "cap_binding_fraction": cap_binding, "decode_overlap_fraction": float(np.mean(
               [sample["active_retrievals"] > 0 for sample in decode_samples])) if decode_samples else 0.0,
           "retrieval_latency_ms": latency, "cpu_percent_mean": float(np.mean(
               [sample["cpu_percent"] for sample in samples])), "thermal_drift_ratio": drift,
           "thermal_spread_ratio": spread, "sentinel": sentinel, "power_clean": power_clean,
           "host_model": host_model, "fan_capable": fan_capable, "samples": len(samples),
           "worker_cap_changes": controller.changes, "scheduler_overhead_ms": controller.overhead_s * 1e3,
           **clean_flags, **delta}
    append_jsonl(raw_path, row)
    append_jsonl(OUT / "raw/single_slo_19ms_requests.jsonl", {"run_id": run_id, "run_key": run_key,
                 "rows": rows})
    print(json.dumps(row, indent=2, default=str))


if __name__ == "__main__":
    main()
