#!/usr/bin/env python3
"""Calibration pilot for fixed concurrency versus static phase-aware caps.

The default invocation orchestrates resume-safe randomized blocks.  ``--run-one``
is an internal/publicly useful single-block mode used by both matrix runners.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import threading
import time
import uuid
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/static_phaseaware"
sys.path.insert(0, str(REPO / "src"))
from common import measure, models, thermal  # noqa: E402
from phaseguard.context_validation import (AlwaysBackloggedHNSW, capture_state,
    state_delta, total_rss_bytes)  # noqa: E402
from phaseguard.cpu_task_manager import CPUTaskManager  # noqa: E402
from phaseguard.metrics import append_jsonl, environment_metadata, percentile  # noqa: E402
from phaseguard.phase_monitor import GPUPhase, PhaseMonitor  # noqa: E402
from phaseguard.policies import (FixedWorkerPolicy, PolicyController,
    StaticCapsPolicy)  # noqa: E402
from phaseguard.request_pipeline import GPUTicket, GPUWorker  # noqa: E402
from phaseguard.static_phaseaware import (counter_rate_drift, latency_drift_ratio,
    linear_slope, phase_counter_delta, phase_duration_s, phase_transition_metrics)  # noqa: E402


POLICIES_MINIMAL = (
    ("fixed1", "fixed", 1, 1),
    ("fixed2", "fixed", 2, 2),
    ("phasegate4to0", "phasegate", 4, 0),
    ("phasegate4to1", "phasegate", 4, 1),
    ("phasegate4to2", "phasegate", 4, 2),
    ("fixed4", "fixed", 4, 4),
)
POLICIES_FULL = POLICIES_MINIMAL + (
    ("serialization", "serialization", 0, 0),
    ("fixed3", "fixed", 3, 3),
    ("phasegate4to3", "phasegate", 4, 3),
)


def fan_capable_host() -> tuple[bool, str]:
    raw = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True,
                         text=True, timeout=30).stdout
    model = next((line.split(":", 1)[1].strip() for line in raw.splitlines()
                  if "Model Name:" in line), "unknown")
    return model != "unknown" and "MacBook Air" not in model, model


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def raw_path(stage: str, smoke: bool) -> Path:
    return ROOT / stage / "raw" / ("smoke_runs.jsonl" if smoke else "runs.jsonl")


def request_path(stage: str, smoke: bool) -> Path:
    return ROOT / stage / "raw" / ("smoke_requests.jsonl" if smoke else "requests.jsonl")


def baseline_path(stage: str, smoke: bool) -> Path:
    return ROOT / stage / ("baseline_smoke.json" if smoke else "baseline.json")


def memory_preflight(pids: list[int], seconds: float, minimum_headroom_gb: float) -> dict[str, Any]:
    before = capture_state(pids)
    time.sleep(seconds)
    after = capture_state(pids)
    delta = state_delta(before, after)
    headroom = min(int(delta["headroom_before_bytes"]), int(delta["headroom_after_bytes"]))
    passed = (
        int(delta["pageouts_delta"]) == 0
        and int(delta["swap_used_delta_bytes"] or 0) == 0
        and headroom >= int(minimum_headroom_gb * 1024**3)
    )
    return {"passed": passed, "idle_seconds": seconds,
            "minimum_headroom_gb": minimum_headroom_gb,
            "observed_minimum_headroom_gb": headroom / 1024**3, **delta}


def prefill_sentinel(model: Any, context: int, reps: int) -> float:
    return float(np.median(measure.time_prefill(model, context, reps=reps, warmup=1)))


def await_sentinel(model: Any, context: int, reference_path: Path, tolerance: float,
                   cooldown_s: float, attempts: int, reps: int) -> dict[str, Any]:
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    if reference_path.exists():
        reference = float(json.loads(reference_path.read_text())["median_ms"])
    else:
        reference = prefill_sentinel(model, context, max(3, reps))
        reference_path.write_text(json.dumps(
            {"context": context, "median_ms": reference, "created": datetime.now().isoformat()},
            indent=2) + "\n")
    measured: list[dict[str, Any]] = []
    for attempt in range(attempts):
        value = prefill_sentinel(model, context, reps)
        ratio = value / reference
        passed = abs(ratio - 1.0) <= tolerance
        measured.append({"attempt": attempt, "median_ms": value, "ratio": ratio,
                         "deviation": ratio - 1.0, "passed": passed})
        if passed:
            return {"reference_ms": reference, "passed": True, "attempts": measured}
        if attempt + 1 < attempts:
            time.sleep(cooldown_s)
    return {"reference_ms": reference, "passed": False, "attempts": measured}


def zero_snapshot() -> dict[str, int]:
    return {"permitted_workers": 0, "active_retrievals": 0, "inflight_tasks": 0,
            "retrieval_queue_depth": 0, "paused_inflight_tasks": 0,
            "effective_backlog_tasks": 0, "outstanding_tasks": 0,
            "admitted_queries": 0, "completed_queries": 0}


def execute_gpu_trace(model: Any, monitor: PhaseMonitor, requests: int, context: int,
                      output_tokens: int, prompt_seed: int) -> list[dict[str, Any]]:
    gpu = GPUWorker(model, context, output_tokens, monitor, prompt_seed=prompt_seed)
    gpu.start()
    tickets: list[GPUTicket] = []
    for sequence in range(requests):
        now = time.perf_counter()
        empty = {"submitted": now, "started": now, "ended": now,
                 "queries": 0, "worker_id": -1}
        ticket = GPUTicket(f"static-{sequence:03d}", sequence, now, empty, now)
        tickets.append(ticket)
        gpu.submit(ticket)
    for ticket in tickets:
        if not ticket.done.wait(timeout=3600):
            raise TimeoutError(ticket.request_id)
        if ticket.error:
            raise ticket.error
    gpu.close()
    return [dict(ticket.result) for ticket in tickets if ticket.result is not None]


def policy_caps(args: argparse.Namespace) -> tuple[int, int]:
    if args.policy == "llm-only":
        return 0, 0
    if args.policy == "fixed":
        return args.fixed_workers, args.fixed_workers
    if args.policy == "fixed0":
        return 0, 0
    if args.policy == "serialization":
        return 0, 0
    return args.prefill_cap, args.decode_cap


def policy_name(args: argparse.Namespace) -> str:
    if args.policy == "fixed":
        return f"fixed{args.fixed_workers}"
    if args.policy == "phasegate":
        return f"phasegate{args.prefill_cap}to{args.decode_cap}"
    if args.policy == "fixed0":
        return "fixed0"
    return args.policy


def create_policy(args: argparse.Namespace) -> FixedWorkerPolicy | StaticCapsPolicy:
    if args.policy == "fixed":
        return FixedWorkerPolicy(args.max_workers, args.fixed_workers)
    if args.policy == "serialization":
        return StaticCapsPolicy(args.max_workers, 0, 0)
    if args.policy == "fixed0":
        # Canonical strict-SLO serialization baseline: drain only while the
        # GPU is IDLE; never admit retrieval during an active LLM request.
        return StaticCapsPolicy(args.max_workers, 0, 0)
    if args.policy == "phasegate":
        return StaticCapsPolicy(args.max_workers, args.prefill_cap, args.decode_cap)
    raise ValueError(f"no CPU policy for {args.policy}")


def write_baseline(stage: str, smoke: bool, repeats: int) -> dict[str, Any]:
    rows = [row for row in read_jsonl(raw_path(stage, smoke))
            if row.get("policy_arg") == "llm-only" and row.get("status") == "valid"]
    by_repeat: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_repeat.setdefault(int(row["repeat"]), row)
    selected = [by_repeat[index] for index in sorted(by_repeat)[:repeats]]
    if len(selected) < repeats:
        raise RuntimeError(f"need {repeats} valid isolated baselines, found {len(selected)}")
    payload = {
        "stage": stage,
        "smoke": smoke,
        "valid_repeats": len(selected),
        "p95_tpot_ms": float(np.median([row["p95_tpot_ms"] for row in selected])),
        "p95_ttft_ms": float(np.median([row["p95_ttft_ms"] for row in selected])),
        "tpot_slo_multiplier": 1.10,
        "ttft_slo_multiplier": 1.10,
        "run_keys": [row["run_key"] for row in selected],
    }
    path = baseline_path(stage, smoke)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def run_block(args: argparse.Namespace) -> None:
    if args.max_workers != 4:
        raise SystemExit("the pilot fixes max-workers=4")
    if not 0 <= args.fixed_workers <= args.max_workers:
        raise SystemExit("fixed workers outside [0, 4]")
    if not 0 <= args.prefill_cap <= args.max_workers or not 0 <= args.decode_cap <= args.max_workers:
        raise SystemExit("phase caps outside [0, 4]")
    if args.llm_requests < 2 or args.output_tokens < 8:
        raise SystemExit("need at least two requests and eight output tokens")
    if args.feeders <= args.max_workers or args.queries_per_task < args.chunk:
        raise SystemExit("feeders must exceed workers and queries-per-task must cover chunk")

    label = policy_name(args)
    run_key = (f"static_{args.stage}_{'smoke_' if args.smoke else ''}{label}_"
               f"r{args.repeat:02d}_a{args.attempt:02d}")
    output = raw_path(args.stage, args.smoke)
    if any(row.get("run_key") == run_key for row in read_jsonl(output)):
        print(f"[resume] {run_key}")
        return
    fan_capable, host_model = fan_capable_host()
    if not fan_capable and not args.allow_fanless_pilot:
        raise SystemExit(f"fan-cooled host required unless explicitly allowed; detected {host_model}")
    power_clean = thermal.assert_power(args.allow_battery)
    measure.set_mem_limit_gb(args.mem_limit_gb)
    run_id = uuid.uuid4().hex
    log_dir = ROOT / args.stage / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": run_id, "run_key": run_key, "started": datetime.now().isoformat(),
                "command": " ".join(map(shlex.quote, sys.argv)), "arguments": vars(args),
                "environment": environment_metadata(REPO), "host_model": host_model}
    (log_dir / f"{run_key}_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n")

    preload = memory_preflight([], args.memory_idle_seconds, args.min_headroom_gb)
    if not preload["passed"]:
        append_jsonl(output, {"run_id": run_id, "run_key": run_key, "status": "invalid",
                     "invalid_reason": "preload_memory_preflight", "stage": args.stage,
                     "smoke": args.smoke, "policy": label, "policy_arg": args.policy,
                     "prefill_cap": policy_caps(args)[0], "decode_cap": policy_caps(args)[1],
                     "repeat": args.repeat,
                     "attempt": args.attempt, "preload_memory_preflight": preload})
        return

    model, _ = models.load_model(args.model)
    mx.eval(model.parameters())
    measure.global_warmup(model)
    sentinel_path = (ROOT / args.stage / "logs"
                     / f"sentinel_{'smoke_' if args.smoke else ''}ctx{args.context}_mem{args.mem_limit_gb:g}.json")
    sentinel_before = await_sentinel(model, args.context, sentinel_path,
                                     args.sentinel_tolerance, args.sentinel_cooldown,
                                     args.sentinel_attempts, args.sentinel_reps)
    if not sentinel_before["passed"]:
        append_jsonl(output, {"run_id": run_id, "run_key": run_key, "status": "invalid",
                     "invalid_reason": "sentinel_before", "stage": args.stage,
                     "smoke": args.smoke, "policy": label, "policy_arg": args.policy,
                     "prefill_cap": policy_caps(args)[0], "decode_cap": policy_caps(args)[1],
                     "repeat": args.repeat,
                     "attempt": args.attempt, "sentinel_before": sentinel_before,
                     "preload_memory_preflight": preload})
        return

    manager_context: Any = (
        nullcontext(None) if args.policy == "llm-only"
        else CPUTaskManager(str(args.index), args.max_workers, args.ef_search, args.top_k)
    )
    with manager_context as manager:
        pids = [] if manager is None else manager.worker_pids()
        resident = memory_preflight(pids, args.memory_idle_seconds, args.min_headroom_gb)
        if not resident["passed"]:
            append_jsonl(output, {"run_id": run_id, "run_key": run_key, "status": "invalid",
                         "invalid_reason": "resident_memory_preflight", "stage": args.stage,
                         "smoke": args.smoke, "policy": label, "policy_arg": args.policy,
                         "prefill_cap": policy_caps(args)[0], "decode_cap": policy_caps(args)[1],
                         "repeat": args.repeat,
                         "attempt": args.attempt, "sentinel_before": sentinel_before,
                         "preload_memory_preflight": preload,
                         "resident_memory_preflight": resident})
            return

        monitor = PhaseMonitor()
        controller = None
        load = None
        if manager is not None:
            policy = create_policy(args)
            controller = PolicyController(policy, manager, args.context)
            monitor.set_listener(controller)
            controller(GPUPhase.IDLE, None)
            load = AlwaysBackloggedHNSW(
                manager, args.feeders, args.queries_per_task, args.chunk, args.query_seed
            ).start()
            time.sleep(args.warmup_s)

        q0 = manager.demand_snapshot() if manager is not None else zero_snapshot()
        l0 = load.snapshot() if load is not None else {
            "submitted_tasks": 0.0, "completed_tasks": 0.0,
            "completed_queries": 0.0, "latency_s": 0.0, "latencies_s": []}
        samples: list[dict[str, Any]] = []
        stop = threading.Event()
        sample_counter = [0]
        last_rss = [total_rss_bytes([os.getpid(), *pids])]

        def sample_loop() -> None:
            while not stop.is_set():
                phase, request_id, phase_started = monitor.snapshot()
                snapshot = manager.demand_snapshot() if manager is not None else zero_snapshot()
                if sample_counter[0] % args.rss_sample_stride == 0:
                    last_rss[0] = total_rss_bytes([os.getpid(), *pids])
                samples.append({"timestamp": time.perf_counter(), "phase": phase.value,
                                "phase_started": phase_started, "request_id": request_id,
                                "rss_bytes": last_rss[0], **snapshot})
                sample_counter[0] += 1
                stop.wait(args.sample_ms / 1000.0)

        measure.reset_peak()
        host_before = capture_state(pids)
        sampler = threading.Thread(target=sample_loop, daemon=True,
                                   name="static-phaseaware-sampler")
        sampler.start()
        started = time.perf_counter()
        rows = execute_gpu_trace(model, monitor, args.llm_requests, args.context,
                                 args.output_tokens, args.prompt_seed)
        duration = time.perf_counter() - started
        stop.set()
        sampler.join(timeout=10)
        host_after = capture_state(pids)
        q1 = manager.demand_snapshot() if manager is not None else zero_snapshot()
        l1 = load.snapshot() if load is not None else l0
        if load is not None:
            load.stop()
        events = monitor.events()

    # The post-sentinel must be LLM-only. Exiting the manager context first
    # terminates persistent FAISS workers and any outstanding non-preemptive task.
    sentinel_after = await_sentinel(model, args.context, sentinel_path,
                                    args.sentinel_tolerance, args.sentinel_cooldown,
                                    args.sentinel_attempts, args.sentinel_reps)

    tpot_request = [float(row["p95_tpot_ms"]) for row in rows]
    ttft_request = [float(row["ttft_ms"]) for row in rows]
    p95_tpot = percentile(tpot_request, 95)
    p95_ttft = percentile(ttft_request, 95)
    tpot_drift = latency_drift_ratio(tpot_request)
    ttft_drift = latency_drift_ratio(ttft_request)
    tpot_slope = linear_slope([float(row["completion"]) for row in rows], tpot_request)
    delta = state_delta(host_before, host_after)
    prefill_cap, decode_cap = policy_caps(args)
    prefill_samples = [row for row in samples if row["phase"] == GPUPhase.PREFILL.value]
    decode_samples = [row for row in samples if row["phase"] == GPUPhase.DECODE.value]
    queue_fraction = (float(np.mean([row["retrieval_queue_depth"] > 0 for row in samples]))
                      if manager is not None else 1.0)
    completed_queries = int(q1["completed_queries"]) - int(q0["completed_queries"])
    total_qps = completed_queries / duration if manager is not None else 0.0
    task_latencies = list(l1.get("latencies_s", []))[len(list(l0.get("latencies_s", []))):]
    transition = (phase_transition_metrics(samples, decode_cap) if manager is not None
                  else {"phase_transition_to_cap_ms": 0.0,
                        "phase_transition_to_cap_max_ms": 0.0,
                        "decode_cap_overshoot_fraction": 0.0,
                        "decode_cap_overshoot_worker_mean": 0.0,
                        "decode_cap_overshoot_worker_max": 0.0,
                        "decode_cap_applied_fraction": 1.0})
    qps_drift = counter_rate_drift(samples, "completed_queries")
    prefill_applied = (float(np.mean(
        [int(row["permitted_workers"]) == prefill_cap for row in prefill_samples]
    )) if prefill_samples and manager is not None else 1.0)
    cap_binding = (float(np.mean([
        int(row["outstanding_tasks"]) > int(row["permitted_workers"])
        and int(row["permitted_workers"]) < args.max_workers for row in samples
    ])) if manager is not None else 0.0)
    active_log_present = bool(prefill_samples and decode_samples)
    cap_applied = prefill_applied >= .95 and transition["decode_cap_applied_fraction"] >= .95
    drift_clean = (
        abs(tpot_drift - 1.0) <= args.within_block_drift_tolerance
        and abs(ttft_drift - 1.0) <= args.within_block_drift_tolerance
    )
    memory_clean = int(delta["pageouts_delta"]) == 0 and int(delta["swap_used_delta_bytes"] or 0) == 0
    cpu_disabled_during_llm = prefill_cap == 0 and decode_cap == 0
    enough_progress = (
        args.policy == "llm-only" or cpu_disabled_during_llm
        or completed_queries >= args.min_completed_queries
    )
    decode_admission_zero = (
        manager is None or decode_cap != 0
        or int(phase_counter_delta(samples, "DECODE", "admitted_queries")) == 0
    )
    validity = {
        "queue_saturated": args.policy == "llm-only" or queue_fraction >= .95,
        "memory_clean": memory_clean,
        "sentinel_clean": bool(sentinel_before["passed"] and sentinel_after["passed"]),
        "within_block_drift_clean": drift_clean,
        "active_worker_log_present": active_log_present,
        "policy_cap_applied": args.policy == "llm-only" or cap_applied,
        "steady_duration_sufficient": duration >= args.min_duration_s,
        "completion_count_sufficient": enough_progress,
        "decode_admission_zero": decode_admission_zero,
    }
    invalid_reasons = [key for key, passed in validity.items() if not passed]

    baseline = None
    if args.policy != "llm-only":
        path = args.baseline_file or baseline_path(args.stage, args.smoke)
        if not path.exists():
            raise RuntimeError(f"missing isolated baseline: {path}")
        baseline = json.loads(path.read_text())
    tpot_norm = p95_tpot / float(baseline["p95_tpot_ms"]) if baseline else 1.0
    ttft_norm = p95_ttft / float(baseline["p95_ttft_ms"]) if baseline else 1.0
    tpot_pass = tpot_norm <= 1.10
    ttft_pass = ttft_norm <= 1.10

    row = {
        "run_id": run_id, "run_key": run_key,
        "status": "valid" if all(validity.values()) else "invalid",
        "invalid_reason": ",".join(invalid_reasons) if invalid_reasons else None,
        "stage": args.stage, "smoke": args.smoke, "policy": label,
        "policy_arg": args.policy, "prefill_cap": prefill_cap, "decode_cap": decode_cap,
        "fixed_workers": args.fixed_workers if args.policy == "fixed" else None,
        "repeat": args.repeat, "attempt": args.attempt, "model": args.model,
        "context": args.context, "output_tokens": args.output_tokens,
        "llm_requests": args.llm_requests, "prompt_seed": args.prompt_seed,
        "query_seed": args.query_seed, "duration_s": duration,
        "p95_tpot_ms": p95_tpot, "p95_ttft_ms": p95_ttft,
        "normalized_p95_tpot": tpot_norm, "normalized_p95_ttft": ttft_norm,
        "tpot_slo_pass": tpot_pass, "ttft_slo_pass": ttft_pass,
        "joint_slo_pass": tpot_pass and ttft_pass, "raw_19ms_tpot_pass": p95_tpot <= 19.0,
        "baseline_p95_tpot_ms": None if baseline is None else baseline["p95_tpot_ms"],
        "baseline_p95_ttft_ms": None if baseline is None else baseline["p95_ttft_ms"],
        "total_retrieval_goodput_qps": total_qps,
        "total_completed_retrieval_queries": completed_queries,
        "application_goodput_rps": len(rows) / duration,
        "llm_request_throughput_rps": len(rows) / duration,
        "retrieval_latency_p50_ms": (
            percentile([value * 1e3 for value in task_latencies], 50) if task_latencies else None),
        "retrieval_latency_p95_ms": (
            percentile([value * 1e3 for value in task_latencies], 95) if task_latencies else None),
        "prefill_retrieval_qps": (
            phase_counter_delta(samples, "PREFILL", "completed_queries")
            / phase_duration_s(samples, "PREFILL") if phase_duration_s(samples, "PREFILL") else 0.0),
        "decode_retrieval_qps": (
            phase_counter_delta(samples, "DECODE", "completed_queries")
            / phase_duration_s(samples, "DECODE") if phase_duration_s(samples, "DECODE") else 0.0),
        "admitted_queries_prefill": phase_counter_delta(samples, "PREFILL", "admitted_queries"),
        "admitted_queries_decode": phase_counter_delta(samples, "DECODE", "admitted_queries"),
        "completed_queries_prefill": phase_counter_delta(samples, "PREFILL", "completed_queries"),
        "completed_queries_decode": phase_counter_delta(samples, "DECODE", "completed_queries"),
        "active_retrieval_worker_mean": float(np.mean(
            [row["active_retrievals"] for row in samples])),
        "active_retrieval_worker_p95": percentile(
            [row["active_retrievals"] for row in samples], 95),
        "decode_overlap_fraction": (float(np.mean(
            [row["active_retrievals"] > 0 for row in decode_samples])) if decode_samples else 0.0),
        "prefill_active_retrieval_worker_mean": (float(np.mean(
            [row["active_retrievals"] for row in prefill_samples])) if prefill_samples else 0.0),
        "prefill_active_retrieval_worker_p95": percentile(
            [row["active_retrievals"] for row in prefill_samples], 95) if prefill_samples else 0.0,
        "decode_active_retrieval_worker_mean": (float(np.mean(
            [row["active_retrievals"] for row in decode_samples])) if decode_samples else 0.0),
        "decode_active_retrieval_worker_p95": percentile(
            [row["active_retrievals"] for row in decode_samples], 95) if decode_samples else 0.0,
        "cap_binding_fraction": cap_binding,
        "queue_nonempty_fraction": queue_fraction,
        "beginning_backlog": int(q0["outstanding_tasks"]),
        "ending_backlog": int(q1["outstanding_tasks"]),
        "backlog_slope_tasks_s": (
            (int(q1["outstanding_tasks"]) - int(q0["outstanding_tasks"])) / duration),
        "prefill_cap_applied_fraction": prefill_applied,
        **transition,
        "resident_memory_bytes": int(delta["experiment_rss_after_bytes"]),
        "peak_resident_memory_bytes": max([int(row["rss_bytes"]) for row in samples], default=0),
        "peak_mlx_memory_mb": measure.peak_mb(),
        "tpot_within_block_drift_ratio": tpot_drift,
        "ttft_within_block_drift_ratio": ttft_drift,
        "tpot_within_block_slope_ms_per_s": tpot_slope,
        "retrieval_qps_first_third": qps_drift["first_qps"],
        "retrieval_qps_last_third": qps_drift["last_qps"],
        "retrieval_qps_within_block_ratio": qps_drift["ratio"],
        "retrieval_qps_within_block_slope_qps_per_s": qps_drift["slope_qps_per_s"],
        "sentinel_before": sentinel_before, "sentinel_after": sentinel_after,
        "sentinel_before_deviation": sentinel_before["attempts"][-1]["deviation"],
        "sentinel_after_deviation": sentinel_after["attempts"][-1]["deviation"],
        "sentinel_before_initial_deviation": sentinel_before["attempts"][0]["deviation"],
        "sentinel_after_initial_deviation": sentinel_after["attempts"][0]["deviation"],
        "power_clean": power_clean, "host_model": host_model, "fan_capable": fan_capable,
        "memory_limit_gb": args.mem_limit_gb, "preload_memory_preflight": preload,
        "resident_memory_preflight": resident, "validity": validity,
        "scheduler_overhead_ms": 0.0 if controller is None else controller.overhead_s * 1e3,
        "worker_cap_changes": 0 if controller is None else controller.changes,
        "sample_count": len(samples), **delta,
    }
    append_jsonl(output, row)
    append_jsonl(request_path(args.stage, args.smoke), {
        "run_id": run_id, "run_key": run_key, "requests": rows})
    timeline_dir = ROOT / args.stage / "raw" / ("smoke_timelines" if args.smoke else "timelines")
    timeline_dir.mkdir(parents=True, exist_ok=True)
    (timeline_dir / f"{run_key}.json").write_text(json.dumps({
        "run_id": run_id, "run_key": run_key, "events": events,
        "demand_samples": samples}, separators=(",", ":")) + "\n")
    print(json.dumps(row, indent=2, default=str))


def valid_count(stage: str, smoke: bool, label: str, repeat: int) -> int:
    return sum(row.get("status") == "valid" and row.get("policy") == label
               and int(row.get("repeat", -1)) == repeat
               for row in read_jsonl(raw_path(stage, smoke)))


def measured_count(stage: str, smoke: bool, label: str, repeat: int) -> int:
    """Functional smoke needs one complete metric row, even if the thermal gate fails."""
    return sum(
        row.get("policy") == label
        and int(row.get("repeat", -1)) == repeat
        and "p95_tpot_ms" in row
        for row in read_jsonl(raw_path(stage, smoke))
    )


def max_attempt(stage: str, smoke: bool, label: str, repeat: int) -> int:
    marker = f"_{label}_r{repeat:02d}_"
    return max((
        int(row.get("attempt", 0))
        for row in read_jsonl(raw_path(stage, smoke))
        if (row.get("policy") == label or marker in str(row.get("run_key", "")))
        and int(row.get("repeat", -1)) == repeat
    ), default=0)


def block_command(args: argparse.Namespace, policy: str, prefill: int, decode: int,
                  repeat: int, attempt: int) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), "--run-one",
               "--stage", args.stage, "--policy", policy, "--repeat", str(repeat),
               "--attempt", str(attempt), "--prefill-cap", str(prefill),
               "--decode-cap", str(decode), "--fixed-workers", str(decode),
               "--prompt-seed", str(args.seed + repeat * 10_000),
               "--query-seed", str(args.seed + repeat * 10_000 + 5_000),
               "--context", str(args.context), "--output-tokens", str(args.output_tokens),
               "--llm-requests", str(args.llm_requests), "--sample-ms", str(args.sample_ms),
               "--max-workers", str(args.max_workers), "--feeders", str(args.feeders),
               "--queries-per-task", str(args.queries_per_task), "--chunk", str(args.chunk),
               "--ef-search", str(args.ef_search), "--top-k", str(args.top_k),
               "--index", str(args.index), "--warmup-s", str(args.warmup_s),
               "--mem-limit-gb", str(args.mem_limit_gb),
               "--min-headroom-gb", str(args.min_headroom_gb),
               "--memory-idle-seconds", str(args.memory_idle_seconds),
               "--sentinel-tolerance", str(args.sentinel_tolerance),
               "--sentinel-cooldown", str(args.sentinel_cooldown),
               "--sentinel-attempts", str(args.sentinel_attempts),
               "--sentinel-reps", str(args.sentinel_reps),
               "--within-block-drift-tolerance", str(args.within_block_drift_tolerance),
               "--within-block-qps-drift-tolerance",
               str(args.within_block_qps_drift_tolerance),
               "--min-duration-s", str(args.min_duration_s),
               "--min-completed-queries", str(args.min_completed_queries),
               "--rss-sample-stride", str(args.rss_sample_stride)]
    if args.smoke:
        command.append("--smoke")
    if args.allow_fanless_pilot:
        command.append("--allow-fanless-pilot")
    if args.allow_battery:
        command.append("--allow-battery")
    if args.baseline_file is not None:
        command += ["--baseline-file", str(args.baseline_file)]
    return command


def orchestrate(args: argparse.Namespace) -> None:
    args.stage = "calibration"
    policies = POLICIES_FULL if args.full else POLICIES_MINIMAL
    for repeat in range(args.repeats):
        while valid_count(args.stage, args.smoke, "llm-only", repeat) < 1:
            attempt = max_attempt(args.stage, args.smoke, "llm-only", repeat) + 1
            if attempt > args.max_attempts:
                raise RuntimeError(f"could not obtain baseline repeat {repeat}")
            subprocess.run(block_command(args, "llm-only", 0, 0, repeat, attempt),
                           cwd=REPO, check=True)
    baseline = write_baseline(args.stage, args.smoke, args.repeats)
    blocks = [(repeat, item) for repeat in range(args.repeats) for item in policies]
    random.Random(args.seed).shuffle(blocks)
    for repeat, (label, policy, prefill, decode) in blocks:
        count = measured_count if args.smoke else valid_count
        while count(args.stage, args.smoke, label, repeat) < 1:
            attempt = max_attempt(args.stage, args.smoke, label, repeat) + 1
            if attempt > args.max_attempts:
                break
            subprocess.run(block_command(args, policy, prefill, decode, repeat, attempt),
                           cwd=REPO, check=True)
    log_dir = ROOT / args.stage / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / ("smoke_matrix.json" if args.smoke else "matrix.json")).write_text(
        json.dumps({"seed": args.seed, "repeats": args.repeats, "full": args.full,
                    "baseline": baseline,
                    "randomized_blocks": [{"repeat": repeat, "policy": item[0]}
                                          for repeat, item in blocks]}, indent=2) + "\n")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-one", action="store_true")
    ap.add_argument("--stage", choices=("calibration", "evaluation", "paired_pilot", "decodecap0_pilot"),
                    default="calibration")
    ap.add_argument("--policy", choices=("llm-only", "serialization", "fixed", "fixed0", "phasegate"),
                    default="llm-only")
    ap.add_argument("--fixed-workers", type=int, default=1)
    ap.add_argument("--prefill-cap", type=int, default=4)
    ap.add_argument("--decode-cap", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=8)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--prompt-seed", type=int, default=20260731)
    ap.add_argument("--query-seed", type=int, default=20265731)
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--llm-requests", type=int, default=4)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--feeders", type=int, default=16)
    ap.add_argument("--queries-per-task", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--ef-search", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--index", type=Path,
                    default=Path("experiments/phaseguard/index/hnsw_100k_d384.faiss"))
    ap.add_argument("--sample-ms", type=float, default=5.0)
    ap.add_argument("--rss-sample-stride", type=int, default=20)
    ap.add_argument("--warmup-s", type=float, default=1.0)
    ap.add_argument("--mem-limit-gb", type=float, default=6.0)
    ap.add_argument("--min-headroom-gb", type=float, default=6.5)
    ap.add_argument("--memory-idle-seconds", type=float, default=30.0)
    ap.add_argument("--sentinel-tolerance", type=float, default=0.05)
    ap.add_argument("--sentinel-cooldown", type=float, default=30.0)
    ap.add_argument("--sentinel-attempts", type=int, default=4)
    ap.add_argument("--sentinel-reps", type=int, default=2)
    ap.add_argument("--within-block-drift-tolerance", type=float, default=0.10)
    ap.add_argument("--within-block-qps-drift-tolerance", type=float, default=0.15)
    ap.add_argument("--min-duration-s", type=float, default=5.0)
    ap.add_argument("--min-completed-queries", type=int, default=1000)
    ap.add_argument("--baseline-file", type=Path)
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--allow-fanless-pilot", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap


def main() -> None:
    args = parser().parse_args()
    args.index = args.index if args.index.is_absolute() else REPO / args.index
    if args.baseline_file is not None and not args.baseline_file.is_absolute():
        args.baseline_file = REPO / args.baseline_file
    if args.smoke:
        args.context = min(args.context, 512)
        args.output_tokens = min(args.output_tokens, 16)
        args.llm_requests = min(args.llm_requests, 3)
        args.memory_idle_seconds = min(args.memory_idle_seconds, 1.0)
        args.sentinel_cooldown = min(args.sentinel_cooldown, 1.0)
        args.sentinel_reps = 1
        args.min_duration_s = min(args.min_duration_s, 0.1)
        args.min_completed_queries = min(args.min_completed_queries, 64)
        # Functional smoke still rejects paging; allow the resident model plus
        # four persistent FAISS indexes while retaining the 6.5 GB primary gate.
        args.min_headroom_gb = min(args.min_headroom_gb, 5.5)
        args.within_block_drift_tolerance = max(args.within_block_drift_tolerance, 0.30)
    if args.run_one:
        run_block(args)
    else:
        if args.repeats < 1 or args.max_attempts < args.repeats:
            raise SystemExit("invalid repeat/attempt counts")
        orchestrate(args)


if __name__ == "__main__":
    main()
