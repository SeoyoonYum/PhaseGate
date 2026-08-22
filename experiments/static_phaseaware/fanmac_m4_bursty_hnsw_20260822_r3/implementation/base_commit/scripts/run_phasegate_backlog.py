#!/usr/bin/env python3
"""Run one thermally gated PhaseGate backlog-demand block."""
from __future__ import annotations

import argparse
import json
import random
import shlex
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
from phaseguard.backlog_pipeline import run_staggered_backlog  # noqa: E402
from phaseguard.context_validation import capture_state, state_delta  # noqa: E402
from phaseguard.cpu_task_manager import CPUTaskManager  # noqa: E402
from phaseguard.metrics import append_jsonl, environment_metadata, percentile  # noqa: E402
from phaseguard.phase_monitor import GPUPhase, PhaseMonitor  # noqa: E402
from phaseguard.policies import (DemandAwareCapPolicy, DemandAwareProfilePolicy,
    FixedWorkerPolicy, PolicyController, SerializedPolicy, StaticPhasePolicy,
    UncoordinatedPolicy, load_profile)  # noqa: E402

DEMANDS: dict[str, dict[str, float | int]] = {
    "low": {"requests": 8, "burst": 1, "burst_spacing_s": 0.0, "interarrival_s": 0.002},
    "medium": {"requests": 12, "burst": 2, "burst_spacing_s": 0.02, "interarrival_s": 0.002},
    "high": {"requests": 16, "burst": 4, "burst_spacing_s": 0.02, "interarrival_s": 0.002},
}


def fan_capable_host() -> tuple[bool, str]:
    raw = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True,
                         text=True, timeout=30).stdout
    model = next((line.split(":", 1)[1].strip() for line in raw.splitlines()
                  if "Model Name:" in line), "unknown")
    return model != "unknown" and "MacBook Air" not in model, model


def sentinel(model: Any, context: int, reps: int) -> tuple[float, list[float]]:
    values = [float(value) for value in measure.time_prefill(model, context, reps=reps, warmup=1)]
    return float(np.median(values)), values


def await_sentinel(model: Any, context: int, reps: int, reference_path: Path,
                   tolerance: float, cooldown_s: float, max_attempts: int) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    if reference_path.exists():
        reference = float(json.loads(reference_path.read_text())["median_ms"])
    else:
        time.sleep(cooldown_s)
        reference, values = sentinel(model, context, reps)
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(json.dumps({"context": context, "median_ms": reference,
                                  "values_ms": values}, indent=2) + "\n")
        return {"reference_ms": reference, "passed": True, "attempts": [
            {"median_ms": reference, "ratio": 1.0, "values_ms": values}]}
    for index in range(max_attempts):
        measured, values = sentinel(model, context, reps)
        ratio = measured / reference
        passed = abs(ratio - 1.0) <= tolerance
        attempts.append({"attempt": index, "median_ms": measured, "ratio": ratio,
                         "values_ms": values, "passed": passed})
        if passed:
            return {"reference_ms": reference, "passed": True, "attempts": attempts}
        if index + 1 < max_attempts:
            time.sleep(cooldown_s)
    return {"reference_ms": reference, "passed": False, "attempts": attempts}


def make_policy(args: argparse.Namespace, manager: CPUTaskManager, baseline: float,
                slo_multiplier: float):
    if args.policy == "serialized":
        return SerializedPolicy(args.max_workers)
    if args.policy == "fixed":
        return FixedWorkerPolicy(args.max_workers, args.fixed_workers)
    if args.policy == "uncoordinated":
        return UncoordinatedPolicy(args.max_workers)
    if args.policy == "static0":
        return StaticPhasePolicy(args.max_workers, 0)
    if args.policy == "static":
        return StaticPhasePolicy(args.max_workers, args.decode_workers)
    if args.phaseguard_decode_cap is not None:
        return DemandAwareCapPolicy(args.max_workers, args.phaseguard_decode_cap,
                                    manager.demand_snapshot)
    profile = load_profile(args.profile)
    return DemandAwareProfilePolicy(args.max_workers, profile, args.model, "faiss-hnsw",
                                    baseline, slo_multiplier, manager.demand_snapshot)


def demand_metrics(samples: list[dict[str, Any]], max_workers: int,
                   request_ids: set[str] | None = None) -> dict[str, float]:
    decode = [row for row in samples if row["phase"] == GPUPhase.DECODE.value]
    if request_ids is not None:
        decode = [row for row in decode if row["request_id"] in request_ids]
    if not decode:
        raise RuntimeError("no decode demand samples")
    active = np.asarray([row["active_retrievals"] for row in decode], dtype=float)
    outstanding = np.asarray([row["outstanding_tasks"] for row in decode], dtype=float)
    permits = np.asarray([row["permitted_workers"] for row in decode], dtype=float)
    queues = np.asarray([row["retrieval_queue_depth"] for row in decode], dtype=float)
    paused = np.asarray([row["paused_inflight_tasks"] for row in decode], dtype=float)
    backlog = np.asarray([row["effective_backlog_tasks"] for row in decode], dtype=float)
    permit_binding = outstanding > permits
    scheduler_binding = permit_binding & (permits < max_workers)
    saturated = (outstanding > active) & ((permits == 0) | (active >= permits))
    progress_queries = 0
    progress_seconds = 0.0
    for previous, current in zip(decode, decode[1:]):
        if previous["request_id"] != current["request_id"]:
            continue
        elapsed = float(current["timestamp"]) - float(previous["timestamp"])
        completed = int(current["completed_queries"]) - int(previous["completed_queries"])
        if elapsed > 0 and completed >= 0:
            progress_seconds += elapsed
            progress_queries += completed
    return {
        "decode_active_retrieval_mean": float(active.mean()),
        "decode_active_retrieval_p95": float(np.percentile(active, 95)),
        "decode_retrieval_overlap_fraction": float(np.mean(active > 0)),
        "decode_permit_binding_fraction": float(permit_binding.mean()),
        "decode_scheduler_binding_fraction": float(scheduler_binding.mean()),
        "decode_cap_saturated_fraction": float(saturated.mean()),
        "decode_queue_depth_mean": float(queues.mean()),
        "decode_queue_depth_p95": float(np.percentile(queues, 95)),
        "decode_paused_inflight_mean": float(paused.mean()),
        "decode_paused_inflight_p95": float(np.percentile(paused, 95)),
        "decode_effective_backlog_mean": float(backlog.mean()),
        "decode_effective_backlog_p95": float(np.percentile(backlog, 95)),
        "decode_outstanding_mean": float(outstanding.mean()),
        "decode_retrieval_progress_queries": float(progress_queries),
        "decode_retrieval_progress_s": progress_seconds,
        "decode_retrieval_progress_qps": (
            float(progress_queries / progress_seconds) if progress_seconds > 0 else 0.0),
        "decode_sample_count": float(len(decode)),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("serialized", "fixed", "uncoordinated",
                                        "static0", "static", "phaseguard"), required=True)
    ap.add_argument("--demand", choices=tuple(DEMANDS), required=True)
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--split", choices=("calibration", "evaluation"), default="evaluation")
    ap.add_argument("--fixed-workers", type=int, default=1)
    ap.add_argument("--decode-workers", type=int, default=1,
                    help="decode permit cap for --policy static")
    ap.add_argument("--phaseguard-decode-cap", type=int,
                    help="cap selected on separate calibration runs for PhaseGuard")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--retrieval-queries", type=int, default=4096)
    ap.add_argument("--retrieval-chunk", type=int, default=16)
    ap.add_argument("--ef-search", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--index", type=Path, default=Path("experiments/phaseguard/index/hnsw_100k_d384.faiss"))
    ap.add_argument("--profile", type=Path, default=Path("experiments/phaseguard/processed/profile_primary.csv"))
    ap.add_argument("--slo", type=float, default=1.15)
    ap.add_argument("--target-tpot-ms", type=float,
                    help="absolute p95 TPOT target; overrides --slo for PhaseGuard")
    ap.add_argument("--sample-ms", type=float, default=2.0)
    ap.add_argument("--sentinel-reps", type=int, default=3)
    ap.add_argument("--sentinel-tolerance", type=float, default=0.05)
    ap.add_argument("--sentinel-cooldown", type=float, default=30.0)
    ap.add_argument("--sentinel-attempts", type=int, default=8)
    ap.add_argument("--sentinel-reference", type=Path, default=Path("experiments/phaseguard/backlog_validation/logs/sentinel_reference.json"))
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--mem-limit-gb", type=float, default=12.0)
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--allow-fanless-smoke", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.fixed_workers <= args.max_workers:
        raise SystemExit("fixed workers outside [0, max-workers]")
    if not 0 <= args.decode_workers <= args.max_workers:
        raise SystemExit("decode workers outside [0, max-workers]")
    if args.phaseguard_decode_cap is not None:
        if args.policy != "phaseguard":
            raise SystemExit("--phaseguard-decode-cap is valid only with PhaseGuard")
        if not 0 <= args.phaseguard_decode_cap <= args.max_workers:
            raise SystemExit("PhaseGuard decode cap outside [0, max-workers]")
    if args.sample_ms <= 0 or not 0 < args.sentinel_tolerance < 0.5:
        raise SystemExit("invalid sampling or sentinel configuration")
    if args.slo < 1:
        raise SystemExit("--slo must be >= 1")
    if args.target_tpot_ms is not None:
        if args.policy != "phaseguard":
            raise SystemExit("--target-tpot-ms is valid only with --policy phaseguard")
        if args.target_tpot_ms <= 0:
            raise SystemExit("--target-tpot-ms must be positive")
    args.index = args.index if args.index.is_absolute() else REPO / args.index
    args.profile = args.profile if args.profile.is_absolute() else REPO / args.profile
    args.sentinel_reference = (args.sentinel_reference if args.sentinel_reference.is_absolute()
                               else REPO / args.sentinel_reference)
    fan_capable, host_model = fan_capable_host()
    if not fan_capable and not (args.smoke and args.allow_fanless_smoke):
        raise SystemExit(f"fan-cooled host required; detected {host_model!r}")
    demand = dict(DEMANDS[args.demand])
    if args.smoke:
        args.context, args.output_tokens = 512, 8
        args.retrieval_queries, args.retrieval_chunk = 512, 16
        args.sentinel_reps, args.sentinel_cooldown, args.sentinel_attempts = 1, 1.0, 2
        smoke_demand = {"low": (4, 1), "medium": (6, 2), "high": (8, 3)}
        requests, burst = smoke_demand[args.demand]
        demand = {"requests": requests, "burst": burst,
                  "burst_spacing_s": 0.01, "interarrival_s": 0.002}
        args.sentinel_reference = OUT / "logs/sentinel_reference_smoke.json"
    if args.policy == "fixed":
        policy_label = f"fixed{args.fixed_workers}"
    elif args.policy == "static":
        policy_label = f"static{args.decode_workers}"
    else:
        policy_label = args.policy
    if args.target_tpot_ms is not None:
        policy_label += f"_target{args.target_tpot_ms:g}ms"
    if args.phaseguard_decode_cap is not None:
        policy_label += f"_cap{args.phaseguard_decode_cap}"
    run_key = f"{args.split}_{args.demand}_{policy_label}_r{args.repeat:02d}"
    runs_path = OUT / "raw/runs.jsonl"
    if runs_path.exists() and any(json.loads(line).get("run_key") == run_key
                                  for line in runs_path.read_text().splitlines() if line.strip()):
        print(f"[resume] {run_key}")
        return

    power_ok = thermal.assert_power(args.allow_battery)
    measure.set_mem_limit_gb(args.mem_limit_gb)
    model, _ = models.load_model(args.model)
    mx.eval(model.parameters())
    measure.global_warmup(model)
    sentinel_state = await_sentinel(model, args.context, args.sentinel_reps,
                                    args.sentinel_reference, args.sentinel_tolerance,
                                    args.sentinel_cooldown, args.sentinel_attempts)
    if not sentinel_state["passed"]:
        append_jsonl(OUT / "raw/failed_blocks.jsonl", {"run_key": run_key,
            "status": "sentinel_failed", "host_model": host_model,
            "sentinel": sentinel_state, "command": " ".join(map(shlex.quote, sys.argv))})
        raise SystemExit("baseline sentinel did not return within tolerance")

    run_id = uuid.uuid4().hex
    manifest = {"run_id": run_id, "run_key": run_key, "started": datetime.now().isoformat(),
                "command": " ".join(map(shlex.quote, sys.argv)), "arguments": vars(args),
                "demand_trace": demand, "host_model": host_model, "fan_capable": fan_capable,
                "sentinel": sentinel_state, "environment": environment_metadata(REPO)}
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs" / f"{run_key}_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n")

    with CPUTaskManager(str(args.index), args.max_workers, args.ef_search, args.top_k) as manager:
        profile_rows = load_profile(args.profile)
        baseline_points = [point for point in profile_rows if point.model == args.model
                           and point.phase == "DECODE" and point.workers == 0]
        if not baseline_points:
            raise RuntimeError("profile lacks an uncontended decode baseline")
        baseline_tpot_ms = float(min(baseline_points,
            key=lambda point: abs(point.context - args.context)).p95_ms)
        slo_tpot_ms = (args.target_tpot_ms if args.target_tpot_ms is not None
                       else baseline_tpot_ms * args.slo)
        if slo_tpot_ms < baseline_tpot_ms:
            raise RuntimeError(
                f"TPOT target {slo_tpot_ms:.3f} ms is below the profiled uncontended "
                f"baseline {baseline_tpot_ms:.3f} ms")
        effective_slo_multiplier = slo_tpot_ms / baseline_tpot_ms
        policy = make_policy(args, manager, baseline_tpot_ms, effective_slo_multiplier)
        if isinstance(policy, DemandAwareProfilePolicy):
            profiled_decode_cap = policy.profiled_decode_cap(args.context)
        elif isinstance(policy, DemandAwareCapPolicy):
            profiled_decode_cap = policy.decode_cap
        else:
            profiled_decode_cap = None
        monitor = PhaseMonitor()
        controller = PolicyController(policy, manager, args.context)
        monitor.set_listener(controller)
        controller(GPUPhase.IDLE, None)
        samples: list[dict[str, Any]] = []
        sampler_overhead_s = [0.0]
        sampler_stop = threading.Event()

        def sample_loop() -> None:
            next_refresh = 0.0
            while not sampler_stop.is_set():
                sample_started = time.perf_counter()
                phase, request_id, _ = monitor.snapshot()
                now = time.perf_counter()
                if args.policy == "phaseguard" and now >= next_refresh:
                    controller(phase, request_id)
                    next_refresh = now + 0.01
                samples.append({"timestamp": now, "phase": phase.value,
                                "request_id": request_id, **manager.demand_snapshot()})
                sampler_overhead_s[0] += time.perf_counter() - sample_started
                sampler_stop.wait(args.sample_ms / 1000.0)

        pids = manager.worker_pids()
        host_before = capture_state(pids)
        sampler = threading.Thread(target=sample_loop, name="demand-sampler", daemon=True)
        sampler.start()
        started = time.perf_counter()
        rows = run_staggered_backlog(model, manager, monitor, int(demand["requests"]),
            args.context, args.output_tokens, args.retrieval_queries, args.retrieval_chunk,
            int(demand["burst"]), float(demand["burst_spacing_s"]),
            float(demand["interarrival_s"]), args.seed + args.repeat * 1_000_000)
        duration = time.perf_counter() - started
        sampler_stop.set()
        sampler.join(timeout=5)
        host_after = capture_state(pids)
        events = monitor.events()

    intervals = [float(value) for row in rows for value in row["tpot_intervals_ms"]]
    ttft_gpu = [float(row["ttft_ms"]) for row in rows]
    ttft_app = [(float(row["first_token"]) - float(row["arrival"])) * 1e3 for row in rows]
    e2e = [float(row["end_to_end_ms"]) for row in rows]
    total_queries = sum(int(row["retrieval_queries"]) for row in rows)
    demand_summary = demand_metrics(samples, args.max_workers)
    last_arrival = max(float(row["arrival"]) for row in rows)
    steady_ids = {str(row["request_id"]) for row in rows
                  if float(row["decode_start"]) <= last_arrival + 0.05}
    steady_demand = {"steady_" + key: value for key, value in
                     demand_metrics(samples, args.max_workers, steady_ids).items()}
    steady_intervals = [float(value) for row in rows if str(row["request_id"]) in steady_ids
                        for value in row["tpot_intervals_ms"]]
    host_delta = state_delta(host_before, host_after)
    summary: dict[str, Any] = {"run_id": run_id, "run_key": run_key, "status": "ok",
        "split": args.split, "policy": policy.name, "policy_arg": args.policy,
        "smoke": args.smoke,
        "fixed_workers": args.fixed_workers if args.policy == "fixed" else None,
        "decode_workers": args.decode_workers if args.policy == "static" else None,
        "requested_demand": args.demand, "demand_trace": demand, "repeat": args.repeat,
        "host_model": host_model, "fan_capable": fan_capable, "model": args.model,
        "context": args.context, "output_tokens": args.output_tokens, "requests": len(rows),
        "duration_s": duration, "application_goodput_rps": len(rows) / duration,
        "retrieval_goodput_qps": total_queries / duration,
        "ttft_gpu_p50_ms": percentile(ttft_gpu, 50), "ttft_gpu_p95_ms": percentile(ttft_gpu, 95),
        "ttft_application_p50_ms": percentile(ttft_app, 50),
        "ttft_application_p95_ms": percentile(ttft_app, 95),
        "tpot_p50_ms": percentile(intervals, 50), "tpot_p95_ms": percentile(intervals, 95),
        "tpot_p99_ms": percentile(intervals, 99), "end_to_end_p50_ms": percentile(e2e, 50),
        "end_to_end_p95_ms": percentile(e2e, 95), "slo_ms": slo_tpot_ms,
        "baseline_tpot_ms": baseline_tpot_ms,
        "target_tpot_ms": args.target_tpot_ms,
        "effective_slo_multiplier": effective_slo_multiplier,
        "profiled_decode_worker_cap": profiled_decode_cap,
        "steady_decode_requests": len(steady_ids),
        "steady_tpot_p95_ms": percentile(steady_intervals, 95),
        "steady_token_slo_violation_rate": float(np.mean(
            np.asarray(steady_intervals) > slo_tpot_ms)),
        "token_slo_violation_rate": float(np.mean(np.asarray(intervals) > slo_tpot_ms)),
        "scheduler_overhead_ms": controller.overhead_s * 1e3,
        "demand_sampler_overhead_ms": sampler_overhead_s[0] * 1e3,
        "demand_sampler_overhead_fraction": sampler_overhead_s[0] / duration,
        "worker_cap_changes": controller.changes, "sentinel": sentinel_state,
        "power_clean": power_ok, **demand_summary, **steady_demand, **host_delta}
    append_jsonl(runs_path, summary)
    for row in rows:
        row.update({"run_id": run_id, "run_key": run_key, "split": args.split,
                    "requested_demand": args.demand, "policy": policy.name})
        append_jsonl(OUT / "raw/requests.jsonl", row)
    (OUT / "raw/timelines").mkdir(parents=True, exist_ok=True)
    (OUT / "raw/timelines" / f"{run_key}.json").write_text(json.dumps({
        "run_id": run_id, "run_key": run_key, "events": events, "demand_samples": samples,
        "retrieval": [{key: row[key] for key in ("request_id", "retrieval_start", "retrieval_end")}
                      for row in rows]}, separators=(",", ":")) + "\n")
    print(json.dumps({key: summary[key] for key in ("run_key", "application_goodput_rps",
        "retrieval_goodput_qps", "slo_ms", "profiled_decode_worker_cap", "tpot_p95_ms",
        "steady_tpot_p95_ms", "decode_active_retrieval_mean",
        "steady_decode_active_retrieval_mean",
        "steady_decode_retrieval_progress_qps",
        "decode_retrieval_overlap_fraction", "decode_scheduler_binding_fraction",
        "decode_queue_depth_mean")}, indent=2))


if __name__ == "__main__":
    main()
