#!/usr/bin/env python3
"""Run one randomized, resumable isolated context-validation attempt."""
from __future__ import annotations

import argparse
import json
import random
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
from common import cpuload, measure, models, thermal  # noqa: E402
from phaseguard.context_validation import (HNSWLoad, SyntheticRandomLoad, capture_state,
        classify_clean, decode_iterations, distribution_metrics, prefill_iterations, state_delta)  # noqa: E402
from phaseguard.cpu_task_manager import CPUTaskManager  # noqa: E402
from phaseguard.metrics import append_jsonl, environment_metadata  # noqa: E402


def completed_keys(path: Path) -> set[str]:
    if not path.exists(): return set()
    return {json.loads(line)["run_key"] for line in path.read_text().splitlines()
            if line.strip() and json.loads(line).get("status") == "ok"}


def idle_window(seconds: float, pids: list[int]) -> dict[str, Any]:
    before = capture_state(pids); time.sleep(seconds); after = capture_state(pids)
    delta = state_delta(before, after)
    delta.update({"seconds": seconds, "pageout_rate_s": delta["pageouts_delta"] / seconds,
                  "before": asdict(before), "after": asdict(after)})
    return delta


def blocked_order(contexts: list[int], workloads: list[str], phases: list[str], seed: int):
    rng = random.Random(seed); blocks = [(w, p) for w in workloads for p in phases]; rng.shuffle(blocks)
    result = []
    for workload, phase in blocks:
        pair = list(contexts); rng.shuffle(pair)
        result.extend((context, workload, phase) for context in pair)
    return result


def main() -> None:
    try: sys.stdout.reconfigure(line_buffering=True)
    except AttributeError: pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", type=int, required=True)
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--contexts", default="2048,4096")
    ap.add_argument("--workloads", default="none,hnsw1,hnsw4,random")
    ap.add_argument("--phases", default="prefill,decode")
    ap.add_argument("--prefill-warmups", type=int, default=3)
    ap.add_argument("--prefill-iterations", type=int, default=20)
    ap.add_argument("--decode-warmups", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--idle-seconds", type=float, default=30.0)
    ap.add_argument("--cooldown", type=float, default=10.0)
    ap.add_argument("--min-headroom-gb", type=float, default=4.0)
    ap.add_argument("--index", default="experiments/phaseguard/index/hnsw_100k_d384.faiss")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--ef-search", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--queries-per-task", type=int, default=32)
    ap.add_argument("--mem-limit-gb", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.attempt < 0 or args.prefill_warmups < 0 or args.prefill_iterations < 2:
        raise SystemExit("invalid attempt/warmup/iteration arguments")
    contexts = [int(value) for value in args.contexts.split(",")]
    workloads = [value.strip() for value in args.workloads.split(",") if value.strip()]
    phases = [value.strip() for value in args.phases.split(",") if value.strip()]
    if any(c < 128 for c in contexts) or not set(workloads) <= {"none", "hnsw1", "hnsw4", "random"}:
        raise SystemExit("invalid contexts/workloads")
    if not set(phases) <= {"prefill", "decode"}: raise SystemExit("invalid phases")
    if args.smoke:
        contexts, workloads = [512], ["none", "hnsw1"]
        args.prefill_warmups, args.prefill_iterations = 1, 2
        args.decode_warmups, args.decode_tokens = 1, 8
        args.idle_seconds, args.cooldown = 1.0, 0.0
    index = Path(args.index); index = index if index.is_absolute() else REPO / index
    if not index.exists(): raise SystemExit(f"missing index: {index}")
    power_ok = thermal.assert_power(args.allow_battery); measure.set_mem_limit_gb(args.mem_limit_gb)
    cpuload.ensure_built()
    runs_path, iterations_path = OUT / "raw/isolated_runs.jsonl", OUT / "raw/iterations.jsonl"
    complete = completed_keys(runs_path)
    attempt_id = uuid.uuid4().hex
    command = " ".join(shlex.quote(value) for value in sys.argv)
    manifest = {"attempt_id": attempt_id, "attempt": args.attempt, "command": command,
                "started": datetime.now().isoformat(timespec="seconds"), "arguments": vars(args),
                "contexts": contexts, "workloads": workloads, "phases": phases,
                "environment": environment_metadata(REPO), "power_ok": power_ok,
                "temperature_clock": "unavailable without successful powermetrics access"}
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs" / f"attempt_{args.attempt:02d}_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str, sort_keys=True) + "\n")
    print(f"[load] {models.resolve_model_id(args.model)}")
    model, tokenizer = models.load_model(args.model); mx.eval(model.parameters()); measure.global_warmup(model)
    del tokenizer
    with CPUTaskManager(str(index), args.max_workers, args.ef_search, args.top_k) as manager:
        manager.set_permits(0); pids = manager.worker_pids()
        print(f"[idle-before] {args.idle_seconds}s")
        idle_before = idle_window(args.idle_seconds, pids)
        order = blocked_order(contexts, workloads, phases, args.seed + args.attempt)
        for order_index, (context, workload, phase) in enumerate(order):
            run_key = f"a{args.attempt:02d}_ctx{context}_{workload}_{phase}"
            if run_key in complete:
                print(f"[resume] {run_key}"); continue
            load: HNSWLoad | SyntheticRandomLoad | None = None
            workload_start = time.perf_counter(); status, error = "ok", None
            try:
                if workload.startswith("hnsw"):
                    load = HNSWLoad(manager, int(workload[-1]), args.queries_per_task,
                                    args.seed + args.attempt * 1_000_000 + order_index).start()
                elif workload == "random":
                    manager.set_permits(0); load = SyntheticRandomLoad().start()
                else: manager.set_permits(0)
                state_before = capture_state(pids); q0 = load.snapshot() if isinstance(load, HNSWLoad) else 0
                iter_base = {"attempt_id": attempt_id, "attempt": args.attempt, "run_key": run_key,
                             "order": order_index, "context": context, "workload": workload,
                             "phase": phase, "prompt_token_count": context,
                             "prompt_tensor_shape": [1, context], "model": args.model}
                def record(index_: int, latency: float) -> None:
                    append_jsonl(iterations_path, {**iter_base, "iteration": index_,
                                                  "latency_ms": latency, "timestamp": time.time()})
                if phase == "prefill":
                    timings = prefill_iterations(model, context, args.prefill_warmups,
                                                 args.prefill_iterations, record)
                else:
                    timings = decode_iterations(model, context, args.decode_warmups,
                                                args.decode_tokens, record)
                q1 = load.snapshot() if isinstance(load, HNSWLoad) else 0
                state_after = capture_state(pids); elapsed = time.perf_counter() - workload_start
                if isinstance(load, SyntheticRandomLoad):
                    load.stop(); throughput = load.throughput(); load = None
                else:
                    throughput = (q1 - q0) / elapsed if isinstance(load, HNSWLoad) else 0.0
                metrics = distribution_metrics(timings); delta = state_delta(state_before, state_after)
                flags = classify_clean(delta, metrics, int(args.min_headroom_gb * 1024**3))
            except Exception as exc:
                timings, metrics, delta, flags, throughput = [], {}, {}, {}, 0.0
                status, error = "failed", repr(exc)
            finally:
                if load is not None: load.stop()
                manager.set_permits(0)
            row: dict[str, Any] = {"attempt_id": attempt_id, "attempt": args.attempt,
                   "run_key": run_key, "order": order_index, "start_time": state_before.timestamp if status == "ok" else None,
                   "end_time": state_after.timestamp if status == "ok" else None,
                   "context": context, "workload": workload, "phase": phase, "model": args.model,
                   "model_id": models.resolve_model_id(args.model), "quantization": "4bit",
                   "prompt_token_count": context, "prompt_tensor_shape": [1, context],
                   "generated_tokens": args.decode_tokens if phase == "decode" else 0,
                   "warmups": args.prefill_warmups if phase == "prefill" else args.decode_warmups,
                   "measured_iterations": len(timings), "timings_ms": timings,
                   "workload_throughput": throughput,
                   "workload_throughput_unit": "queries/s" if workload.startswith("hnsw") else "logical GB/s",
                   "idle_before_pageout_rate_s": idle_before["pageout_rate_s"],
                   "power_clean": power_ok, "temperature_c": None, "gpu_frequency_mhz": None,
                   **metrics, **delta, **flags, "status": status, "error": error}
            append_jsonl(runs_path, row)
            metric_text = f"median={metrics.get('median_ms', float('nan')):.2f} p95={metrics.get('p95_ms', float('nan')):.2f}"
            print(f"[{order_index+1}/{len(order)}] {run_key} {metric_text} clean={flags.get('clean', False)}")
            time.sleep(args.cooldown)
        print(f"[idle-after] {args.idle_seconds}s")
        idle_after = idle_window(args.idle_seconds, pids)
    idle_record = {"attempt_id": attempt_id, "attempt": args.attempt,
                   "idle_before": idle_before, "idle_after": idle_after}
    (OUT / "logs" / f"attempt_{args.attempt:02d}_idle.json").write_text(
        json.dumps(idle_record, indent=2, sort_keys=True) + "\n")
    print(f"[out] {runs_path}\n[out] {iterations_path}")


if __name__ == "__main__": main()
