#!/usr/bin/env python3
"""CPU-only shared-index HNSW scaling and pre-LLM K_hi freeze."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import resource
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from phaseguard.context_validation import capture_state, state_delta  # noqa: E402
from phaseguard.observer import EventBuffer, NativeMemoryMonitor  # noqa: E402
from phaseguard.shared_index_manager import SharedIndexTaskManager  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_percentile(values: list[tuple[int, float]], percentile: float) -> float:
    """Percentile of a piecewise-constant state weighted by interval duration."""
    if not values:
        return 0.0
    ordered = sorted(values)
    threshold = sum(weight for _, weight in ordered) * percentile / 100.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


def measured_events(events: list[dict[str, object]], started: float,
                    ended: float) -> tuple[int, list[float], float, float]:
    """Reconstruct measured completions, latency, and time-weighted concurrency."""
    selected = sorted((event for event in events
                       if started <= float(event["timestamp"]) <= ended),
                      key=lambda event: float(event["timestamp"]))
    starts: dict[str, tuple[float, int]] = {}
    completed = 0
    latencies: list[float] = []
    active = 0
    previous = started
    active_intervals: list[tuple[int, float]] = []
    for event in selected:
        timestamp = float(event["timestamp"])
        active_intervals.append((active, max(0.0, timestamp - previous)))
        kind = str(event["event_type"])
        query_id = str(event.get("query_id", ""))
        if kind == "query_started":
            starts[query_id] = (timestamp, int(event.get("query_count", 0)))
        elif kind == "query_completed":
            count = int(event.get("query_count", 0)); completed += count
            if query_id in starts and count > 0:
                query_start, _ = starts.pop(query_id)
                latencies.extend([(timestamp - query_start) / count] * count)
        if "active_query_count" in event:
            active = int(event["active_query_count"])
        previous = timestamp
    active_intervals.append((active, max(0.0, ended - previous)))
    active_mean = (sum(value * weight for value, weight in active_intervals) /
                   max(ended - started, 1e-12))
    return completed, latencies, active_mean, weighted_percentile(active_intervals, 95)


def one_run(index: Path, cap: int, repeat: int, duration: float,
            warmup: float, seed: int) -> dict[str, object]:
    before = capture_state([])
    events = EventBuffer()
    started_wall = datetime.now(timezone.utc).isoformat()
    with SharedIndexTaskManager(index, cap, ef_search=128, top_k=10,
                                event_sink=events.emit) as manager:
        manager.set_permits(cap)
        sequence = [0]
        sequence_lock = threading.Lock()

        def feeder(slot: int, stop_event: threading.Event, seed_base: int) -> None:
            while not stop_event.is_set():
                with sequence_lock:
                    query_sequence = sequence[0]; sequence[0] += 1
                task = manager.submit(f"cpu-{slot}", 64, 1,
                                      seed_base + query_sequence)
                manager.wait(task, timeout=300)

        warmup_stop = threading.Event()
        warmup_threads = [threading.Thread(
            target=feeder, args=(slot, warmup_stop, seed + 10_000_000), daemon=True)
            for slot in range(cap * 4)]
        for thread in warmup_threads:
            thread.start()
        time.sleep(warmup)
        warmup_stop.set()
        for thread in warmup_threads:
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("CPU scaling warmup feeder did not drain")

        sequence[0] = 0
        stop = threading.Event()
        measure_start = time.perf_counter()
        usage_start = resource.getrusage(resource.RUSAGE_SELF)
        memory_monitor = NativeMemoryMonitor(1.0).start()
        threads = [threading.Thread(target=feeder, args=(slot, stop, seed), daemon=True)
                   for slot in range(cap * 4)]
        for thread in threads:
            thread.start()
        stop.wait(duration)
        measure_end = time.perf_counter()
        memory_monitor.stop()
        usage_end = resource.getrusage(resource.RUSAGE_SELF)
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("CPU scaling measured feeder did not drain")
        runtime_events = events.events()
    after = capture_state([])
    delta = state_delta(before, after)
    elapsed = measure_end - measure_start
    accounting, latencies, active_mean, active_p95 = measured_events(
        runtime_events, measure_start, measure_end)
    cpu_seconds = ((usage_end.ru_utime + usage_end.ru_stime)
                   - (usage_start.ru_utime + usage_start.ru_stime))
    event_submitted = sum(int(event.get("query_count", 0)) for event in runtime_events
                          if event["event_type"] == "query_admitted")
    event_completed = sum(int(event.get("query_count", 0)) for event in runtime_events
                          if event["event_type"] == "query_completed")
    event_accounting_exact = event_submitted == event_completed
    pressure_ok = all(value is None or value >= 10 for value in
                      (delta["memory_free_percent_before"], delta["memory_free_percent_after"]))
    hard_failure = bool((delta["swap_used_delta_bytes"] or 0) > 0 or not pressure_ok
                        or active_p95 != cap or not latencies or not event_accounting_exact
                        or memory_monitor.audit()["subprocess_count"] != 0
                        or memory_monitor.audit()["observed_rate_hz"] > 1.01)
    return {
        "cap": cap, "repeat": repeat, "started_utc": started_wall,
        "ended_utc": datetime.now(timezone.utc).isoformat(), "duration_s": elapsed,
        "seed": seed, "queries_completed": accounting,
        "event_total_admitted_queries": event_submitted,
        "event_total_completed_queries": event_completed,
        "event_accounting_exact": event_accounting_exact, "qps": accounting / elapsed,
        "query_latency_p50_ms": float(np.percentile(latencies, 50) * 1e3),
        "query_latency_p95_ms": float(np.percentile(latencies, 95) * 1e3),
        "cpu_util_percent_mean": cpu_seconds / elapsed * 100.0,
        "rss_bytes_max": max((int(sample["peak_rss_bytes"])
                              for sample in memory_monitor.samples), default=0),
        "physical_footprint_bytes_max": "unavailable",
        "active_concurrency_mean": active_mean,
        "active_concurrency_p95": active_p95,
        "observer_mode": "event", "observer_audit": json.dumps(memory_monitor.audit()),
        "observer_subprocess_count": 0,
        "pageouts_delta": delta["pageouts_delta"],
        "swap_used_delta_bytes": delta["swap_used_delta_bytes"],
        "memory_pressure_ok": pressure_ok,
        "soft_pageout_flag": int(delta["pageouts_delta"]) > 0,
        "status": "invalid" if hard_failure else "valid",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--caps", default="1,2,4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--warmup-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026080501)
    args = parser.parse_args()
    caps = [int(value) for value in args.caps.split(",")]
    if caps != sorted(set(caps)) or any(cap < 1 for cap in caps):
        raise SystemExit("caps must be sorted unique positive integers")
    protocol_path = args.campaign / "CPU_SCALING_PROTOCOL_FREEZE.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["candidate_caps"] != caps or protocol["repeats"] != args.repeats:
            raise SystemExit("CPU scaling arguments differ from frozen protocol")
    else:
        order = [{"cap": cap, "repeat": repeat}
                 for cap in caps for repeat in range(args.repeats)]
        random.Random(args.seed).shuffle(order)
        protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
                    "repository_commit": subprocess.run(
                        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                        capture_output=True, text=True).stdout.strip(),
                    "candidate_caps": caps, "repeats": args.repeats,
                    "duration_s": args.duration_s, "warmup_s": args.warmup_s,
                    "seed": args.seed, "block_order": order,
                    "observer_mode": "event", "subprocess_sampling": False,
                    "query_stream": "shared deterministic seed sequence; identical prefixes"}
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    rows: list[dict[str, object]] = []
    for block in protocol["block_order"]:
        cap, repeat = int(block["cap"]), int(block["repeat"])
        accepted = False
        for attempt in range(1, 3):
            row = one_run(args.index, cap, repeat, args.duration_s,
                          args.warmup_s, args.seed)
            row["attempt"] = attempt
            rows.append(row)
            if row["status"] == "valid":
                accepted = True
                break
        if not accepted:
            write_csv(args.campaign / "cpu_scaling_runs.csv", rows)
            raise RuntimeError(f"cap {cap} repeat {repeat} failed twice")
    write_csv(args.campaign / "cpu_scaling_runs.csv", rows)
    valid = [row for row in rows if row["status"] == "valid"]
    summary = []
    for cap in caps:
        group = [row for row in valid if row["cap"] == cap]
        summary.append({"cap": cap, "valid_repeats": len(group),
                        "median_qps": float(np.median([row["qps"] for row in group])),
                        "median_p50_ms": float(np.median([row["query_latency_p50_ms"] for row in group])),
                        "median_p95_ms": float(np.median([row["query_latency_p95_ms"] for row in group])),
                        "median_cpu_util_percent": float(np.median([row["cpu_util_percent_mean"] for row in group])),
                        "median_rss_bytes": float(np.median([row["rss_bytes_max"] for row in group])),
                        "active_concurrency_p95": float(np.median([row["active_concurrency_p95"] for row in group])),
                        "pageout_soft_flags": sum(bool(row["soft_pageout_flag"]) for row in group),
                        "swap_growth_runs": sum((row["swap_used_delta_bytes"] or 0) > 0 for row in group)})
    write_csv(args.campaign / "cpu_scaling_summary.csv", summary)
    maximum = max(float(row["median_qps"]) for row in summary)
    selected = min(int(row["cap"]) for row in summary
                   if float(row["median_qps"]) >= .9 * maximum)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout.strip()
    freeze = {"rule": "smallest tested cap reaching >=90% of maximum median CPU-only QPS; lower-cap tie-break",
              "candidate_caps": caps, "measured_median_qps": {str(row["cap"]): row["median_qps"] for row in summary},
              "selected_K_hi": selected, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "repository_commit": commit, "index_sha256": sha256(args.index)}
    (args.campaign / "K_HI_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot([row["cap"] for row in summary], [row["median_qps"] for row in summary], marker="o")
    ax.axhline(.9 * maximum, color="gray", linestyle="--", label="90% of max")
    ax.axvline(selected, color="tab:red", linestyle=":", label=f"K_hi={selected}")
    ax.set(xlabel="Worker cap", ylabel="Median HNSW QPS", title="Apple M4 CPU-only HNSW scaling")
    ax.legend(); fig.tight_layout(); fig.savefig(args.campaign / "figure_cpu_scaling.pdf")
    fig.savefig(args.campaign / "figure_cpu_scaling.png", dpi=180); plt.close(fig)
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
