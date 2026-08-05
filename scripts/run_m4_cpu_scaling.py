#!/usr/bin/env python3
"""CPU-only shared-index HNSW scaling and pre-LLM K_hi freeze."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
from phaseguard.shared_index_manager import SharedIndexTaskManager  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_cpu_percent() -> float:
    raw = subprocess.run(["ps", "-o", "%cpu=", "-p", str(os.getpid())],
                         capture_output=True, text=True, timeout=5).stdout.strip()
    return float(raw or 0.0)


def one_run(index: Path, cap: int, repeat: int, duration: float,
            warmup: float, seed: int) -> dict[str, object]:
    before = capture_state([])
    latencies: list[float] = []
    active: list[int] = []
    cpu: list[float] = []
    rss: list[int] = []
    stop = threading.Event()
    started_wall = datetime.now(timezone.utc).isoformat()
    with SharedIndexTaskManager(index, cap, ef_search=128, top_k=10) as manager:
        manager.set_permits(cap)
        counters = [0] * (cap * 4)

        def feeder(slot: int) -> None:
            sequence = 0
            while not stop.is_set():
                task = manager.submit(f"cpu-{slot}", 64, 1,
                                      seed + slot * 1_000_000 + sequence)
                result = manager.wait(task, timeout=300)
                if time.perf_counter() >= measure_start:
                    latencies.extend(float(x) for x in result["query_latencies_s"])
                    counters[slot] += int(result["queries"])
                sequence += 1

        measure_start = float("inf")
        threads = [threading.Thread(target=feeder, args=(slot,), daemon=True)
                   for slot in range(cap * 4)]
        for thread in threads:
            thread.start()
        time.sleep(warmup)
        measure_start = time.perf_counter()
        q0 = manager.demand_snapshot()
        while time.perf_counter() - measure_start < duration:
            snap = manager.demand_snapshot()
            active.append(int(snap["active_retrievals"]))
            cpu.append(process_cpu_percent())
            rss.append(int(capture_state([]).experiment_rss_bytes))
            time.sleep(.1)
        q1 = manager.demand_snapshot()
        ended = time.perf_counter()
        stop.set()
        manager.set_permits(cap)
        for thread in threads:
            thread.join(timeout=5)
        accounting = int(q1["completed_queries"]) - int(q0["completed_queries"])
    after = capture_state([])
    delta = state_delta(before, after)
    elapsed = ended - measure_start
    measured = int(sum(counters))
    # Boundary-straddling tasks can differ from the counter snapshot by at most
    # one task per feeder; both counts are retained for audit.
    active_p95 = float(np.percentile(active, 95))
    pressure_ok = all(value is None or value >= 10 for value in
                      (delta["memory_free_percent_before"], delta["memory_free_percent_after"]))
    hard_failure = bool((delta["swap_used_delta_bytes"] or 0) > 0 or not pressure_ok
                        or active_p95 != cap or not latencies)
    return {
        "cap": cap, "repeat": repeat, "started_utc": started_wall,
        "ended_utc": datetime.now(timezone.utc).isoformat(), "duration_s": elapsed,
        "seed": seed, "queries_completed": accounting,
        "feeder_counted_queries": measured, "qps": accounting / elapsed,
        "query_latency_p50_ms": float(np.percentile(latencies, 50) * 1e3),
        "query_latency_p95_ms": float(np.percentile(latencies, 95) * 1e3),
        "cpu_util_percent_mean": float(np.mean(cpu)),
        "rss_bytes_max": max(rss), "physical_footprint_bytes_max": "unavailable",
        "active_concurrency_mean": float(np.mean(active)),
        "active_concurrency_p95": active_p95,
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
    parser.add_argument("--caps", default="1,2")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--warmup-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026080501)
    args = parser.parse_args()
    caps = [int(value) for value in args.caps.split(",")]
    if caps != sorted(set(caps)) or any(cap < 1 for cap in caps):
        raise SystemExit("caps must be sorted unique positive integers")
    rows: list[dict[str, object]] = []
    for cap in caps:
        for repeat in range(args.repeats):
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
