#!/usr/bin/env python3
"""Run and audit one frozen r5 baseline set without changing its definitions."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BLOCK = REPO / "scripts/run_static_phaseaware_pilot.py"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.strip()


def run_repeat(campaign: Path, freeze: dict[str, Any], set_name: str,
               trace: dict[str, int], log: Any) -> None:
    stage = f"r5_baseline_{set_name}"
    runs_path = campaign / stage / "raw/runs.jsonl"
    repeat = int(trace["repeat"])
    existing = [row for row in read_jsonl(runs_path) if int(row.get("repeat", -1)) == repeat]
    if any(row.get("status") == "valid" for row in existing):
        return
    attempt = max([int(row.get("attempt", 0)) for row in existing], default=0) + 1
    if attempt > 2:
        raise RuntimeError(f"hard-invalid retry exhausted: set {set_name} repeat {repeat}")
    artifacts, workload = freeze["artifacts"], freeze["workload"]
    command = [
        sys.executable, str(BLOCK), "--run-one", "--stage", stage,
        "--policy", "llm-only", "--observer-mode", "event",
        "--repeat", str(repeat), "--attempt", str(attempt),
        "--prompt-seed", str(trace["prompt_seed"]), "--query-seed", str(trace["query_seed"]),
        "--model", artifacts["model_local_path"], "--index", artifacts["index_path"],
        "--context", str(workload["context_tokens"]),
        "--output-tokens", str(workload["output_tokens"]),
        "--llm-requests", str(workload["measured_requests"]),
        "--max-workers", "4", "--feeders", "8", "--queries-per-task", "4096",
        "--chunk", "16", "--ef-search", "128", "--top-k", "10",
        "--memory-sample-interval-s", "1", "--warmup-s", "2",
        "--mem-limit-gb", "5.5", "--min-headroom-gb", "3.0",
        "--memory-idle-seconds", "2", "--sentinel-tolerance", "0.03",
        "--sentinel-cooldown", "2", "--sentinel-attempts", "2",
        "--sentinel-reps", "2", "--sentinel-reference-warmup-s", "120",
        "--within-block-drift-tolerance", "0.10",
        "--within-block-qps-drift-tolerance", "0.20",
        "--min-duration-s", "5", "--min-completed-queries", "1000",
    ]
    env = {**os.environ, "PHASEGATE_CAMPAIGN_ROOT": str(campaign),
           "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                   check=True)
    rows = [row for row in read_jsonl(runs_path) if int(row.get("repeat", -1)) == repeat]
    if not rows or rows[-1].get("status") != "valid":
        if attempt == 1:
            run_repeat(campaign, freeze, set_name, trace, log)
            return
        raise RuntimeError(f"baseline block invalid twice: set {set_name} repeat {repeat}")


def export_and_judge(campaign: Path, freeze: dict[str, Any], set_name: str) -> bool:
    stage = f"r5_baseline_{set_name}"
    runs = [row for row in read_jsonl(campaign / stage / "raw/runs.jsonl")
            if row.get("status") == "valid"]
    by_repeat = {int(row["repeat"]): row for row in runs}
    if len(by_repeat) != 5:
        raise RuntimeError(f"expected five valid runs, found {len(by_repeat)}")
    selected = [by_repeat[i] for i in range(5)]
    med_tpot = median(float(row["p95_tpot_ms"]) for row in selected)
    med_ttft = median(float(row["p95_ttft_ms"]) for row in selected)
    run_rows = []
    for row in selected:
        memory = row["observer_memory_audit"]
        audit = row["observer_reconstruction_audit"]
        tpot_dev = abs(float(row["p95_tpot_ms"]) / med_tpot - 1)
        ttft_dev = abs(float(row["p95_ttft_ms"]) / med_ttft - 1)
        run_rows.append({
            "baseline_set": set_name, "repeat": row["repeat"], "run_key": row["run_key"],
            "official_p95_tpot_ms": row["p95_tpot_ms"], "tpot_abs_deviation": tpot_dev,
            "official_p95_ttft_ms": row["p95_ttft_ms"], "ttft_abs_deviation": ttft_dev,
            "within_3pct_both": tpot_dev <= .03 and ttft_dev <= .03,
            "duration_s": row["duration_s"], "request_count": row["llm_requests"],
            "observer_mode": row["observer_mode"], "observer_rate_hz": memory["observed_rate_hz"],
            "observer_missed_intervals": memory["missed_intervals"],
            "observer_subprocess_count": memory["subprocess_count"],
            "event_reconstruction_exact": all(audit[key] for key in
                ("sequence_exact", "timestamps_monotonic", "query_accounting_exact", "admissions_within_cap")),
            "token_data_clean": row["validity"]["token_data_clean"],
            "pageouts_delta": row["pageouts_delta"], "swap_used_delta_bytes": row["swap_used_delta_bytes"],
            "memory_pressure_clean": row["memory_pressure_clean"],
            "rss_bytes": row["resident_memory_bytes"], "peak_rss_bytes": row["peak_resident_memory_bytes"],
        })
    write_csv(campaign / "r5_baseline_runs.csv", run_rows)
    request_objects = read_jsonl(campaign / stage / "raw/requests.jsonl")
    request_rows = []
    for obj in request_objects:
        run = next(row for row in selected if row["run_key"] == obj["run_key"])
        for index, item in enumerate(obj["requests"]):
            gaps = [float(value) for value in item["tpot_intervals_ms"]]
            request_rows.append({
                "baseline_set": set_name, "repeat": run["repeat"], "run_key": run["run_key"],
                "request_index": index, "request_id": item["request_id"],
                "ttft_ms": item["ttft_ms"], "mean_tpot_ms": item["mean_tpot_ms"],
                "p50_tpot_ms": item["p50_tpot_ms"], "p95_tpot_ms": item["p95_tpot_ms"],
                "p99_tpot_ms": item["p99_tpot_ms"], "max_gap_ms": max(gaps),
                "above_14ms_diagnostic": float(item["p95_tpot_ms"]) > 14.0,
                "token_timestamp_count": len(item["token_timestamps"]),
            })
    write_csv(campaign / "r5_baseline_request_metrics.csv", request_rows)
    passed = all(row["within_3pct_both"] for row in run_rows)
    summary = {"baseline_set": set_name, "repository_commit": freeze["repository_commit"],
               "median_official_p95_tpot_ms": med_tpot,
               "median_official_p95_ttft_ms": med_ttft, "passed": passed,
               "rule": freeze["acceptance"]["definition"], "run_keys": [r["run_key"] for r in selected]}
    (campaign / f"R5_BASELINE_{set_name}_RESULT.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("freeze", type=Path)
    parser.add_argument("--set", choices=("A", "B"), default="A")
    args = parser.parse_args()
    freeze_path = args.freeze.resolve(); campaign = freeze_path.parent
    freeze = json.loads(freeze_path.read_text())
    if git_head() != freeze["repository_commit"]:
        raise SystemExit("repository commit differs from R5_BASELINE_FREEZE.json")
    if args.set == "B" and not (campaign / "R5_BASELINE_B_AUTHORIZATION.json").exists():
        raise SystemExit("set B requires a documented environmental-correction authorization")
    with (campaign / "r5_baseline_orchestration.log").open("a") as log:
        for trace in freeze["trace_sets"][args.set]:
            run_repeat(campaign, freeze, args.set, trace, log)
    if not export_and_judge(campaign, freeze, args.set):
        raise SystemExit(f"baseline set {args.set} failed frozen +/-3% stability gate")


if __name__ == "__main__":
    main()
