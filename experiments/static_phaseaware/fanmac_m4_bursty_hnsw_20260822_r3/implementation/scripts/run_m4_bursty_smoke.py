#!/usr/bin/env python3
"""Freeze and run the bursty mechanics and 100%-ON compatibility smoke stages."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BLOCK = REPO / "scripts/run_static_phaseaware_pilot.py"
R5 = REPO / "experiments/static_phaseaware/fanmac_m4_causal_shape_20260805_r5"
sys.path.insert(0, str(REPO / "src"))

from phaseguard.demand_gate import FrozenDemandTrace, generate_demand_trace  # noqa: E402


POLICIES = (
    {"name": "fixed1", "family": "fixed", "high": 1, "low": 1},
    {"name": "phasegate4to1", "family": "phasegate", "high": 4, "low": 1},
    {"name": "timegate4to1", "family": "timegate", "high": 4, "low": 1},
)
MODEL = Path("/Users/m1/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-4bit/snapshots/8b403126fc14f14cfc99bb4cfa72ecbc129ea677")
INDEX = REPO / "experiments/phaseguard/index/hnsw_100k_d384.faiss"
TIMEGATE = R5 / "timegate_schedule_freeze.json"
BASELINE = R5 / "r5_normalization_baseline.json"


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.strip()


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


def coverage_offset(trace: FrozenDemandTrace, expected_duration_s: float) -> float:
    """Choose an outcome-blind offset solely by scheduled-duty coverage."""
    candidates = np.arange(0.0, trace.duration_s, 0.25)
    return float(min(candidates, key=lambda offset: (
        abs(trace.scheduled_on_fraction(float(offset), expected_duration_s)
            - trace.demand_level), float(offset))))


def freeze_mechanics(root: Path) -> dict[str, Any]:
    path = root / "SMOKE_PROTOCOL_FREEZE.json"
    if path.exists():
        return json.loads(path.read_text())
    trace_dir = root / "frozen_traces"; trace_dir.mkdir(parents=True, exist_ok=True)
    traces: dict[str, dict[str, Any]] = {}
    seeds = {"5": 2026082105, "25": 2026082125, "100": 2026082100}
    for key, level in (("5", .05), ("25", .25), ("100", 1.0)):
        trace = generate_demand_trace(demand_level=level, seed=seeds[key])
        trace_path = trace_dir / f"smoke_duty{key}_seed{trace.seed}.json"
        trace.write(trace_path)
        traces[key] = {"demand_level": level, "seed": trace.seed,
                       "path": str(trace_path.resolve()),
                       "offset_s": 0.0 if level == 1.0 else coverage_offset(trace, 72.0),
                       "offset_selection": "minimum scheduled-duty error at a fixed 72-second design window; selected before smoke performance"}
    cells = [{"duty": duty, "policy": dict(policy)}
             for duty in ("5", "25", "100") for policy in POLICIES]
    random.Random(2026082191).shuffle(cells)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_head(), "order_seed": 2026082191,
        "requests": 20, "context_tokens": 2048, "output_tokens": 128,
        "observer_mode": "event", "traces": traces, "randomized_cells": cells,
        "prompt_query_seeds": {
            duty: {"prompt_seed": 4200000011 + index * 10007,
                   "query_seed": 4200500014 + index * 10007}
            for index, duty in enumerate(("5", "25", "100"))
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def command(*, stage: str, policy: dict[str, Any], repeat: int, attempt: int,
            requests: int, prompt_seed: int, query_seed: int,
            trace: Path | None, offset_s: float) -> list[str]:
    result = [sys.executable, str(BLOCK), "--run-one", "--stage", stage,
        "--policy", str(policy["family"]), "--observer-mode", "event",
        "--repeat", str(repeat), "--attempt", str(attempt),
        "--fixed-workers", str(policy["low"]), "--prefill-cap", str(policy["high"]),
        "--decode-cap", str(policy["low"]), "--prompt-seed", str(prompt_seed),
        "--query-seed", str(query_seed), "--model", str(MODEL), "--index", str(INDEX),
        "--context", "2048", "--output-tokens", "128", "--llm-requests", str(requests),
        "--max-workers", "4", "--feeders", "8", "--queries-per-task", "4096",
        "--chunk", "16", "--ef-search", "128", "--top-k", "10",
        "--memory-sample-interval-s", "1", "--warmup-s", "2", "--mem-limit-gb", "5.5",
        "--min-headroom-gb", "3", "--memory-idle-seconds", "2",
        "--sentinel-tolerance", "0.03", "--sentinel-cooldown", "2",
        "--sentinel-attempts", "2", "--sentinel-reps", "2",
        "--sentinel-reference-warmup-s", "120", "--within-block-drift-tolerance", "0.10",
        "--within-block-qps-drift-tolerance", "0.20", "--min-duration-s", "5",
        "--min-completed-queries", "64", "--baseline-file", str(BASELINE),
        "--slo-multiplier", "1.25"]
    if policy["family"] == "timegate":
        result += ["--timegate-schedule", str(TIMEGATE), "--timegate-offset-s", "0"]
    if trace is not None:
        result += ["--demand-trace", str(trace), "--demand-offset-s", str(offset_s)]
    return result


def run_command(root: Path, command_line: list[str], log: Any) -> None:
    environment = {**os.environ, "PHASEGATE_CAMPAIGN_ROOT": str(root),
                   "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    subprocess.run(command_line, cwd=REPO, env=environment, stdout=log,
                   stderr=subprocess.STDOUT, check=True)


def mechanics(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    freeze = freeze_mechanics(root)
    runs_path = root / "mechanics/raw/runs.jsonl"
    with (root / "mechanics_smoke.log").open("a") as log:
        for cell in freeze["randomized_cells"]:
            duty = str(cell["duty"]); policy = dict(cell["policy"])
            trace = freeze["traces"][duty]
            existing = [row for row in read_jsonl(runs_path)
                        if row.get("policy") == policy["name"]
                        and abs(float(row.get("bursty_demand_audit", {}).get(
                            "demand_level", -1)) - float(trace["demand_level"])) < 1e-9]
            if existing:
                continue
            seeds = freeze["prompt_query_seeds"][duty]
            run_command(root, command(
                stage="mechanics", policy=policy, repeat=int(duty), attempt=1,
                requests=20, prompt_seed=int(seeds["prompt_seed"]),
                query_seed=int(seeds["query_seed"]), trace=Path(trace["path"]),
                offset_s=float(trace["offset_s"])), log)
    rows = read_jsonl(runs_path)
    if len(rows) != 9:
        raise RuntimeError(f"mechanics smoke expected 9 rows, found {len(rows)}")
    output = []
    for row in rows:
        audit = row["bursty_demand_audit"]
        output.append({"run_key": row["run_key"], "status": row["status"],
            "invalid_reason": row["invalid_reason"], "policy": row["policy"],
            "demand_level": audit["demand_level"], "duration_s": row["duration_s"],
            "scheduled_on_fraction": audit["scheduled_on_fraction"],
            "actual_hnsw_active_fraction": audit["actual_hnsw_active_fraction"],
            "new_chunk_starts_during_off": audit["new_chunk_starts_during_off"],
            "off_active_excluding_bounded_drain_s": audit["off_active_excluding_bounded_drain_s"],
            "query_accounting_exact": row["observer_reconstruction_audit"]["query_accounting_exact"],
            "admissions_within_demand": row["observer_reconstruction_audit"]["admissions_within_demand"],
            "token_data_clean": row["validity"]["token_data_clean"],
            "policy_cap_exact": row["validity"]["policy_cap_applied"],
            "demand_semantics_clean": row["validity"]["demand_semantics_clean"],
            "demand_duty_clean": row["validity"]["demand_duty_clean"],
            "timegate_phase_blind": (row["timegate_audit"] is None or
                                     row["timegate_audit"]["phase_state_consulted"] is False),
            "demand_phase_blind": row["demand_gate_audit"]["phase_state_consulted"] is False,
            "demand_policy_blind": row["demand_gate_audit"]["policy_identity_consulted"] is False,
            "observer_subprocess_count": row["observer_subprocess_count_during_block"]})
    write_csv(root / "mechanics_smoke_audit.csv", output)
    duty_imbalances = {}
    for duty in (.05, .25, 1.0):
        values = [float(item["scheduled_on_fraction"]) for item in output
                  if abs(float(item["demand_level"]) - duty) < 1e-9]
        duty_imbalances[str(duty)] = max(values) - min(values)
    mechanics_passed = all(
        item["new_chunk_starts_during_off"] == 0
        and item["off_active_excluding_bounded_drain_s"] <= 1e-9
        and item["query_accounting_exact"] and item["admissions_within_demand"]
        and item["token_data_clean"] and item["policy_cap_exact"]
        and item["demand_semantics_clean"] and item["demand_duty_clean"]
        and item["timegate_phase_blind"] and item["demand_phase_blind"]
        and item["demand_policy_blind"] and item["observer_subprocess_count"] == 0
        for item in output) and all(value <= .02 for value in duty_imbalances.values())
    (root / "MECHANICS_SMOKE_RESULT.json").write_text(json.dumps({
        "passed": mechanics_passed,
        "scope": "mechanics only; performance is non-paper diagnostic and is not pooled",
        "performance_valid_blocks": sum(item["status"] == "valid" for item in output),
        "performance_invalid_blocks": [
            {"run_key": item["run_key"], "invalid_reason": item["invalid_reason"]}
            for item in output if item["status"] != "valid"],
        "matched_triplet_scheduled_duty_max_difference": duty_imbalances,
        "measured_block_durations_s": [item["duration_s"] for item in output],
        "median_measured_block_duration_s": median(item["duration_s"] for item in output),
        "rows": output}, indent=2) + "\n")


def freeze_compatibility(root: Path) -> dict[str, Any]:
    path = root / "COMPATIBILITY_PROTOCOL_FREEZE.json"
    if path.exists():
        return json.loads(path.read_text())
    trace_dir = root / "frozen_traces"; trace_dir.mkdir(parents=True, exist_ok=True)
    trace = generate_demand_trace(demand_level=1.0, seed=2026082200)
    trace_path = trace_dir / "compatibility_100pct.json"; trace.write(trace_path)
    repeats = []
    for repeat in range(3):
        order = ["original", "demand100"]
        random.Random(2026082201 + repeat).shuffle(order)
        repeats.append({"repeat": repeat, "order": order,
                        "prompt_seed": 4210000011 + repeat * 10007,
                        "query_seed": 4210500014 + repeat * 10007})
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "repository_commit": git_head(), "requests": 100,
               "policy": "fixed1", "trace": str(trace_path.resolve()),
               "trace_seed": trace.seed, "repeats": repeats}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def compatibility(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    freeze = freeze_compatibility(root)
    runs_path = root / "compatibility/raw/runs.jsonl"
    fixed = dict(POLICIES[0])
    with (root / "compatibility_smoke.log").open("a") as log:
        for spec in freeze["repeats"]:
            repeat = int(spec["repeat"])
            for condition in spec["order"]:
                want_demand = condition == "demand100"
                existing = [row for row in read_jsonl(runs_path)
                            if int(row.get("repeat", -1)) == repeat
                            and (row.get("bursty_demand_audit") is not None) == want_demand]
                if existing:
                    continue
                run_command(root, command(
                    stage="compatibility", policy=fixed, repeat=repeat, attempt=1,
                    requests=100, prompt_seed=int(spec["prompt_seed"]),
                    query_seed=int(spec["query_seed"]),
                    trace=Path(freeze["trace"]) if want_demand else None,
                    offset_s=0.0), log)
    runs = read_jsonl(runs_path)
    requests = {item["run_key"]: item["requests"] for item in
                read_jsonl(root / "compatibility/raw/requests.jsonl")}
    pair_rows = []
    for repeat in range(3):
        selected = [row for row in runs if int(row["repeat"]) == repeat]
        original = next(row for row in selected if row["bursty_demand_audit"] is None)
        demand = next(row for row in selected if row["bursty_demand_audit"] is not None)
        original_requests = requests[original["run_key"]]
        demand_requests = requests[demand["run_key"]]
        request_median_original = median(float(item["mean_tpot_ms"])
                                         for item in original_requests)
        request_median_demand = median(float(item["mean_tpot_ms"])
                                       for item in demand_requests)
        all_token_original = float(np.mean([
            gap for item in original_requests for gap in item["tpot_intervals_ms"]]))
        all_token_demand = float(np.mean([
            gap for item in demand_requests for gap in item["tpot_intervals_ms"]]))
        pair_rows.append({"repeat": repeat, "original_run_key": original["run_key"],
            "demand100_run_key": demand["run_key"],
            "request_median_tpot_relative_change": request_median_demand / request_median_original - 1,
            "all_token_mean_gap_relative_change": all_token_demand / all_token_original - 1,
            "original_official_p95_tpot_ms": original["p95_tpot_ms"],
            "demand100_official_p95_tpot_ms": demand["p95_tpot_ms"],
            "original_official_p95_ttft_ms": original["p95_ttft_ms"],
            "demand100_official_p95_ttft_ms": demand["p95_ttft_ms"],
            "original_accounting_exact": original["observer_reconstruction_audit"]["query_accounting_exact"],
            "demand100_accounting_exact": demand["observer_reconstruction_audit"]["query_accounting_exact"],
            "demand100_admissions_within_demand": demand["observer_reconstruction_audit"]["admissions_within_demand"],
            "original_policy_cap_exact": original["validity"]["policy_cap_applied"],
            "demand100_policy_cap_exact": demand["validity"]["policy_cap_applied"]})
    write_csv(root / "compatibility_pairwise.csv", pair_rows)
    median_request_change = median(row["request_median_tpot_relative_change"] for row in pair_rows)
    median_token_change = median(row["all_token_mean_gap_relative_change"] for row in pair_rows)
    semantic = all(row[key] for row in pair_rows for key in (
        "original_accounting_exact", "demand100_accounting_exact",
        "demand100_admissions_within_demand", "original_policy_cap_exact",
        "demand100_policy_cap_exact"))
    result = {"semantic_accounting_agreement": semantic,
              "median_paired_request_median_tpot_relative_change": median_request_change,
              "median_paired_all_token_mean_gap_relative_change": median_token_change,
              "request_median_gate_pass": abs(median_request_change) <= .01,
              "all_token_mean_gate_pass": abs(median_token_change) <= .01,
              "passed": semantic and abs(median_request_change) <= .01
                        and abs(median_token_change) <= .01,
              "pairs": pair_rows}
    (root / "COMPATIBILITY_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanics-root", type=Path, required=True)
    parser.add_argument("--compatibility-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("mechanics", "compatibility", "all"),
                        default="all")
    args = parser.parse_args()
    if args.stage in ("mechanics", "all"):
        mechanics(args.mechanics_root.resolve())
    if args.stage in ("compatibility", "all"):
        compatibility(args.compatibility_root.resolve())


if __name__ == "__main__":
    main()
