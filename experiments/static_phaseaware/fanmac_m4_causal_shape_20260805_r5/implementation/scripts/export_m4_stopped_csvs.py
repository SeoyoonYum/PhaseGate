#!/usr/bin/env python3
"""Export sanitized run/request CSVs from the stopped M4 campaign."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


RUN_FIELDS = [
    "stage", "run_key", "policy", "policy_arg", "repeat", "attempt", "status",
    "invalid_reason", "context", "output_tokens", "llm_requests", "prompt_seed",
    "query_seed", "prefill_cap", "decode_cap", "fixed_workers", "duration_s",
    "p95_tpot_ms", "p95_ttft_ms", "normalized_p95_tpot", "normalized_p95_ttft",
    "total_retrieval_goodput_qps", "total_completed_retrieval_queries",
    "retrieval_latency_p50_ms", "retrieval_latency_p95_ms", "prefill_retrieval_qps",
    "decode_retrieval_qps", "admitted_queries_prefill", "admitted_queries_decode",
    "completed_queries_prefill", "completed_queries_decode",
    "prefill_active_retrieval_worker_mean", "prefill_active_retrieval_worker_p95",
    "decode_active_retrieval_worker_mean", "decode_active_retrieval_worker_p95",
    "decode_overlap_fraction", "queue_nonempty_fraction", "cap_binding_fraction",
    "phase_transition_to_cap_ms", "phase_transition_to_cap_max_ms",
    "decode_cap_overshoot_fraction", "decode_cap_overshoot_worker_mean",
    "decode_cap_overshoot_worker_max", "tpot_within_block_drift_ratio",
    "ttft_within_block_drift_ratio", "retrieval_qps_within_block_ratio",
    "sentinel_before_deviation", "sentinel_after_deviation", "pageouts_delta",
    "swap_used_delta_bytes", "memory_free_percent_before", "memory_free_percent_after",
    "resident_memory_bytes", "peak_resident_memory_bytes", "peak_mlx_memory_mb",
    "single_faiss_process", "shared_index_load_count", "memory_pressure_clean",
    "pageout_soft_flag", "power_clean",
]


def run_rows(root: Path, stage: str) -> list[dict]:
    return [{field: row.get(field) for field in RUN_FIELDS}
            for row in read_jsonl(root / stage / "raw/runs.jsonl")]


def request_rows(root: Path, stages: list[str]) -> list[dict]:
    output = []
    for stage in stages:
        runs = {row["run_key"]: row for row in read_jsonl(root / stage / "raw/runs.jsonl")}
        for record in read_jsonl(root / stage / "raw/requests.jsonl"):
            run = runs[record["run_key"]]
            for request in record["requests"]:
                gaps = np.asarray(request["tpot_intervals_ms"], dtype=float)
                first = gaps[:4]
                output.append({
                    "stage": stage, "run_key": record["run_key"],
                    "policy": run["policy"], "repeat": run["repeat"],
                    "attempt": run["attempt"], "request_id": request["request_id"],
                    "prefill_ms": request["prefill_ms"], "ttft_ms": request["ttft_ms"],
                    "mean_tpot_ms": request["mean_tpot_ms"],
                    "p50_tpot_ms": request["p50_tpot_ms"],
                    "p95_tpot_ms": request["p95_tpot_ms"],
                    "p99_tpot_ms": request["p99_tpot_ms"],
                    "max_inter_token_gap_ms": float(gaps.max()),
                    "prefill_to_first_token_gap_ms": (
                        float(request["first_token"] - request["decode_start"]) * 1000),
                    "first_four_gap_mean_ms": float(first.mean()),
                    "first_four_gap_max_ms": float(first.max()),
                    "gpu_queue_ms": request["gpu_queue_ms"],
                    "end_to_end_ms": request["end_to_end_ms"],
                    "token_timestamp_count": len(request["token_timestamps"]),
                    "token_timestamps_monotonic": all(
                        b > a for a, b in zip(request["token_timestamps"],
                                              request["token_timestamps"][1:])),
                })
    return output


def stability_rows(root: Path, stages: list[str]) -> list[dict]:
    output = []
    for stage in stages:
        rows = read_jsonl(root / stage / "raw/runs.jsonl")
        median_tpot = float(np.median([row["p95_tpot_ms"] for row in rows]))
        median_ttft = float(np.median([row["p95_ttft_ms"] for row in rows]))
        for row in rows:
            output.append({
                "baseline_set": stage, "repeat": row["repeat"], "attempt": row["attempt"],
                "run_key": row["run_key"], "status": row["status"],
                "p95_tpot_ms": row["p95_tpot_ms"], "median_p95_tpot_ms": median_tpot,
                "p95_tpot_deviation": row["p95_tpot_ms"] / median_tpot - 1,
                "p95_ttft_ms": row["p95_ttft_ms"], "median_p95_ttft_ms": median_ttft,
                "p95_ttft_deviation": row["p95_ttft_ms"] / median_ttft - 1,
                "within_3_percent_joint": (
                    abs(row["p95_tpot_ms"] / median_tpot - 1) <= .03
                    and abs(row["p95_ttft_ms"] / median_ttft - 1) <= .03),
                "tpot_within_block_drift_ratio": row["tpot_within_block_drift_ratio"],
                "ttft_within_block_drift_ratio": row["ttft_within_block_drift_ratio"],
                "sentinel_before_deviation": row["sentinel_before_deviation"],
                "sentinel_after_deviation": row["sentinel_after_deviation"],
                "pageouts_delta": row["pageouts_delta"],
                "swap_used_delta_bytes": row["swap_used_delta_bytes"],
                "memory_pressure_clean": row["memory_pressure_clean"],
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    root = args.campaign.resolve()
    baselines = ["isolated_baseline", "isolated_baseline_revalidation"]
    write_csv(root / "m4_mechanism_runs.csv", run_rows(root, "mechanism"))
    write_csv(root / "m4_baseline_runs.csv",
              [row for stage in baselines for row in run_rows(root, stage)])
    write_csv(root / "m4_request_metrics.csv",
              request_rows(root, ["mechanism", *baselines]))
    write_csv(root / "m4_baseline_stability.csv", stability_rows(root, baselines))


if __name__ == "__main__":
    main()
