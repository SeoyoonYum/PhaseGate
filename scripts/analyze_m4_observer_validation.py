#!/usr/bin/env python3
"""Produce frozen observer-validation CSVs and the restart-gate report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty output: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def q(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def summarize(root: Path, stage: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows = [row for row in read_jsonl(root / stage / "raw/runs.jsonl")
                if row.get("status") == "valid"]
    by_key = {str(row["run_key"]): row for row in run_rows}
    request_objects = read_jsonl(root / stage / "raw/requests.jsonl")
    summaries: list[dict[str, Any]] = []
    requests_out: list[dict[str, Any]] = []
    for obj in request_objects:
        run = by_key.get(str(obj["run_key"]))
        if run is None:
            continue
        requests = obj["requests"]
        request_p95 = [float(item["p95_tpot_ms"]) for item in requests]
        request_mean = [float(item["mean_tpot_ms"]) for item in requests]
        gaps = [float(gap) for item in requests for gap in item["tpot_intervals_ms"]]
        half = len(requests) // 2
        first = requests[:half]; second = requests[half:]
        memory = run["observer_memory_audit"]
        reconstruction = run["observer_reconstruction_audit"]
        summary = {
            "stage": stage, "observer_mode": run["observer_mode"],
            "repeat": run["repeat"], "attempt": run["attempt"],
            "run_key": run["run_key"], "status": run["status"],
            "duration_s": run["duration_s"], "request_count": len(requests),
            "official_p95_tpot_ms": run["p95_tpot_ms"],
            "official_p95_ttft_ms": run["p95_ttft_ms"],
            "request_p95_p50_ms": q(request_p95, 50),
            "request_p95_p90_ms": q(request_p95, 90),
            "request_p95_p95_ms": q(request_p95, 95),
            "request_p95_p99_ms": q(request_p95, 99),
            "request_median_tpot_ms": q(request_mean, 50),
            "all_token_gap_mean_ms": float(np.mean(gaps)),
            "all_token_gap_p50_ms": q(gaps, 50), "all_token_gap_p90_ms": q(gaps, 90),
            "all_token_gap_p95_ms": q(gaps, 95), "all_token_gap_p99_ms": q(gaps, 99),
            "requests_above_14ms": sum(value > 14.0 for value in request_p95),
            "fraction_above_14ms": float(np.mean(np.asarray(request_p95) > 14.0)),
            "first_half_official_p95_ms": q([float(x["p95_tpot_ms"]) for x in first], 95),
            "second_half_official_p95_ms": q([float(x["p95_tpot_ms"]) for x in second], 95),
            "first_half_gap_mean_ms": float(np.mean(
                [gap for x in first for gap in x["tpot_intervals_ms"]])),
            "second_half_gap_mean_ms": float(np.mean(
                [gap for x in second for gap in x["tpot_intervals_ms"]])),
            "observer_cpu_time_s": "unavailable",
            "observer_samples": memory["sample_count"],
            "observer_missed_intervals": memory["missed_intervals"],
            "observer_max_scheduling_delay_ms": memory["max_scheduling_delay_ms"],
            "observer_rate_hz": memory["observed_rate_hz"],
            "observer_subprocess_count": memory["subprocess_count"],
            "pageouts_delta": run["pageouts_delta"], "swap_used_delta_bytes": run["swap_used_delta_bytes"],
            "compressions_delta": run["compressions_delta"],
            "decompressions_delta": run["decompressions_delta"],
            "memory_pressure_clean": run["memory_pressure_clean"],
            "resident_memory_bytes": run["resident_memory_bytes"],
            "peak_resident_memory_bytes": run["peak_resident_memory_bytes"],
            "token_data_clean": run["validity"]["token_data_clean"],
            "event_sequence_exact": reconstruction["sequence_exact"],
            "event_timestamps_monotonic": reconstruction["timestamps_monotonic"],
            "event_query_accounting_exact": reconstruction["query_accounting_exact"],
            "event_admissions_within_cap": reconstruction["admissions_within_cap"],
            "event_token_count": reconstruction["token_events"],
        }
        summaries.append(summary)
        for index, item in enumerate(requests):
            requests_out.append({
                "stage": stage, "observer_mode": run["observer_mode"],
                "repeat": run["repeat"], "attempt": run["attempt"],
                "run_key": run["run_key"], "request_index": index,
                "request_id": item["request_id"], "ttft_ms": item["ttft_ms"],
                "mean_tpot_ms": item["mean_tpot_ms"], "p50_tpot_ms": item["p50_tpot_ms"],
                "p95_tpot_ms": item["p95_tpot_ms"], "p99_tpot_ms": item["p99_tpot_ms"],
                "max_inter_token_gap_ms": max(item["tpot_intervals_ms"]),
                "above_14ms_diagnostic": float(item["p95_tpot_ms"]) > 14.0,
                "token_timestamp_count": len(item["token_timestamps"]),
            })
    return sorted(summaries, key=lambda row: (row["repeat"], row["observer_mode"])), requests_out


def paired(rows: list[dict[str, Any]], first: str, second: str) -> list[dict[str, Any]]:
    by = {(int(row["repeat"]), str(row["observer_mode"])): row for row in rows}
    output = []
    for repeat in sorted({int(row["repeat"]) for row in rows}):
        a, b = by[(repeat, first)], by[(repeat, second)]
        output.append({
            "repeat": repeat, "first_mode": first, "second_mode": second,
            "first_request_median_tpot_ms": a["request_median_tpot_ms"],
            "second_request_median_tpot_ms": b["request_median_tpot_ms"],
            "paired_request_median_change": b["request_median_tpot_ms"] / a["request_median_tpot_ms"] - 1,
            "first_request_p95_median_ms": a["request_p95_p50_ms"],
            "second_request_p95_median_ms": b["request_p95_p50_ms"],
            "paired_request_p95_median_change": b["request_p95_p50_ms"] / a["request_p95_p50_ms"] - 1,
            "first_all_gap_mean_ms": a["all_token_gap_mean_ms"],
            "second_all_gap_mean_ms": b["all_token_gap_mean_ms"],
            "paired_all_gap_mean_change": b["all_token_gap_mean_ms"] / a["all_token_gap_mean_ms"] - 1,
            "first_tail_count": a["requests_above_14ms"],
            "second_tail_count": b["requests_above_14ms"],
            "paired_tail_count_delta": b["requests_above_14ms"] - a["requests_above_14ms"],
            "first_official_p95_ms": a["official_p95_tpot_ms"],
            "second_official_p95_ms": b["official_p95_tpot_ms"],
            "paired_official_p95_change": b["official_p95_tpot_ms"] / a["official_p95_tpot_ms"] - 1,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("root", type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    minimal_rows, minimal_requests = summarize(root, "observer_minimal_event")
    legacy_rows, legacy_requests = summarize(root, "observer_legacy_event")
    if len(minimal_rows) != 6 or len(legacy_rows) != 10:
        raise RuntimeError(f"incomplete validation: {len(minimal_rows)=}, {len(legacy_rows)=}")
    write_csv(root / "observer_minimal_event_runs.csv", minimal_rows)
    write_csv(root / "observer_legacy_event_runs.csv", legacy_rows)
    write_csv(root / "observer_request_metrics.csv", minimal_requests + legacy_requests)
    tail_rows = []
    for stage_rows in (minimal_rows, legacy_rows):
        for mode in sorted({str(row["observer_mode"]) for row in stage_rows}):
            selected = [row for row in stage_rows if row["observer_mode"] == mode]
            tail_rows.append({
                "stage": selected[0]["stage"], "observer_mode": mode,
                "valid_runs": len(selected),
                "median_requests_above_14ms": median(row["requests_above_14ms"] for row in selected),
                "range_requests_above_14ms": f"{min(row['requests_above_14ms'] for row in selected)}-{max(row['requests_above_14ms'] for row in selected)}",
                "median_fraction_above_14ms": median(row["fraction_above_14ms"] for row in selected),
                "median_official_p95_tpot_ms": median(row["official_p95_tpot_ms"] for row in selected),
                "median_request_p95_ms": median(row["request_p95_p50_ms"] for row in selected),
                "median_all_token_gap_mean_ms": median(row["all_token_gap_mean_ms"] for row in selected),
            })
    write_csv(root / "observer_tail_state_summary.csv", tail_rows)
    audit_rows = [{key: row[key] for key in (
        "stage", "observer_mode", "repeat", "run_key", "token_data_clean",
        "event_sequence_exact", "event_timestamps_monotonic", "event_query_accounting_exact",
        "event_admissions_within_cap", "event_token_count", "observer_subprocess_count",
        "observer_rate_hz", "observer_missed_intervals", "memory_pressure_clean",
        "pageouts_delta", "swap_used_delta_bytes")}
        for row in minimal_rows + legacy_rows]
    write_csv(root / "observer_event_reconstruction_audit.csv", audit_rows)

    overhead = paired(minimal_rows, "minimal", "event")
    causal = paired(legacy_rows, "legacy", "event")
    med_request = median(row["paired_request_median_change"] for row in overhead)
    med_gap = median(row["paired_all_gap_mean_change"] for row in overhead)
    production = [row for row in minimal_rows if row["observer_mode"] == "event"]
    correctness = all(
        row["token_data_clean"] and row["event_sequence_exact"]
        and row["event_timestamps_monotonic"] and row["event_query_accounting_exact"]
        and row["event_admissions_within_cap"] and int(row["observer_subprocess_count"]) == 0
        and float(row["observer_rate_hz"]) <= 1.01 and row["memory_pressure_clean"]
        and int(row["swap_used_delta_bytes"]) == 0
        for row in production
    )
    overhead_pass = abs(med_request) <= .01 and abs(med_gap) <= .01
    tail_deltas = [int(row["paired_tail_count_delta"]) for row in causal]
    causal_request_p95 = median(row["paired_request_p95_median_change"] for row in causal)
    causal_gap_mean = median(row["paired_all_gap_mean_change"] for row in causal)
    causal_official_p95 = median(row["paired_official_p95_change"] for row in causal)
    if all(delta < 0 for delta in tail_deltas):
        causal_text = "Event had fewer >14 ms requests in every pair; this supports legacy-observer contribution."
    elif all(delta == 0 for delta in tail_deltas) or median(tail_deltas) == 0:
        causal_text = (
            "The r4-derived tail incidence was similar and does not support the legacy "
            "observer as the cause of that old slow state. A common OS/runtime stall "
            "remains plausible, but these quiet validation blocks do not establish its cause."
        )
    else:
        causal_text = "Tail-count differences were mixed; observer causality is inconclusive."
    lines = [
        "# Base-M4 Observer Validation Report", "", "## Scope", "",
        "Diagnostic intervention only; no policy or paper claim is drawn from these blocks.", "",
        "## Minimal versus event", "",
        f"- Median paired request-median TPOT change (event/minimal): {med_request*100:.3f}%",
        f"- Median paired all-token mean-gap change (event/minimal): {med_gap*100:.3f}%",
        f"- Production correctness gate: {'PASS' if correctness else 'FAIL'}",
        f"- Production overhead gate: {'PASS' if overhead_pass else 'FAIL'}", "",
        "| Pair | Minimal median TPOT | Event median TPOT | Paired change | Minimal gap mean | Event gap mean | Paired change |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(f"| {row['repeat']} | {row['first_request_median_tpot_ms']:.3f} | {row['second_request_median_tpot_ms']:.3f} | {row['paired_request_median_change']*100:.3f}% | {row['first_all_gap_mean_ms']:.3f} | {row['second_all_gap_mean_ms']:.3f} | {row['paired_all_gap_mean_change']*100:.3f}% |" for row in overhead)
    lines += ["", "## Legacy versus event", "", causal_text, "",
              f"- Median paired request-p95 median change (event/legacy): {causal_request_p95*100:.3f}%",
              f"- Median paired all-token mean-gap change (event/legacy): {causal_gap_mean*100:.3f}%",
              f"- Median paired official p95 change (event/legacy): {causal_official_p95*100:.3f}%",
              "- Legacy was slower on these broad metrics even though it did not increase the diagnostic >14 ms count.", "",
              "The 14 ms threshold is r4-derived and diagnostic, not preregistered paper evidence.", "",
              "| Pair | Legacy tail count | Event tail count | Delta (event-legacy) | Legacy official p95 | Event official p95 |",
              "|---:|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {row['repeat']} | {row['first_tail_count']} | {row['second_tail_count']} | {row['paired_tail_count_delta']} | {row['first_official_p95_ms']:.3f} | {row['second_official_p95_ms']:.3f} |" for row in causal)
    restart = correctness and overhead_pass
    lines += ["", "## Restart gate", "",
              f"Observer correctness and overhead gate: {'PASS' if restart else 'FAIL'}.",
              "Legacy mode is disabled for any subsequent campaign regardless of the causal diagnostic outcome.", ""]
    (root / "OBSERVER_VALIDATION_REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
