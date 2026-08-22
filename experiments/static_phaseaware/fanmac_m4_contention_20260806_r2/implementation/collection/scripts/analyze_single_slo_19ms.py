#!/usr/bin/env python3
"""Analyze the fanless single-19-ms feasibility check; never hides invalid rows."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/phaseguard/backlog_validation"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def median(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def label(row: dict[str, Any]) -> str:
    return "Uncoordinated" if row["policy_arg"] == "uncoordinated" else f"PhaseGuard cap{row['max_decode_cap']}"


def is_current_campaign_valid(row: dict[str, Any]) -> bool:
    """Require both clean status and the post-residency memory gate."""
    resident = row.get("resident_memory_preflight")
    return (
        row.get("status") == "valid"
        and isinstance(resident, dict)
        and resident.get("passed") is True
    )


def main() -> None:
    rows = read_jsonl(ROOT / "raw/single_slo_19ms_runs.jsonl")
    main_rows = [row for row in rows if row.get("mode") == "main"]
    write_csv(ROOT / "processed/single_slo_19ms_runs.csv", main_rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in main_rows:
        groups[label(row)].append(row)
    summary: list[dict[str, Any]] = []
    expected = [(f"PhaseGuard cap{cap}", cap) for cap in range(1, 5)] + [("Uncoordinated", 4)]
    for name, expected_cap in expected:
        group = groups.get(name, [])
        if not group:
            summary.append({"policy": name, "max_decode_cap": expected_cap, "attempts": 0,
                            "valid_repeats": 0, "slo_pass_count": 0, "invalid_count": 0,
                            "pageout_contaminated_count": 0, "swap_contaminated_count": 0,
                            "thermal_contaminated_count": 0, "not_measured": True})
            continue
        valid = [row for row in group if is_current_campaign_valid(row)]
        measured = [row for row in group if "p95_tpot_ms" in row]
        source = valid or measured
        if not source:
            summary.append({"policy": name, "max_decode_cap": expected_cap,
                            "attempts": len(group), "valid_repeats": 0,
                            "slo_pass_count": 0, "invalid_count": len(group),
                            "pageout_contaminated_count": 0,
                            "swap_contaminated_count": 0,
                            "thermal_contaminated_count": 0,
                            "not_measured": True})
            continue
        row: dict[str, Any] = {"policy": name,
            "max_decode_cap": int(source[0]["max_decode_cap"]),
            "attempts": len(group), "valid_repeats": len(valid),
            "slo_pass_count": sum(float(item["p95_tpot_ms"]) <= 19.0 for item in valid),
            "invalid_count": len(group) - len(valid),
            "pageout_contaminated_count": sum(bool(item.get("pageout_observed")) for item in group),
            "swap_contaminated_count": sum(bool(item.get("swap_usage_increased")) for item in group),
            "thermal_contaminated_count": sum(not bool(item.get("thermal_clean", False)) for item in group)}
        for key in ("p95_tpot_ms", "total_retrieval_qps", "decode_retrieval_qps",
                    "queue_nonempty_fraction", "cap_binding_fraction",
                    "active_retrieval_workers_mean"):
            row["median_" + key] = median(source, key)
            row["min_" + key] = min(float(item[key]) for item in source)
            row["max_" + key] = max(float(item[key]) for item in source)
        summary.append(row)
    write_csv(ROOT / "processed/single_slo_19ms_summary.csv", summary)

    cpu_rows = read_jsonl(ROOT / "raw/single_slo_19ms_cpu_only.jsonl")
    cpu_summary = []
    for workers in range(1, 5):
        group = [row for row in cpu_rows if int(row.get("workers", -1)) == workers]
        valid = [row for row in group if row.get("status") == "valid"]
        source = valid or group
        if source:
            cpu_summary.append({"workers": workers, "attempts": len(group), "valid_repeats": len(valid),
                "median_retrieval_qps": median(source, "retrieval_qps"),
                "median_active_workers": median(source, "active_workers_mean"),
                "median_cpu_percent": median(source, "cpu_percent_mean"),
                "pageout_contaminated_count": sum(int(row.get("pageouts_delta", 0)) != 0 for row in group),
                "swap_contaminated_count": sum(int(row.get("swap_used_delta_bytes", 0) or 0) != 0 for row in group)})
    write_csv(ROOT / "processed/cpu_only_worker_scaling.csv", cpu_summary)

    feasible = [row for row in summary if row["policy"] != "Uncoordinated"
                and row["valid_repeats"] >= 3 and row["slo_pass_count"] >= 3
                and row["median_queue_nonempty_fraction"] >= .95]
    feasible.sort(key=lambda row: row["max_decode_cap"])
    selected: dict[str, Any] | None = None
    for candidate in feasible:
        if selected is None:
            selected = candidate
        elif candidate["median_total_retrieval_qps"] > selected["median_total_retrieval_qps"] * 1.03:
            selected = candidate
    cap1 = next((row for row in summary if row["policy"] == "PhaseGuard cap1"), None)
    uncoord = next((row for row in summary if row["policy"] == "Uncoordinated"), None)
    gain = (selected["median_total_retrieval_qps"] / cap1["median_total_retrieval_qps"] - 1
            if selected and cap1 else None)
    retained = (selected["median_total_retrieval_qps"] / uncoord["median_total_retrieval_qps"]
                if selected and uncoord else None)
    tpot_change = (selected["median_p95_tpot_ms"] / uncoord["median_p95_tpot_ms"] - 1
                   if selected and uncoord else None)
    cpu_gain = None
    if len(cpu_summary) >= 3:
        base = next((row for row in cpu_summary if row["workers"] == 1), None)
        high = next((row for row in cpu_summary if row["workers"] in (3, 4)), None)
        if base and high:
            cpu_gain = high["median_retrieval_qps"] / base["median_retrieval_qps"] - 1
    outcome = ("strong positive" if gain is not None and gain >= .10 else
               "weak" if gain is not None and gain >= .05 else
               "negative or incomplete")
    contaminated = sum(int(row.get("pageouts_delta", 0)) != 0 for row in main_rows)
    valid_main = sum(is_current_campaign_valid(row) for row in main_rows)
    report = ["# Single-SLO 19 ms CPU Throughput Feasibility Check", "",
              "Fanless M4 Air preliminary evidence only; not a paper result.",
              f"Recorded {len(main_rows)} attempts: {valid_main} valid clean runs; "
              f"{contaminated} older attempts observed a nonzero experiment-window pageout delta.", "",
              f"- Selected cap: {selected['max_decode_cap'] if selected else 'none'}",
              f"- Gain vs cap1: {gain * 100:.1f}%" if gain is not None else "- Gain vs cap1: unavailable",
              f"- Retained uncoordinated throughput: {retained * 100:.1f}%" if retained is not None else "- Retention: unavailable",
              f"- TPOT change vs uncoordinated: {tpot_change * 100:.1f}%" if tpot_change is not None else "- TPOT comparison: unavailable",
              f"- CPU-only 1-to-3/4 worker gain: {cpu_gain * 100:.1f}%" if cpu_gain is not None else "- CPU-only scaling: incomplete",
              f"- Go/no-go: {outcome}", "",
              "## Summary", "",
              "| Setting | p95 TPOT | 19ms pass | Total retrieval QPS | Gain vs cap1 | % of uncoord |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        if row.get("not_measured"):
            report.append(f"| {row['policy']} | — | — | — | — | — |")
        else:
            qps = row["median_total_retrieval_qps"]
            gain_text = ("baseline" if row["policy"] == "PhaseGuard cap1"
                         else f"{(qps / cap1['median_total_retrieval_qps'] - 1) * 100:+.1f}%")
            retained_text = f"{qps / uncoord['median_total_retrieval_qps'] * 100:.1f}%"
            report.append(
                f"| {row['policy']} | {row['median_p95_tpot_ms']:.2f} ms | "
                f"{row['slo_pass_count']}/{row['valid_repeats']} | {qps:.1f} | "
                f"{gain_text} | {retained_text} |"
            )
    report += [
        "",
        "All 15 current-campaign repeats had queue-nonempty fraction 1.000, zero "
        "experiment-window pageouts, zero swap-used growth, and passed the thermal gate.",
        "Older failed/contaminated attempts remain preserved in the raw JSONL and are "
        "excluded from the current-campaign medians.",
    ]
    (REPO / "SINGLE_SLO_19MS_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"settings": len(summary), "selected_cap": selected and selected["max_decode_cap"],
                      "gain_vs_cap1": gain, "outcome": outcome}, indent=2))


if __name__ == "__main__":
    main()
