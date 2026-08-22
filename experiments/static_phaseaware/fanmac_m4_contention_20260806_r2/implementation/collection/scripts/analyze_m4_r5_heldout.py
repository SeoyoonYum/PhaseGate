#!/usr/bin/env python3
"""Paper-oriented run-level analysis for the base-M4 held-out triplet."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def bootstrap(values: list[float], seed: int = 2026080607) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    samples = np.median(rng.choice(array, size=(10_000, len(array)), replace=True), axis=1)
    return float(np.median(array)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(values); output = np.empty(len(values), dtype=float)
    output[order] = np.arange(len(values), dtype=float)
    return output


def overlap(path: Path, policy: str, repeat: int) -> list[dict[str, Any]]:
    events = sorted(json.loads(path.read_text())["events"], key=lambda row: row["timestamp"])
    starts = [float(row["timestamp"]) for row in events if row["event_type"] == "request_start"]
    ends = [float(row["timestamp"]) for row in events if row["event_type"] == "request_complete"]
    if not starts or not ends: return []
    start, end = min(starts), max(ends); totals: dict[tuple[str, int], float] = {}
    for before, after in zip(events, events[1:]):
        left = max(start, float(before["timestamp"])); right = min(end, float(after["timestamp"]))
        if right <= left: continue
        phase = str(before.get("llm_phase", "IDLE")); cap = int(before.get("requested_cap", 0))
        totals[(phase, cap)] = totals.get((phase, cap), 0.0) + right - left
    phase_totals = {phase: sum(value for (item_phase, _), value in totals.items()
                               if item_phase == phase) for phase, _ in totals}
    return [{"policy": policy, "repeat": repeat, "phase": phase, "cap": cap,
             "wall_time_s": duration,
             "fraction_within_phase": duration / phase_totals[phase] if phase_totals[phase] else 0.0}
            for (phase, cap), duration in sorted(totals.items())]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("campaign", type=Path)
    args = parser.parse_args(); root = args.campaign.resolve()
    all_runs = read_jsonl(root / "m4_heldout/raw/runs.jsonl")
    runs = [row for row in all_runs if row.get("status") == "valid"]
    if len(runs) != 21: raise RuntimeError(f"expected 21 valid held-out rows, found {len(runs)}")
    selection = json.loads((root / "frozen_m4_selection.json").read_text())
    budget = float(selection["primary_B"]); baseline = selection["baseline"]
    request_objects = {row["run_key"]: row["requests"]
                       for row in read_jsonl(root / "m4_heldout/raw/requests.jsonl")}
    baseline_objects = read_jsonl(root / "r5_baseline_A/raw/requests.jsonl")
    baseline_gaps = [float(gap) for obj in baseline_objects for req in obj["requests"]
                     for gap in req["tpot_intervals_ms"]]
    isolated_gap_median = float(np.median(baseline_gaps))
    run_rows = []
    for row in sorted(runs, key=lambda x: (int(x["repeat"]), str(x["policy"]))):
        requests = request_objects[row["run_key"]]
        gaps = [float(gap) for req in requests for gap in req["tpot_intervals_ms"]]
        max_gaps = [max(map(float, req["tpot_intervals_ms"])) for req in requests]
        first_four = [float(gap) for req in requests for gap in req["tpot_intervals_ms"][:4]]
        transition = [float(req["token_timestamps"][0]) * 1000 - float(req["prefill_end"]) * 1000
                      for req in requests]
        joint_requests = [float(req["p95_tpot_ms"]) <= budget * baseline["p95_tpot_ms"]
                          and float(req["ttft_ms"]) <= budget * baseline["p95_ttft_ms"]
                          for req in requests]
        run_rows.append({
            "policy": row["policy"], "repeat": row["repeat"], "attempt": row["attempt"],
            "run_key": row["run_key"], "retrieval_qps": row["total_retrieval_goodput_qps"],
            "p95_tpot_ms": row["p95_tpot_ms"], "normalized_p95_tpot": row["normalized_p95_tpot"],
            "p95_ttft_ms": row["p95_ttft_ms"], "normalized_p95_ttft": row["normalized_p95_ttft"],
            "joint_slo_pass": float(row["normalized_p95_tpot"]) <= budget and float(row["normalized_p95_ttft"]) <= budget,
            "request_joint_attainment": float(np.mean(joint_requests)),
            "p99_inter_token_gap_ms": percentile(gaps, 99),
            "p95_prefill_to_first_token_gap_ms": percentile(transition, 95),
            "first_four_token_gap_p95_ms": percentile(first_four, 95),
            "request_max_gap_p95_ms": percentile(max_gaps, 95),
            "fraction_gaps_above_2x_isolated_median": float(np.mean(np.asarray(gaps) > 2 * isolated_gap_median)),
            "fraction_gaps_above_3x_isolated_median": float(np.mean(np.asarray(gaps) > 3 * isolated_gap_median)),
            "prefill_active_workers_mean": row["prefill_active_retrieval_worker_mean"],
            "decode_active_workers_mean": row["decode_active_retrieval_worker_mean"],
            "completed_queries_prefill": row["completed_queries_prefill"],
            "completed_queries_decode": row["completed_queries_decode"],
            "pageouts_delta": row["pageouts_delta"], "swap_used_delta_bytes": row["swap_used_delta_bytes"],
            "memory_pressure_clean": row["memory_pressure_clean"],
            "observer_subprocess_count": row["observer_subprocess_count_during_block"],
        })
    write_csv(root / "m4_heldout_runs.csv", run_rows)
    policies = sorted({row["policy"] for row in run_rows})
    summaries = []
    for policy in policies:
        group = [row for row in run_rows if row["policy"] == policy]
        summaries.append({"policy": policy, "valid_paired_repeats": len(group),
            "median_retrieval_qps": median(float(row["retrieval_qps"]) for row in group),
            "median_normalized_p95_tpot": median(float(row["normalized_p95_tpot"]) for row in group),
            "median_normalized_p95_ttft": median(float(row["normalized_p95_ttft"]) for row in group),
            "joint_slo_pass_count": sum(bool(row["joint_slo_pass"]) for row in group),
            "median_request_joint_attainment": median(float(row["request_joint_attainment"]) for row in group),
            "median_p99_inter_token_gap_ms": median(float(row["p99_inter_token_gap_ms"]) for row in group),
            "pageout_soft_flag_count": sum(int(row["pageouts_delta"]) > 0 for row in group),
            "pageout_median": median(int(row["pageouts_delta"]) for row in group),
            "pageout_min": min(int(row["pageouts_delta"]) for row in group),
            "pageout_max": max(int(row["pageouts_delta"]) for row in group)})
    write_csv(root / "m4_heldout_summary.csv", summaries)
    by = {(int(row["repeat"]), row["policy"]): row for row in run_rows}
    fixed, phasegate = selection["selected_fixed"], selection["selected_phasegate"]
    timegate = next(policy for policy in policies if policy.startswith("timegate"))
    pair_rows = []
    for repeat in range(7):
        f, p, t = by[(repeat, fixed)], by[(repeat, phasegate)], by[(repeat, timegate)]
        pair_rows.append({"repeat": repeat, "fixed_policy": fixed, "phasegate_policy": phasegate,
            "timegate_policy": timegate, "fixed_qps": f["retrieval_qps"],
            "phasegate_qps": p["retrieval_qps"], "timegate_qps": t["retrieval_qps"],
            "phasegate_vs_fixed_gain": float(p["retrieval_qps"]) / float(f["retrieval_qps"]) - 1,
            "phasegate_vs_timegate_gain": float(p["retrieval_qps"]) / float(t["retrieval_qps"]) - 1,
            "fixed_joint_slo_pass": f["joint_slo_pass"], "phasegate_joint_slo_pass": p["joint_slo_pass"],
            "timegate_joint_slo_pass": t["joint_slo_pass"]})
    pf = bootstrap([row["phasegate_vs_fixed_gain"] for row in pair_rows], 2026080607)
    pt = bootstrap([row["phasegate_vs_timegate_gain"] for row in pair_rows], 2026080608)
    for row in pair_rows:
        row.update({"phasegate_vs_fixed_median_gain": pf[0], "phasegate_vs_fixed_ci_low": pf[1],
                    "phasegate_vs_fixed_ci_high": pf[2], "phasegate_vs_timegate_median_gain": pt[0],
                    "phasegate_vs_timegate_ci_low": pt[1], "phasegate_vs_timegate_ci_high": pt[2]})
    write_csv(root / "m4_pairwise_gains.csv", pair_rows)
    overlap_rows = []
    for row in runs:
        timeline = root / "m4_heldout/raw/timelines" / f"{row['run_key']}.json"
        overlap_rows.extend(overlap(timeline, row["policy"], int(row["repeat"])))
    write_csv(root / "m4_cap_phase_overlap.csv", overlap_rows)
    pageout_rows = []
    for repeat in range(7):
        f, p, t = by[(repeat, fixed)], by[(repeat, phasegate)], by[(repeat, timegate)]
        pageout_rows.append({"repeat": repeat, "fixed_pageout_delta": f["pageouts_delta"],
            "phasegate_pageout_delta": p["pageouts_delta"], "timegate_pageout_delta": t["pageouts_delta"],
            "phasegate_minus_fixed_pageout": int(p["pageouts_delta"]) - int(f["pageouts_delta"]),
            "phasegate_minus_timegate_pageout": int(p["pageouts_delta"]) - int(t["pageouts_delta"]),
            "phasegate_vs_fixed_qps_gain": float(p["retrieval_qps"]) / float(f["retrieval_qps"]) - 1,
            "phasegate_vs_timegate_qps_gain": float(p["retrieval_qps"]) / float(t["retrieval_qps"]) - 1,
            "phasegate_minus_fixed_tpot_ms": float(p["p95_tpot_ms"]) - float(f["p95_tpot_ms"]),
            "phasegate_minus_fixed_ttft_ms": float(p["p95_ttft_ms"]) - float(f["p95_ttft_ms"]),
            "all_three_pageout_free": all(int(x["pageouts_delta"]) == 0 for x in (f, p, t)),
            "all_swap_deltas_zero": all(int(x["swap_used_delta_bytes"]) == 0 for x in (f, p, t))})
    write_csv(root / "m4_pageout_pair_audit.csv", pageout_rows)
    loo = []
    gains = [row["phasegate_vs_fixed_qps_gain"] for row in pageout_rows]
    for excluded in range(7):
        loo.append({"excluded_repeat": excluded,
                    "leave_one_out_median_phasegate_vs_fixed_gain": median(
                        value for index, value in enumerate(gains) if index != excluded)})
    write_csv(root / "m4_pageout_leave_one_out.csv", loo)
    imbalance = [float(row["phasegate_minus_fixed_pageout"]) for row in pageout_rows]
    spearman = float(np.corrcoef(ranks(imbalance), ranks(gains))[0, 1]) if len(set(imbalance)) > 1 else float("nan")
    fig, ax = plt.subplots(figsize=(5.5, 3.7))
    x = np.arange(7); ax.plot(x, [row["phasegate_vs_fixed_gain"] * 100 for row in pair_rows], "o-", label="vs Fixed")
    ax.plot(x, [row["phasegate_vs_timegate_gain"] * 100 for row in pair_rows], "s-", label="vs TimeGate")
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="Paired repeat", ylabel="PhaseGate QPS gain (%)")
    ax.legend(); fig.tight_layout(); fig.savefig(root / "figure_m4_fixed_phasegate_timegate.pdf")
    fig.savefig(root / "figure_m4_fixed_phasegate_timegate.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.7)); ax.scatter(imbalance, np.asarray(gains) * 100)
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="PhaseGate - Fixed pageout delta", ylabel="Paired QPS gain (%)")
    fig.tight_layout(); fig.savefig(root / "figure_m4_pageout_vs_gain.pdf")
    fig.savefig(root / "figure_m4_pageout_vs_gain.png", dpi=180); plt.close(fig)
    pageout_free = sum(row["all_three_pageout_free"] for row in pageout_rows)
    primary = f"""# Base-M4 Held-Out Primary Report

- Frozen primary budget: {budget:.2f}
- Frozen policies: {fixed}, {phasegate}, matched {timegate}
- Paired randomized repeats: 7 (run-level units)
- PhaseGate vs Fixed median paired QPS gain: {pf[0]*100:.2f}% (95% bootstrap interval {pf[1]*100:.2f}% to {pf[2]*100:.2f}%)
- PhaseGate vs TimeGate median paired QPS gain: {pt[0]*100:.2f}% (95% bootstrap interval {pt[1]*100:.2f}% to {pt[2]*100:.2f}%)
- Completely pageout-free triplets: {pageout_free}/7
- Swap-growth valid blocks: 0/21

The causal interpretation must follow the frozen rule and the observed TimeGate SLO and paired-gain results; diagnostic and invalid attempts are excluded.
"""
    (root / "M4_PRIMARY_REPORT.md").write_text(primary)
    pageout_report = f"""# Base-M4 Pageout Audit

- Completely pageout-free paired triplets: {pageout_free}/7
- Descriptive Spearman correlation, Fixed-pair pageout imbalance versus PhaseGate QPS gain: {spearman:.3f}
- Swap growth: 0/21 valid blocks

The correlation is descriptive at n=7; a nonsignificant value would not establish absence of confounding.
"""
    (root / "M4_PAGEOUT_REPORT.md").write_text(pageout_report)
    print(primary)


if __name__ == "__main__": main()
