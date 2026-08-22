#!/usr/bin/env python3
"""Predeclared run-level analysis and compact bundling for bursty HNSW."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: raise RuntimeError(f"refusing empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap(values: list[float], seed: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    draws = np.median(array[rng.integers(0, len(array), size=(10_000, len(array)))], axis=1)
    return float(np.median(array)), float(np.percentile(draws, 2.5)), float(
        np.percentile(draws, 97.5))


def selected_runs(campaign: Path) -> list[dict[str, Any]]:
    selection = json.loads((campaign / "PRIMARY_TRIPLET_SELECTION.json").read_text())
    keys = {key for triplet in selection["selected"] for key in triplet["run_keys"]}
    rows = [row for row in read_jsonl(campaign / "primary/raw/runs.jsonl")
            if row.get("run_key") in keys]
    if len(rows) != 45: raise RuntimeError(f"expected 45 selected rows, found {len(rows)}")
    return rows


def duty(row: dict[str, Any]) -> int:
    return round(float(row["bursty_demand_audit"]["demand_level"]) * 100)


def export_runs(campaign: Path, rows: list[dict[str, Any]]) -> None:
    output = []
    demand_rows = []
    overlap_rows = []
    pageout_rows = []
    for row in sorted(rows, key=lambda item: (duty(item), int(item["repeat"]), item["policy"])):
        audit = row["bursty_demand_audit"]
        output.append({"demand_duty_pct": duty(row), "repeat": row["repeat"],
            "policy": row["policy"], "run_key": row["run_key"], "status": row["status"],
            "duration_s": row["duration_s"],
            "retrieval_qps_full_wall_time": audit["retrieval_qps_full_wall_time"],
            "retrieval_qps_scheduled_on_time": audit["retrieval_qps_scheduled_on_time"],
            "completed_queries_per_llm_request": audit["completed_queries_per_llm_request"],
            "normalized_p95_tpot": row["normalized_p95_tpot"],
            "normalized_p95_ttft": row["normalized_p95_ttft"],
            "joint_slo_pass": row["joint_slo_pass"],
            "scheduled_on_fraction": audit["scheduled_on_fraction"],
            "actual_hnsw_active_fraction": audit["actual_hnsw_active_fraction"],
            "prefill_active_worker_time_s": audit["prefill_active_worker_time_s"],
            "decode_active_worker_time_s": audit["decode_active_worker_time_s"],
            "cap_binding_fraction_during_scheduled_on": audit["cap_binding_fraction_during_scheduled_on"],
            "mean_on_off_to_zero_active_s": audit["mean_on_off_to_zero_active_s"],
            "p95_on_off_to_zero_active_s": audit["p95_on_off_to_zero_active_s"],
            "pageouts_delta": row["pageouts_delta"],
            "swap_used_delta_bytes": row["swap_used_delta_bytes"],
            "memory_pressure_clean": row["memory_pressure_clean"],
            "power_clean": row["power_clean"], "pageout_soft_flag": row["pageout_soft_flag"],
            "thermal_proxy_clean": row["validity"]["sentinel_clean"],
            "observer_subprocess_count": row["observer_subprocess_count_during_block"]})
        demand_rows.append({"run_key": row["run_key"], "repeat": row["repeat"],
            "policy": row["policy"], **audit})
        for overlap in row["bursty_phase_overlap"]:
            overlap_rows.append({"run_key": row["run_key"], "repeat": row["repeat"],
                "demand_duty_pct": duty(row), "policy": row["policy"], **overlap})
        pageout_rows.append({"run_key": row["run_key"], "repeat": row["repeat"],
            "demand_duty_pct": duty(row), "policy": row["policy"],
            "pageouts_delta": row["pageouts_delta"], "pageins_delta": row["pageins_delta"],
            "swap_used_delta_bytes": row["swap_used_delta_bytes"],
            "memory_pressure_clean": row["memory_pressure_clean"],
            "pageout_soft_flag": row["pageout_soft_flag"],
            "power_clean": row["power_clean"],
            "thermal_proxy_clean": row["validity"]["sentinel_clean"],
            "compressions_delta": row["compressions_delta"],
            "decompressions_delta": row["decompressions_delta"]})
    write_csv(campaign / "bursty_load_runs.csv", output)
    write_csv(campaign / "bursty_load_demand_audit.csv", demand_rows)
    write_csv(campaign / "bursty_load_phase_overlap.csv", overlap_rows)
    write_csv(campaign / "bursty_load_pageout_audit.csv", pageout_rows)

    objects = {item["run_key"]: item["requests"] for item in
               read_jsonl(campaign / "primary/raw/requests.jsonl")}
    request_rows = []
    for row in output:
        for index, item in enumerate(objects[row["run_key"]]):
            request_rows.append({"run_key": row["run_key"], "repeat": row["repeat"],
                "demand_duty_pct": row["demand_duty_pct"], "policy": row["policy"],
                "request_index": index, "request_id": item["request_id"],
                "ttft_ms": item["ttft_ms"], "mean_tpot_ms": item["mean_tpot_ms"],
                "p50_tpot_ms": item["p50_tpot_ms"], "p95_tpot_ms": item["p95_tpot_ms"],
                "p99_tpot_ms": item["p99_tpot_ms"],
                "token_timestamp_count": len(item["token_timestamps"])})
    write_csv(campaign / "bursty_load_request_metrics.csv", request_rows)


def pairwise_and_summary(campaign: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by = {(duty(row), int(row["repeat"]), row["policy"]): row for row in rows}
    pairs = []
    summaries = []
    for demand in (5, 25, 100):
        duty_pairs = []
        for repeat in range(5):
            fixed = by[(demand, repeat, "fixed1")]
            phase = by[(demand, repeat, "phasegate4to1")]
            timed = by[(demand, repeat, "timegate4to1")]
            f_qps = float(fixed["bursty_demand_audit"]["retrieval_qps_full_wall_time"])
            p_qps = float(phase["bursty_demand_audit"]["retrieval_qps_full_wall_time"])
            t_qps = float(timed["bursty_demand_audit"]["retrieval_qps_full_wall_time"])
            duty_pairs.append({"demand_duty_pct": demand, "repeat": repeat,
                "fixed_qps": f_qps, "phasegate_qps": p_qps, "timegate_qps": t_qps,
                "phasegate_fixed_ratio": p_qps / f_qps,
                "phasegate_timegate_ratio": p_qps / t_qps,
                "phasegate_minus_fixed_qps": p_qps - f_qps,
                "fixed_joint_slo_pass": fixed["joint_slo_pass"],
                "phasegate_joint_slo_pass": phase["joint_slo_pass"],
                "timegate_joint_slo_pass": timed["joint_slo_pass"],
                "phasegate_minus_fixed_slo_pass": int(phase["joint_slo_pass"]) - int(fixed["joint_slo_pass"]),
                "pageout_clean_pair": not bool(phase["pageout_soft_flag"])
                    and not bool(fixed["pageout_soft_flag"])})
        pf = bootstrap([item["phasegate_fixed_ratio"] for item in duty_pairs], 440000 + demand)
        pt = bootstrap([item["phasegate_timegate_ratio"] for item in duty_pairs], 450000 + demand)
        absolute = bootstrap([item["phasegate_minus_fixed_qps"] for item in duty_pairs], 460000 + demand)
        clean_ratios = [item["phasegate_fixed_ratio"] for item in duty_pairs
                        if item["pageout_clean_pair"]]
        ranges = {
            "phasegate_fixed_pair_min": min(item["phasegate_fixed_ratio"] for item in duty_pairs),
            "phasegate_fixed_pair_max": max(item["phasegate_fixed_ratio"] for item in duty_pairs),
            "phasegate_timegate_pair_min": min(item["phasegate_timegate_ratio"] for item in duty_pairs),
            "phasegate_timegate_pair_max": max(item["phasegate_timegate_ratio"] for item in duty_pairs),
            "phasegate_minus_fixed_qps_min": min(item["phasegate_minus_fixed_qps"] for item in duty_pairs),
            "phasegate_minus_fixed_qps_max": max(item["phasegate_minus_fixed_qps"] for item in duty_pairs),
            "pageout_clean_pair_count": len(clean_ratios),
            "pageout_clean_phasegate_fixed_ratio_median": median(clean_ratios) if clean_ratios else None,
        }
        for item in duty_pairs:
            item.update({"phasegate_fixed_median_ratio": pf[0], "phasegate_fixed_ci_low": pf[1],
                "phasegate_fixed_ci_high": pf[2], "phasegate_timegate_median_ratio": pt[0],
                "phasegate_timegate_ci_low": pt[1], "phasegate_timegate_ci_high": pt[2]})
            item.update(ranges)
        pairs.extend(duty_pairs)
        for policy in ("fixed1", "phasegate4to1", "timegate4to1"):
            selected = [by[(demand, repeat, policy)] for repeat in range(5)]
            summaries.append({"demand_duty_pct": demand, "policy": policy,
                "median_retrieval_qps_full_wall_time": median(float(item["bursty_demand_audit"]["retrieval_qps_full_wall_time"]) for item in selected),
                "median_retrieval_qps_scheduled_on_time": median(float(item["bursty_demand_audit"]["retrieval_qps_scheduled_on_time"]) for item in selected),
                "median_completed_queries_per_llm_request": median(float(item["bursty_demand_audit"]["completed_queries_per_llm_request"]) for item in selected),
                "median_normalized_p95_tpot": median(float(item["normalized_p95_tpot"]) for item in selected),
                "median_normalized_p95_ttft": median(float(item["normalized_p95_ttft"]) for item in selected),
                "joint_slo_passes": sum(bool(item["joint_slo_pass"]) for item in selected),
                "median_scheduled_on_fraction": median(float(item["bursty_demand_audit"]["scheduled_on_fraction"]) for item in selected),
                "median_actual_hnsw_active_fraction": median(float(item["bursty_demand_audit"]["actual_hnsw_active_fraction"]) for item in selected),
                "median_prefill_active_worker_time_s": median(float(item["bursty_demand_audit"]["prefill_active_worker_time_s"]) for item in selected),
                "median_decode_active_worker_time_s": median(float(item["bursty_demand_audit"]["decode_active_worker_time_s"]) for item in selected),
                "median_cap_binding_fraction_scheduled_on": median(float(item["bursty_demand_audit"]["cap_binding_fraction_during_scheduled_on"]) for item in selected),
                "median_on_off_to_zero_active_s": median(float(item["bursty_demand_audit"]["mean_on_off_to_zero_active_s"]) for item in selected),
                "median_p95_on_off_to_zero_active_s": median(float(item["bursty_demand_audit"]["p95_on_off_to_zero_active_s"]) for item in selected),
                "pageout_soft_flag_runs": sum(bool(item["pageout_soft_flag"]) for item in selected),
                "swap_growth_runs": sum(int(item["swap_used_delta_bytes"] or 0) != 0 for item in selected),
                "memory_pressure_clean_runs": sum(bool(item["memory_pressure_clean"]) for item in selected),
                "power_clean_runs": sum(bool(item["power_clean"]) for item in selected),
                "thermal_proxy_clean_runs": sum(bool(item["validity"]["sentinel_clean"]) for item in selected),
                "phasegate_fixed_ratio_median": pf[0] if policy == "phasegate4to1" else None,
                "phasegate_fixed_ratio_ci_low": pf[1] if policy == "phasegate4to1" else None,
                "phasegate_fixed_ratio_ci_high": pf[2] if policy == "phasegate4to1" else None,
                "phasegate_timegate_ratio_median": pt[0] if policy == "phasegate4to1" else None,
                "phasegate_timegate_ratio_ci_low": pt[1] if policy == "phasegate4to1" else None,
                "phasegate_timegate_ratio_ci_high": pt[2] if policy == "phasegate4to1" else None,
                "phasegate_minus_fixed_qps_median": absolute[0] if policy == "phasegate4to1" else None,
                "phasegate_minus_fixed_qps_ci_low": absolute[1] if policy == "phasegate4to1" else None,
                "phasegate_minus_fixed_qps_ci_high": absolute[2] if policy == "phasegate4to1" else None,
                "phasegate_fixed_slo_pass_count_contrast": (
                    sum(int(item["phasegate_joint_slo_pass"]) for item in duty_pairs)
                    - sum(int(item["fixed_joint_slo_pass"]) for item in duty_pairs)
                    if policy == "phasegate4to1" else None),
                **({key: value for key, value in ranges.items()}
                   if policy == "phasegate4to1" else {key: None for key in ranges})})
    write_csv(campaign / "bursty_load_pairwise.csv", pairs)
    write_csv(campaign / "bursty_load_summary.csv", summaries)
    return pairs, summaries


def figures(campaign: Path, summaries: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    baseline = json.loads((campaign / "bursty_load_normalization_baseline.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for policy, marker in (("fixed1", "o"), ("phasegate4to1", "s"), ("timegate4to1", "^")):
        chosen = sorted([row for row in summaries if row["policy"] == policy], key=lambda row: row["demand_duty_pct"])
        x = [0] + [row["demand_duty_pct"] for row in chosen]
        ax.plot(x, [1.0] + [row["median_normalized_p95_tpot"] for row in chosen], marker=marker,
                label=f"{policy} TPOT")
        ax.plot(x, [1.0] + [row["median_normalized_p95_ttft"] for row in chosen], marker=marker,
                linestyle="--", label=f"{policy} TTFT")
    ax.axhline(1.25, color="black", linestyle=":", label="B=1.25")
    ax.set(xlabel="Scheduled demand duty (%)", ylabel="Normalized p95 latency",
           title="(a) LLM latency versus intermittent HNSW demand")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(campaign / "figure_bursty_latency_vs_duty.pdf")
    fig.savefig(campaign / "figure_bursty_latency_vs_duty.png", dpi=180); plt.close(fig)

    fig, left = plt.subplots(figsize=(7.2, 4.4)); right = left.twinx()
    for policy, marker in (("fixed1", "o"), ("phasegate4to1", "s"), ("timegate4to1", "^")):
        chosen = sorted([row for row in summaries if row["policy"] == policy], key=lambda row: row["demand_duty_pct"])
        left.plot([row["demand_duty_pct"] for row in chosen],
                  [row["median_retrieval_qps_full_wall_time"] for row in chosen],
                  marker=marker, label=policy)
    grouped = defaultdict(list)
    for row in pairs: grouped[row["demand_duty_pct"]].append(row["phasegate_fixed_ratio"])
    x = sorted(grouped); ratios = [median(grouped[value]) for value in x]
    right.plot(x, ratios, "D-", color="black", label="PhaseGate / Fixed")
    right.axhline(1.0, color="gray", linestyle=":")
    left.set(xlabel="Scheduled demand duty (%)", ylabel="Retrieval QPS (full wall time)",
             title="(b) Retrieval capacity versus intermittent demand")
    right.set_ylabel("Paired PhaseGate / Fixed ratio")
    lines = left.lines + right.lines[:1]
    left.legend(lines, [line.get_label() for line in lines], fontsize=8)
    left.grid(alpha=.2); fig.tight_layout()
    fig.savefig(campaign / "figure_bursty_throughput_vs_duty.pdf")
    fig.savefig(campaign / "figure_bursty_throughput_vs_duty.png", dpi=180); plt.close(fig)
    del baseline


def reports(campaign: Path, pairs: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    phase_rows = {row["demand_duty_pct"]: row for row in summaries
                  if row["policy"] == "phasegate4to1"}
    fixed_rows = {row["demand_duty_pct"]: row for row in summaries
                  if row["policy"] == "fixed1"}
    pair_by_duty = defaultdict(list)
    for row in pairs: pair_by_duty[row["demand_duty_pct"]].append(row)
    lines = ["# Base-M4 Bursty HNSW Demand Sweep Final Report", "",
        "This is a synthetic workload-sensitivity study, not an end-to-end agent benchmark or a claim about typical-user arrival rates.", "",
        "## Frozen scope", "", "Fixed-1, PhaseGate 4->1, TimeGate 4/1, B=1.25, 2,048/128 tokens, one shared 100k-vector HNSW index, five matched repeats per duty.", "",
        "## Results", "", "| Duty | PhaseGate QPS | Fixed QPS | PhaseGate/Fixed [95% paired bootstrap] | PhaseGate SLO | Fixed SLO |", "|---:|---:|---:|---:|---:|---:|"]
    for demand in (5, 25, 100):
        pair = pair_by_duty[demand][0]; phase = phase_rows[demand]; fixed = fixed_rows[demand]
        lines.append(f"| {demand}% | {phase['median_retrieval_qps_full_wall_time']:.2f} | {fixed['median_retrieval_qps_full_wall_time']:.2f} | {pair['phasegate_fixed_median_ratio']:.3f} [{pair['phasegate_fixed_ci_low']:.3f}, {pair['phasegate_fixed_ci_high']:.3f}] | {phase['joint_slo_passes']}/5 | {fixed['joint_slo_passes']}/5 |")
    ratios = [median(item["phasegate_fixed_ratio"] for item in pair_by_duty[demand])
              for demand in (5, 25, 100)]
    if ratios[0] <= 1.0 and ratios[-1] > ratios[0]:
        interpretation = "The median PhaseGate advantage is absent at 5% and grows with duty; the scheduler is most useful when independent retrieval accumulates."
    elif all(value > 1.0 for value in ratios[:2]):
        interpretation = "A positive median PhaseGate/Fixed ratio persists under the tested synthetic 5% and 25% intermittent-demand traces."
    elif ratios[-1] > 1.0 and all(value <= 1.0 for value in ratios[:2]):
        interpretation = "Only the saturated point has a positive median PhaseGate/Fixed ratio; the lower-duty results define a scope boundary for the capacity-bound claim."
    else:
        interpretation = "PhaseGate does not show a consistent retrieval-capacity advantage across the tested duties."
    lines += ["", "## All matched values", ""]
    for demand in (5, 25, 100):
        values = pair_by_duty[demand]
        fixed_values = ", ".join(f"{item['phasegate_fixed_ratio']:.3f}" for item in values)
        timed_values = ", ".join(f"{item['phasegate_timegate_ratio']:.3f}" for item in values)
        absolute_values = ", ".join(f"{item['phasegate_minus_fixed_qps']:.2f}" for item in values)
        lines += [f"- {demand}% PhaseGate/Fixed: {fixed_values} "
                  f"(min--max {min(item['phasegate_fixed_ratio'] for item in values):.3f}--"
                  f"{max(item['phasegate_fixed_ratio'] for item in values):.3f})",
                  f"- {demand}% PhaseGate/TimeGate: {timed_values} "
                  f"(min--max {min(item['phasegate_timegate_ratio'] for item in values):.3f}--"
                  f"{max(item['phasegate_timegate_ratio'] for item in values):.3f})",
                  f"- {demand}% PhaseGate-minus-Fixed QPS: {absolute_values} "
                  f"(min--max {min(item['phasegate_minus_fixed_qps'] for item in values):.2f}--"
                  f"{max(item['phasegate_minus_fixed_qps'] for item in values):.2f})"]
    lines += ["", "## Pageout sensitivity", ""]
    for demand in (5, 25, 100):
        values = pair_by_duty[demand]
        clean = [item["phasegate_fixed_ratio"] for item in values
                 if item["pageout_clean_pair"]]
        lines.append(f"- {demand}%: {len(clean)}/5 pairs have no pageout soft flag; "
                     + (f"clean-pair median ratio {median(clean):.3f}." if clean else
                        "no clean-pair median is estimable."))
    timegate_passes = [next(row["joint_slo_passes"] for row in summaries
                            if row["demand_duty_pct"] == demand
                            and row["policy"] == "timegate4to1")
                       for demand in (5, 25, 100)]
    timegate_statement = (f"TimeGate joint-SLO passes across 5%, 25%, and 100% duty were "
                          f"{timegate_passes[0]}/5, {timegate_passes[1]}/5, and "
                          f"{timegate_passes[2]}/5, respectively.")
    lines += ["", "## Interpretation", "", interpretation, timegate_statement,
              "The three duty points do not establish a continuous causal law and do not generalize to embedding, indexing, file, or network tools.", "",
              "## Integrity and validity", "", "All SLO failures and unfavorable pairwise values are retained. Primary confidence intervals use 10,000 paired resamples of the five run-level comparisons, never individual requests."]
    (campaign / "M4_BURSTY_HNSW_FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    recommendations = """# M4 Bursty HNSW Paper Update Recommendations

- Present this campaign only as a synthetic intermittent-demand sensitivity curve.
- Plot the LLM-only 0% baseline only as a latency anchor, not a throughput policy result.
- Report all three duty points, all five paired values per duty, SLO pass counts, and paired run-level intervals.
- Do not describe 5% or 25% as a measured typical-agent HNSW arrival rate.
- Do not claim end-to-end task acceleration or generalization to heterogeneous tools.
- Keep the paper unchanged until these recommendations are reviewed separately.
"""
    (campaign / "M4_BURSTY_HNSW_PAPER_UPDATE_RECOMMENDATIONS.md").write_text(recommendations)


def raw_checksums(campaign: Path) -> None:
    paths = sorted(path for path in campaign.rglob("*") if path.is_file()
                   and ("/raw/" in str(path) or path.name.endswith(".jsonl")))
    rows = [{"relative_path": str(path.relative_to(campaign)), "size_bytes": path.stat().st_size,
             "sha256": sha256(path)} for path in paths]
    write_csv(campaign / "raw_event_checksums.csv", rows)


def bundle(campaign: Path) -> None:
    deliverables = [
        "M4_BURSTY_HNSW_FINAL_REPORT.md", "M4_BURSTY_HNSW_PAPER_UPDATE_RECOMMENDATIONS.md",
        "BURSTY_LOAD_PROTOCOL_FREEZE.json", "BURSTY_DEMAND_SEMANTIC_AUDIT.md",
        "HARNESS_CHANGELOG.md", "machine_manifest.json", "bursty_load_baseline_runs.csv",
        "bursty_load_baseline_request_metrics.csv", "bursty_load_runs.csv",
        "bursty_load_request_metrics.csv", "bursty_load_demand_audit.csv",
        "bursty_load_phase_overlap.csv", "bursty_load_pairwise.csv", "bursty_load_summary.csv",
        "bursty_load_pageout_audit.csv", "figure_bursty_latency_vs_duty.pdf",
        "figure_bursty_latency_vs_duty.png", "figure_bursty_throughput_vs_duty.pdf",
        "figure_bursty_throughput_vs_duty.png", "raw_event_checksums.csv", "test_output.txt",
        "timegate_schedule_freeze.json", "PRIMARY_TRIPLET_SELECTION.json",
        "MIDPOINT_RECHECK_RESULT.json", "PRIMARY_COLLECTION_COMPLETE.json",
        "bursty_load_normalization_baseline.json", "SMOKE_PROTOCOL_FREEZE.json",
        "MECHANICS_SMOKE_RESULT.json", "mechanics_smoke_audit.csv",
        "COMPATIBILITY_PROTOCOL_FREEZE.json", "COMPATIBILITY_RESULT.json",
        "compatibility_pairwise.csv", "BASELINE_A_RESULT.json",
        "BASELINE_B_RESULT.json", "BASELINE_B_ENVIRONMENTAL_CORRECTION.json"]
    scripts = ["scripts/run_m4_bursty_campaign.py", "scripts/analyze_m4_bursty_campaign.py",
               "scripts/run_m4_bursty_smoke.py", "scripts/generate_bursty_demand_trace.py",
               "scripts/test_bursty_demand_semantics.py", "scripts/run_static_phaseaware_pilot.py",
               "src/phaseguard/demand_gate.py", "src/phaseguard/bursty_audit.py",
               "src/phaseguard/shared_index_manager.py", "src/phaseguard/observer.py"]
    output = campaign / "m4_bursty_results_bundle.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in deliverables:
            path = campaign / name
            if path.exists(): archive.write(path, arcname=name)
        for name in scripts:
            archive.write(REPO / name, arcname=f"implementation/{name}")
        for path in sorted((campaign / "frozen_traces").glob("*.json")):
            archive.write(path, arcname=f"frozen_traces/{path.name}")
    digest = sha256(output)
    (campaign / "m4_bursty_results_bundle.sha256").write_text(
        f"{digest}  {output.name}\n")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_m4_bursty_campaign.py CAMPAIGN")
    campaign = Path(sys.argv[1]).resolve(); rows = selected_runs(campaign)
    export_runs(campaign, rows)
    pairs, summaries = pairwise_and_summary(campaign, rows)
    figures(campaign, summaries, pairs); reports(campaign, pairs, summaries)
    raw_checksums(campaign); bundle(campaign)


if __name__ == "__main__":
    main()
