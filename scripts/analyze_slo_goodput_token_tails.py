#!/usr/bin/env python3
"""Analyze the frozen held-out SLO-goodput campaign at the run level."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slo_goodput_common import add_args, configure_root, resolve, write_csv

BOOTSTRAP_SEED = 820260801


def median(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def valid_by_policy(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "valid":
            grouped[str(row["policy"])].append(row)
    return grouped


def boot_ci(values: list[float]) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    array = np.asarray(values, dtype=float)
    estimates = np.median(rng.choice(array, (10_000, len(array)), replace=True), axis=1)
    return tuple(float(x) for x in np.percentile(estimates, [2.5, 97.5]))


def robust(rows: list[dict[str, Any]], budget: float) -> tuple[bool, int]:
    count = sum(float(r["normalized_p95_tpot"]) <= budget and
                float(r["normalized_p95_ttft"]) <= budget for r in rows)
    needed = 6 if len(rows) == 7 else len(rows)
    return len(rows) in (5, 7) and count >= needed, count


def selected_name(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    return None if value is None else str(value["policy"])


def request_records(campaign: Path) -> dict[str, list[dict[str, Any]]]:
    path = campaign / "evaluation/raw/requests.jsonl"
    output: dict[str, list[dict[str, Any]]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            output[str(record["run_key"])] = record.get("requests", [])
    return output


def token_records(campaign: Path) -> dict[str, list[dict[str, Any]]]:
    path = campaign / "raw/token_timestamps.jsonl.gz"
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if path.exists():
        with gzip.open(path, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("stage") == "evaluation":
                    output[str(row["run_key"])].append(row)
    return output


def savefig(fig: Any, root: Path, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); add_args(parser)
    args = parser.parse_args(); campaign = resolve(args.campaign_dir)
    pilot = configure_root(campaign)
    frozen = json.loads((campaign / "frozen_slo_selection.json").read_text())
    baseline = frozen["baseline"]
    eval_all = pilot.read_jsonl(pilot.raw_path("evaluation", False))
    cal_all = pilot.read_jsonl(pilot.raw_path("calibration", False))
    groups = valid_by_policy(eval_all)
    requests = request_records(campaign); tokens = token_records(campaign)

    invalid = [r for stage in ("smoke", "token_logging_overhead", "isolated_baseline",
               "calibration", "baseline_revalidation", "evaluation")
               for r in pilot.read_jsonl(pilot.raw_path(stage, stage == "smoke"))
               if r.get("status") != "valid"]
    write_csv(campaign / "invalid_runs.csv", invalid)

    summary = []
    for policy, rows in sorted(groups.items()):
        summary.append({"policy": policy, "valid_repeats": len(rows),
            "median_normalized_p95_tpot": median(rows, "normalized_p95_tpot"),
            "median_normalized_p95_ttft": median(rows, "normalized_p95_ttft"),
            "worst_normalized_p95_tpot": max(float(r["normalized_p95_tpot"]) for r in rows),
            "worst_normalized_p95_ttft": max(float(r["normalized_p95_ttft"]) for r in rows),
            "median_retrieval_qps": median(rows, "total_retrieval_goodput_qps"),
            "median_p99_itg_ms": median(rows, "p99_inter_token_gap_ms"),
            "median_transition_p95_ms": median(rows, "p95_transition_gap_ms"),
            "pageout_invalid_attempts": sum(int(r.get("pageouts_delta", 0)) > 0 for r in eval_all
                                             if r.get("policy") == policy),
            "swap_invalid_attempts": sum(int(r.get("swap_used_delta_bytes") or 0) > 0 for r in eval_all
                                          if r.get("policy") == policy)})
    write_csv(campaign / "evaluation_summary_by_policy.csv", summary)

    slo_rows, paired_rows = [], []
    for selection in frozen["selection_by_slo"]:
        budget = float(selection["B"])
        fixed, gate = selected_name(selection, "continuous_fixed"), selected_name(selection, "continuous_phasegate")
        fr, gr = groups.get(fixed or "", []), groups.get(gate or "", [])
        fok, fpasses = robust(fr, budget) if fr else (False, 0)
        gok, gpasses = robust(gr, budget) if gr else (False, 0)
        gains = []
        if fr and gr:
            fb, gb = ({int(r["repeat"]): r for r in fr}, {int(r["repeat"]): r for r in gr})
            for repeat in sorted(set(fb) & set(gb)):
                gain = float(gb[repeat]["total_retrieval_goodput_qps"]) / float(
                    fb[repeat]["total_retrieval_goodput_qps"]) - 1
                gains.append(gain); paired_rows.append({"B": budget, "repeat": repeat,
                    "fixed_policy": fixed, "gate_policy": gate,
                    "fixed_qps": fb[repeat]["total_retrieval_goodput_qps"],
                    "gate_qps": gb[repeat]["total_retrieval_goodput_qps"], "paired_gain": gain})
        slo_rows.append({"B": budget, "continuous_fixed": fixed, "fixed_joint_passes": fpasses,
            "fixed_valid_repeats": len(fr), "fixed_robust": fok,
            "fixed_qps": median(fr, "total_retrieval_goodput_qps") if fr else None,
            "continuous_phasegate": gate, "gate_joint_passes": gpasses,
            "gate_valid_repeats": len(gr), "gate_robust": gok,
            "gate_qps": median(gr, "total_retrieval_goodput_qps") if gr else None,
            "median_paired_gain": float(np.median(gains)) if fok and gok and gains else None})
    write_csv(campaign / "evaluation_summary_by_slo.csv", slo_rows)
    write_csv(campaign / "paired_slo_gains.csv", paired_rows)

    attainment = []
    for selection in frozen["selection_by_slo"]:
        budget = float(selection["B"]); tlim = float(baseline["p95_tpot_ms"]) * budget
        flim = float(baseline["p95_ttft_ms"]) * budget
        names = {selected_name(selection, k) for k in
                 ("continuous_fixed", "continuous_phasegate", "decode_zero_phasegate")}
        names.add("fixed0")
        for name in sorted(x for x in names if x):
            values = [q for run in groups.get(name, []) for q in requests.get(str(run["run_key"]), [])]
            passed = sum(float(q["mean_tpot_ms"]) <= tlim and float(q["user_visible_ttft_ms"]) <= flim
                         for q in values)
            attainment.append({"B": budget, "policy": name, "requests": len(values),
                                "joint_request_passes": passed,
                                "joint_request_attainment": passed / len(values) if values else None})
    write_csv(campaign / "request_slo_attainment.csv", attainment)

    tail, mechanism, samples = [], [], []
    for policy, rows in sorted(groups.items()):
        for row in rows:
            key = str(row["run_key"])
            tail.append({k: row.get(k) for k in ("run_key", "policy", "repeat",
                "p95_inter_token_gap_ms", "p99_inter_token_gap_ms", "max_inter_token_gap_ms",
                "p95_transition_gap_ms", "p99_transition_gap_ms",
                "p95_first_four_decode_gap_ms", "p99_first_four_decode_gap_ms",
                "gap_fraction_over_2x_isolated_median", "gap_fraction_over_3x_isolated_median",
                "per_request_max_gap_p50_ms", "per_request_max_gap_p95_ms",
                "per_request_max_gap_p99_ms")})
            mechanism.append({k: row.get(k) for k in ("run_key", "policy", "repeat",
                "prefill_active_retrieval_worker_mean", "prefill_active_retrieval_worker_p95",
                "decode_active_retrieval_worker_mean", "decode_active_retrieval_worker_p95",
                "admitted_queries_prefill", "admitted_queries_decode", "completed_queries_prefill",
                "completed_queries_decode", "cap_binding_fraction", "phase_transition_to_cap_ms",
                "phase_transition_to_cap_max_ms", "decode_cap_overshoot_fraction",
                "decode_cap_overshoot_worker_mean", "decode_cap_overshoot_worker_max")})
            samples.append({"run_key": key, "policy": policy, "repeat": row["repeat"],
                "requests": row["llm_requests"], "expected_requests": 250,
                "generated_tokens": row["total_generated_tokens"], "expected_tokens": 32000,
                "token_gaps": row["inter_token_gap_count"], "expected_token_gaps": 31750,
                "raw_request_records": len(requests.get(key, [])),
                "raw_token_records": len(tokens.get(key, [])),
                "valid": int(row["llm_requests"]) == 250 and int(row["total_generated_tokens"]) == 32000
                         and int(row["inter_token_gap_count"]) == 31750})
    write_csv(campaign / "token_tail_metrics.csv", tail)
    write_csv(campaign / "mechanism_validation.csv", mechanism)
    write_csv(campaign / "sample_size_validation.csv", samples)
    write_csv(campaign / "evaluation_runs.csv", eval_all)

    figures = campaign / "figures"; grid = [float(r["B"]) for r in slo_rows]
    fig, ax = plt.subplots(); ax.plot(grid, [r["fixed_qps"] for r in slo_rows], "o-", label="Fixed")
    ax.plot(grid, [r["gate_qps"] for r in slo_rows], "s-", label="PhaseGate")
    ax.set(xlabel="Joint SLO budget B", ylabel="Retrieval QPS"); ax.legend(); savefig(fig, figures, "figure1_slo_goodput")
    fig, ax = plt.subplots(); ax.plot(grid, [r["median_paired_gain"] for r in slo_rows], "o-")
    ax.axhline(0, color="black", lw=.8); ax.set(xlabel="Joint SLO budget B", ylabel="Paired QPS gain")
    savefig(fig, figures, "figure2_phasegate_gain")
    fig, ax = plt.subplots();
    for name, vals in defaultdict(list, {p: [r["joint_request_attainment"] for r in attainment if r["policy"] == p]
                                         for p in {r["policy"] for r in attainment}}).items():
        if len(vals) == len(grid): ax.plot(grid, vals, marker="o", label=name)
    ax.set(xlabel="Joint SLO budget B", ylabel="Joint request attainment"); ax.legend(fontsize=7)
    savefig(fig, figures, "figure3_request_attainment")
    primary = float(frozen["primary_continuous_budget"])
    chosen = next(r for r in frozen["selection_by_slo"] if float(r["B"]) == primary)
    pair = [selected_name(chosen, "continuous_fixed"), selected_name(chosen, "continuous_phasegate")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for name in pair:
        rs = groups.get(name or "", []); axes[0].boxplot([float(r["p99_inter_token_gap_ms"]) for r in rs], positions=[pair.index(name)])
        axes[1].boxplot([float(r["p95_transition_gap_ms"]) for r in rs], positions=[pair.index(name)])
    for ax, title in zip(axes, ("p99 inter-token gap", "p95 transition gap")):
        ax.set_xticks(range(2), pair, rotation=15); ax.set_ylabel("ms"); ax.set_title(title)
    savefig(fig, figures, "figure4_token_tail")
    fig, ax = plt.subplots();
    for name in pair:
        rs = groups.get(name or "", []); ax.plot([int(r["repeat"]) for r in rs],
            [float(r["prefill_active_retrieval_worker_mean"]) for r in rs], "o-", label=f"{name} prefill")
    ax.set(xlabel="Paired repeat", ylabel="Mean active retrieval workers"); ax.legend()
    savefig(fig, figures, "figure5_mechanism_timeline")
    cal = valid_by_policy(cal_all); fig, ax = plt.subplots()
    for name, rs in cal.items():
        ax.scatter([float(r["normalized_p95_tpot"]) for r in rs],
                   [float(r["total_retrieval_goodput_qps"]) for r in rs],
                   marker="s" if name.startswith("phasegate") else "o", label=name)
    ax.set(xlabel="Normalized p95 TPOT", ylabel="Retrieval QPS"); ax.legend(fontsize=5, ncol=2)
    savefig(fig, figures, "figure6_calibration_operating_points")

    primary_row = next(r for r in slo_rows if float(r["B"]) == primary)
    pgains = [float(r["paired_gain"]) for r in paired_rows if float(r["B"]) == primary]
    ci = boot_ci(pgains) if pgains else (float("nan"), float("nan"))
    report = ["> At the tightest predeclared joint TPOT/TTFT SLO admitting both continuous families, " +
        ("the frozen PhaseGate achieved higher held-out retrieval goodput." if pgains and np.median(pgains) > 0
         else "the frozen PhaseGate did not establish higher held-out retrieval goodput."), "",
        "# SLO Goodput and Token-Tail Report", "", f"- Frozen SLO grid: {frozen['frozen_slo_grid']}",
        f"- Primary continuous budget: {primary:.2f}",
        f"- Fixed / PhaseGate: {primary_row['continuous_fixed']} / {primary_row['continuous_phasegate']}",
        f"- Joint pass counts: {primary_row['fixed_joint_passes']}/{primary_row['fixed_valid_repeats']} and "
        f"{primary_row['gate_joint_passes']}/{primary_row['gate_valid_repeats']}",
        f"- Median retrieval QPS: {primary_row['fixed_qps']} / {primary_row['gate_qps']}",
        f"- Median paired gain (10,000 run-level bootstrap 95% CI): {np.median(pgains) if pgains else None} "
        f"({ci[0]}, {ci[1]})", f"- PhaseGate wins: {sum(x > 0 for x in pgains)}/{len(pgains)}", "",
        "All conclusions above use only the fresh held-out evaluation. Individual requests and tokens were not "
        "treated as independent experimental repetitions. Sensitivity rows reuse policy runs and are correlated."]
    (campaign / "SLO_GOODPUT_TOKEN_TAIL_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"campaign": str(campaign), "primary_budget": primary,
                      "primary_paired_gain": float(np.median(pgains)) if pgains else None,
                      "bootstrap_ci": ci, "valid_evaluation_blocks": sum(map(len, groups.values()))}, indent=2))


if __name__ == "__main__":
    main()
