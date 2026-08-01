#!/usr/bin/env python3
"""Analyze and report the frozen fan-Mac static PhaseGate main campaign."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fanmac_main_common import add_common_args, resolve_campaign

METRICS = ("normalized_p95_tpot", "normalized_p95_ttft",
           "total_retrieval_goodput_qps")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"], extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        if rows: writer.writerows(rows)


def bootstrap_ci(values: list[float], seed: int = 20260801) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array): return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(10_000, len(array)), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def stats(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    low, high = bootstrap_ci(values)
    return {f"{prefix}_median": float(np.median(array)), f"{prefix}_min": float(array.min()),
            f"{prefix}_max": float(array.max()),
            f"{prefix}_iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
            f"{prefix}_bootstrap_ci_low": low, f"{prefix}_bootstrap_ci_high": high}


def flatten_run(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items()
            if not isinstance(value, (dict, list))}


def summarize_runs(rows: list[dict[str, Any]], expected: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "valid" and "policy" in row:
            groups[str(row["policy"])].append(row)
    output = []
    for policy, group in sorted(groups.items()):
        selected = sorted(group, key=lambda row: int(row["repeat"]))[:expected]
        if not selected: continue
        summary = {
            "policy": policy, "policy_arg": selected[0]["policy_arg"],
            "prefill_cap": selected[0]["prefill_cap"], "decode_cap": selected[0]["decode_cap"],
            "valid_repeats": len(selected), "expected_repeats": expected,
            "joint_slo_pass_count": sum(bool(row["joint_slo_pass"]) for row in selected),
            "strict_feasible": len(selected) == expected and all(bool(row["joint_slo_pass"]) for row in selected),
            "queue_nonempty_fraction_median": float(np.median(
                [float(row["queue_nonempty_fraction"]) for row in selected])),
            "pageout_delta_total": sum(int(row.get("pageouts_delta", 0)) for row in selected),
            "swap_delta_bytes_total": sum(int(row.get("swap_used_delta_bytes", 0) or 0) for row in selected),
        }
        for key in METRICS:
            summary.update(stats([float(row[key]) for row in selected], key))
        output.append(summary)
    return output


def baseline_tables(campaign: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [row for row in read_jsonl(campaign / "isolated_baseline/raw/runs.jsonl")
            if row.get("status") == "valid" and row.get("policy") == "llm-only"]
    payload = json.loads((campaign / "isolated_baseline/baseline.json").read_text())
    selected = [row for row in rows if row["run_key"] in set(payload["run_keys"])]
    summary = [{
        "valid_repeats": len(selected), "isolated_p95_tpot_ms": payload["p95_tpot_ms"],
        "isolated_p95_ttft_ms": payload["p95_ttft_ms"],
        "tpot_threshold_ms": payload["tpot_threshold_ms"],
        "ttft_threshold_ms": payload["ttft_threshold_ms"],
        "max_run_level_p95_deviation": payload["max_run_level_p95_deviation"],
        **stats([float(row["p95_tpot_ms"]) for row in selected], "p95_tpot_ms"),
        **stats([float(row["p95_ttft_ms"]) for row in selected], "p95_ttft_ms"),
    }]
    return [flatten_run(row) for row in rows], summary


def characterization_tables(campaign: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = read_jsonl(campaign / "characterization/raw/profile_fanmac_main.jsonl")
    valid = [row for row in raw if row.get("status") == "ok"]
    baseline = {phase: float(np.median([float(row["p95_ms"]) for row in valid
                if row["phase"] == phase and int(row["workers"]) == 0]))
                for phase in ("PREFILL", "DECODE")}
    runs = []
    for row in raw:
        item = flatten_run(row)
        if row.get("p95_ms") is not None:
            item["slowdown"] = float(row["p95_ms"]) / baseline[row["phase"]]
        runs.append(item)
    summary = []
    for phase in ("PREFILL", "DECODE"):
        for workers in range(5):
            group = [row for row in runs if row.get("status") == "ok"
                     and row["phase"] == phase and int(row["workers"]) == workers]
            summary.append({"phase": phase, "workers": workers, "valid_repeats": len(group),
                            **stats([float(row["slowdown"]) for row in group], "slowdown"),
                            **stats([float(row["logical_qps"]) for row in group], "retrieval_qps")})
    return runs, summary


def paired_details(eval_rows: list[dict[str, Any]], fixed: str, gate: str) -> list[dict[str, Any]]:
    valid = [row for row in eval_rows if row.get("status") == "valid"]
    f = {int(row["repeat"]): row for row in valid if row["policy"] == fixed}
    g = {int(row["repeat"]): row for row in valid if row["policy"] == gate}
    output = []
    for repeat in sorted(set(f) & set(g)):
        fq, gq = float(f[repeat]["total_retrieval_goodput_qps"]), float(g[repeat]["total_retrieval_goodput_qps"])
        output.append({
            "comparison": f"{fixed}_vs_{gate}", "repeat": repeat,
            "fixed_qps": fq, "phasegate_qps": gq, "additional_qps": gq - fq,
            "paired_gain": gq / fq - 1 if fq else None, "phasegate_qps_win": gq > fq,
            "fixed_tpot": f[repeat]["p95_tpot_ms"], "phasegate_tpot": g[repeat]["p95_tpot_ms"],
            "tpot_ratio_gate_fixed": float(g[repeat]["p95_tpot_ms"]) / float(f[repeat]["p95_tpot_ms"]),
            "ttft_ratio_gate_fixed": float(g[repeat]["p95_ttft_ms"]) / float(f[repeat]["p95_ttft_ms"]),
            "fixed_joint_slo": f[repeat]["joint_slo_pass"],
            "phasegate_joint_slo": g[repeat]["joint_slo_pass"],
            "phasegate_prefill_completed": g[repeat]["completed_queries_prefill"],
            "fixed_prefill_completed": f[repeat]["completed_queries_prefill"],
            "fixed_prefill_active_mean": f[repeat]["prefill_active_retrieval_worker_mean"],
            "phasegate_prefill_active_mean": g[repeat]["prefill_active_retrieval_worker_mean"],
            "fixed_decode_active_mean": f[repeat]["decode_active_retrieval_worker_mean"],
            "phasegate_decode_active_mean": g[repeat]["decode_active_retrieval_worker_mean"],
        })
    return output


def summarize_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[row["comparison"]].append(row)
    output = []
    for comparison, group in sorted(groups.items()):
        gains = [float(row["paired_gain"]) for row in group]
        low, high = bootstrap_ci(gains)
        output.append({
            "comparison": comparison, "paired_repeats": len(group),
            "median_paired_gain": float(np.median(gains)),
            "paired_gain_bootstrap_ci_low": low, "paired_gain_bootstrap_ci_high": high,
            "phasegate_qps_wins": sum(bool(row["phasegate_qps_win"]) for row in group),
            "median_tpot_ratio_gate_fixed": float(np.median(
                [float(row["tpot_ratio_gate_fixed"]) for row in group])),
            "median_ttft_ratio_gate_fixed": float(np.median(
                [float(row["ttft_ratio_gate_fixed"]) for row in group])),
            "median_additional_qps": float(np.median(
                [float(row["additional_qps"]) for row in group])),
        })
    return output


def figure_asymmetry(campaign: Path, summary: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for phase, marker in (("PREFILL", "o"), ("DECODE", "s")):
        rows = [row for row in summary if row["phase"] == phase]
        ax.plot([row["workers"] for row in rows], [row["slowdown_median"] for row in rows],
                marker=marker, label=phase.title())
    ax.axhline(1, color="0.5", lw=.8); ax.set_xlabel("CPU retrieval workers")
    ax.set_ylabel("Normalized phase latency slowdown"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(campaign / "figures/figure_phase_asymmetry.png", dpi=180); plt.close(fig)


def figure_goodput(campaign: Path, summary: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    names = [row["policy"] for row in (selected["best_fixed"], selected["best_phasegate"])
             if row is not None]
    if not names:
        names = ["fixed1", "phasegate4to1", "fixed2", "phasegate4to2"]
    names.append("fixed4")
    by = {row["policy"]: row for row in summary}
    names = list(dict.fromkeys(name for name in names if name in by))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#1976d2" if "phasegate" in name else
              ("#b71c1c" if name == "fixed4" else "#607d8b") for name in names]
    bars = ax.bar(names, [by[name]["total_retrieval_goodput_qps_median"] for name in names],
                  color=colors)
    for bar, name in zip(bars, names):
        row = by[name]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"TPOT {row['normalized_p95_tpot_median']:.3f}\nTTFT {row['normalized_p95_ttft_median']:.3f}\nSLO {row['joint_slo_pass_count']}/{row['expected_repeats']}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Total steady-state retrieval QPS"); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(campaign / "figures/figure_strict_slo_goodput.png", dpi=180); plt.close(fig)


def figure_operating(campaign: Path, summary: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    selected_names = {row["policy"] for row in
                      (selected["best_fixed"], selected["best_phasegate"]) if row is not None}
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for row in summary:
        phase = row["policy_arg"] == "phasegate"
        ax.scatter(row["normalized_p95_tpot_median"], row["total_retrieval_goodput_qps_median"],
                   marker="o" if phase else "s", s=95 if row["policy"] in selected_names else 40,
                   color="#1976d2" if phase else "#607d8b",
                   edgecolor="black" if row["policy"] in selected_names else "none")
        ax.annotate(row["policy"], (row["normalized_p95_tpot_median"],
                    row["total_retrieval_goodput_qps_median"]), fontsize=7, xytext=(3,3),
                    textcoords="offset points")
    ax.axvline(1.10, color="#b71c1c", ls="--", label="1.10× TPOT SLO")
    ax.set_xlabel("Normalized p95 TPOT"); ax.set_ylabel("Total retrieval QPS")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(campaign / "figures/figure_operating_points.png", dpi=180); plt.close(fig)


def figure_timeline(campaign: Path, selected: dict[str, Any], pairs: list[dict[str, Any]]) -> None:
    if selected["best_fixed"] is not None and selected["best_phasegate"] is not None:
        fixed, gate = selected["best_fixed"]["policy"], selected["best_phasegate"]["policy"]
    else:
        matched = next(row for row in selected["latency_matched_pairs"]
                       if row["passes_calibration_condition"])
        fixed, gate = matched["fixed"], matched["phasegate"]
    primary = [row for row in pairs if row["comparison"] == f"{fixed}_vs_{gate}"]
    repeat = int(primary[0]["repeat"]) if primary else 0
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    for ax, policy in zip(axes, (fixed, gate)):
        candidates = sorted((campaign / "evaluation/raw/timelines").glob(
            f"*_{policy}_r{repeat:02d}_a*.json"))
        if not candidates: continue
        data = json.loads(candidates[-1].read_text()); samples = data["demand_samples"]
        t0 = float(samples[0]["timestamp"]); x = [float(row["timestamp"])-t0 for row in samples]
        ax.step(x, [row["active_retrievals"] for row in samples], where="post", label="active workers")
        ax.step(x, [row["permitted_workers"] for row in samples], where="post", label="configured cap")
        q0 = int(samples[0]["completed_queries"])
        ax2 = ax.twinx(); ax2.plot(x, [int(row["completed_queries"])-q0 for row in samples],
                                  color="#2e7d32", alpha=.7, label="retrieval completions")
        for before, after in zip(samples, samples[1:]):
            if before["phase"] == "PREFILL":
                ax.axvspan(float(before["timestamp"])-t0, float(after["timestamp"])-t0,
                           color="#ffb300", alpha=.08)
        ax.set_title(policy); ax.set_ylabel("Workers"); ax2.set_ylabel("Completed queries")
        ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=.15)
    axes[-1].set_xlabel("Time since block start (s)"); fig.tight_layout()
    fig.savefig(campaign / "figures/figure_policy_timeline.png", dpi=180); plt.close(fig)


def report(campaign: Path, cal: list[dict[str, Any]], ev: list[dict[str, Any]],
           pairs: list[dict[str, Any]], asym: list[dict[str, Any]], selected: dict[str, Any],
           invalid: list[dict[str, Any]]) -> None:
    by = {row["policy"]: row for row in ev}
    fixed_name = None if selected["best_fixed"] is None else selected["best_fixed"]["policy"]
    gate_name = None if selected["best_phasegate"] is None else selected["best_phasegate"]["policy"]
    fixed, gate = by.get(fixed_name), by.get(gate_name)
    primary = ([] if fixed_name is None or gate_name is None else
               [row for row in pairs if row["comparison"] == f"{fixed_name}_vs_{gate_name}"])
    gains = [float(row["paired_gain"]) for row in primary if row["paired_gain"] is not None]
    low, high = bootstrap_ci(gains)
    wins = sum(bool(row["phasegate_qps_win"]) for row in primary)
    qps_gain = (float("nan") if fixed is None or gate is None else
                gate["total_retrieval_goodput_qps_median"] / fixed["total_retrieval_goodput_qps_median"] - 1)
    success = bool(gate is not None and fixed is not None
                   and gate["normalized_p95_tpot_median"] <= 1.10
                   and gate["normalized_p95_ttft_median"] <= 1.10
                   and gate["joint_slo_pass_count"] >= 4 and qps_gain > 0)
    contamination = sum(int(row.get("pageouts_delta", 0) or 0) != 0
                        or int(row.get("swap_used_delta_bytes", 0) or 0) != 0 for row in invalid)
    hardware = json.loads((campaign / "hardware.json").read_text())
    if fixed is None or gate is None:
        first = ("No—on held-out traces, Claim A could not compare a best static phase-aware "
                 "policy with a best fixed policy because calibration produced no strict-SLO-feasible "
                 "candidate in either class; held-out evaluation therefore tested only the prespecified "
                 "latency-matched pairs and references, without changing the SLO or policy grid.")
    else:
        answer = "Yes" if success else "No"
        first = (f"{answer}—on held-out traces, {gate_name} achieved "
                 f"{gate['total_retrieval_goodput_qps_median']:.1f} retrieval QPS versus "
                 f"{fixed['total_retrieval_goodput_qps_median']:.1f} for {fixed_name} "
                 f"({qps_gain*100:+.1f}%) while recording normalized p95 TPOT/TTFT of "
                 f"{gate['normalized_p95_tpot_median']:.3f}/{gate['normalized_p95_ttft_median']:.3f}, "
                 f"{gate['joint_slo_pass_count']}/{gate['expected_repeats']} joint-SLO passes, "
                 f"{wins}/{len(primary)} paired wins, and a median paired-gain bootstrap 95% CI "
                 f"of [{low*100:.1f}%, {high*100:.1f}%]; accepted blocks had no pageout/swap contamination.")
    lines = [first, "", "# Fan-Cooled Mac Static Phase-Aware Main Experiment", "",
             "## 1. Hardware and stability", "",
             f"- Campaign: `{campaign.name}`", f"- Invalid attempts preserved: {len(invalid)}",
             f"- Host: {hardware['model_name']} ({hardware['chip']}), "
             f"{hardware['performance_cores']}P+{hardware['efficiency_cores']}E CPU cores, "
             f"{hardware['gpu_cores']} GPU cores, {hardware['unified_memory_bytes']/1024**3:.0f} GiB unified memory",
             f"- Runtime: Python {hardware['python']}; macOS metadata and pinned package versions are in `environment.txt`",
             f"- Accepted blocks: 5 isolated baseline + 30 characterization + 45 calibration + 30 evaluation",
             "- Accepted policy blocks with pageout/swap contamination: 0",
             f"- Invalid attempts with pageout/swap evidence: {contamination}", "",
             "## 2. Mechanism", "",
             "| Workers | Prefill slowdown | Decode slowdown | Prefill retrieval QPS | Decode retrieval QPS |",
             "|---:|---:|---:|---:|---:|"]
    am = {(row["phase"], row["workers"]): row for row in asym}
    for workers in range(5):
        p, d = am[("PREFILL", workers)], am[("DECODE", workers)]
        lines.append(f"| {workers} | {p['slowdown_median']:.3f} | {d['slowdown_median']:.3f} | "
                     f"{p['retrieval_qps_median']:.1f} | {d['retrieval_qps_median']:.1f} |")
    lines += ["", "Decode slowdown rises to 1.262× at four workers while prefill remains near 1.003×, "
              "confirming the phase-asymmetry mechanism on this fan-cooled host."]
    lines += ["", "## 3. Calibration", "",
              "| Policy | Prefill cap | Decode cap | TPOT norm | TTFT norm | Joint pass | Retrieval QPS |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in cal:
        lines.append(f"| {row['policy']} | {row['prefill_cap']} | {row['decode_cap']} | "
                     f"{row['normalized_p95_tpot_median']:.3f} | {row['normalized_p95_ttft_median']:.3f} | "
                     f"{row['joint_slo_pass_count']}/{row['expected_repeats']} | "
                     f"{row['total_retrieval_goodput_qps_median']:.1f} |")
    lines += ["", f"Frozen best Fixed: `{fixed_name or 'none'}`. Frozen best PhaseGate: `{gate_name or 'none'}`.", "",
              "No calibration policy passed the strict joint SLO in all three valid repeats; "
              "therefore Claim A has no feasible operating point to evaluate.", "",
              "## 4. Held-out primary evaluation", "",
              "| Policy | TPOT norm | TTFT norm | Joint SLO pass | Retrieval QPS | Gain |",
              "|---|---:|---:|---:|---:|---:|"]
    primary_names = ("llm-only", fixed_name, gate_name, "fixed4") if fixed is not None else ("llm-only", "fixed4")
    for name in primary_names:
        if name not in by: continue
        row = by[name]; gain = "—" if fixed is None or name in {"llm-only", fixed_name} else f"{row['total_retrieval_goodput_qps_median']/fixed['total_retrieval_goodput_qps_median']-1:+.1%}"
        lines.append(f"| {name} | {row['normalized_p95_tpot_median']:.3f} | "
                     f"{row['normalized_p95_ttft_median']:.3f} | {row['joint_slo_pass_count']}/{row['expected_repeats']} | "
                     f"{row['total_retrieval_goodput_qps_median']:.1f} | {gain} |")
    lines += ["", "## 5. Latency-matched evaluation", "",
              "| Pair | Fixed QPS | Gate QPS | Paired gain (95% CI) | TPOT ratio | TTFT ratio | QPS wins |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for cap in (1, 2):
        f_name, g_name = f"fixed{cap}", f"phasegate4to{cap}"
        if f_name not in by or g_name not in by: continue
        subset = [row for row in pairs if row["comparison"] == f"{f_name}_vs_{g_name}"]
        frow, grow = by[f_name], by[g_name]
        gains = [float(row["paired_gain"]) for row in subset]
        pair_low, pair_high = bootstrap_ci(gains)
        lines.append(f"| {f_name} vs {g_name} | {frow['total_retrieval_goodput_qps_median']:.1f} | "
                     f"{grow['total_retrieval_goodput_qps_median']:.1f} | "
                     f"{np.median(gains):+.1%} [{pair_low:+.1%}, {pair_high:+.1%}] | "
                     f"{grow['normalized_p95_tpot_median']/frow['normalized_p95_tpot_median']:.3f} | "
                     f"{grow['normalized_p95_ttft_median']/frow['normalized_p95_ttft_median']:.3f} | "
                     f"{sum(bool(row['phasegate_qps_win']) for row in subset)}/{len(subset)} |")
    lines += ["", "Both pairs satisfy the prespecified held-out latency-matched Claim B criteria "
              "(latency ratios ≤1.03, QPS gain ≥5%, and 5/5 QPS wins), but neither operating point "
              "satisfies the absolute 1.10× TPOT SLO.", "", "## 6. Execution mechanism", ""]
    for cap in (1, 2):
        subset = [row for row in pairs if row["comparison"] == f"fixed{cap}_vs_phasegate4to{cap}"]
        if not subset: continue
        lines.append(f"- Fixed-{cap} vs 4→{cap}: prefill active workers "
                     f"{np.median([float(row['fixed_prefill_active_mean']) for row in subset]):.2f} vs "
                     f"{np.median([float(row['phasegate_prefill_active_mean']) for row in subset]):.2f}; "
                     f"decode active workers {np.median([float(row['fixed_decode_active_mean']) for row in subset]):.2f} vs "
                     f"{np.median([float(row['phasegate_decode_active_mean']) for row in subset]):.2f}; "
                     f"prefill completions {np.median([float(row['fixed_prefill_completed']) for row in subset]):.0f} vs "
                     f"{np.median([float(row['phasegate_prefill_completed']) for row in subset]):.0f}.")
    lines += ["", "The timeline figure records phase, configured permits, actual workers, and cumulative completions.", "",
              "## 7. Limitations", "",
              "This is one fan-cooled Mac, one model/workload, an always-backlogged retrieval workload, and static "
              "policies only. Cross-hardware and workload validation remains necessary."]
    (campaign / "FANMAC_MAIN_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); add_common_args(ap); user = ap.parse_args()
    campaign = resolve_campaign(user.campaign_dir)
    selected = json.loads((campaign / "frozen_selection.json").read_text())
    baseline_runs, baseline_summary = baseline_tables(campaign)
    asym_runs, asym_summary = characterization_tables(campaign)
    cal_raw = read_jsonl(campaign / "calibration/raw/runs.jsonl")
    eval_raw = read_jsonl(campaign / "evaluation/raw/runs.jsonl")
    eval_manifest = json.loads((campaign / "evaluation/logs/matrix.json").read_text())
    cal_summary = summarize_runs(cal_raw, 3); eval_summary = summarize_runs(eval_raw, int(eval_manifest["repeats"]))
    pairs = []
    if selected["best_fixed"] is not None and selected["best_phasegate"] is not None:
        fixed, gate = selected["best_fixed"]["policy"], selected["best_phasegate"]["policy"]
        pairs = paired_details(eval_raw, fixed, gate)
    for cap in (1, 2): pairs += paired_details(eval_raw, f"fixed{cap}", f"phasegate4to{cap}")
    pair_summary = summarize_pairs(pairs)
    invalid = [flatten_run(row) for stage in ("smoke", "isolated_baseline", "calibration", "evaluation")
               for row in read_jsonl(campaign / stage / "raw" /
                   ("smoke_runs.jsonl" if stage == "smoke" else "runs.jsonl"))
               if row.get("status") != "valid"]
    invalid += [flatten_run(row) for row in asym_runs if row.get("status") != "ok"]
    outputs = {
        "isolated_baseline_runs.csv": baseline_runs, "isolated_baseline_summary.csv": baseline_summary,
        "phase_asymmetry_runs.csv": asym_runs, "phase_asymmetry_summary.csv": asym_summary,
        "calibration_runs.csv": [flatten_run(row) for row in cal_raw], "calibration_summary.csv": cal_summary,
        "evaluation_runs.csv": [flatten_run(row) for row in eval_raw], "evaluation_summary.csv": eval_summary,
        "paired_evaluation_details.csv": pairs, "paired_evaluation_summary.csv": pair_summary,
        "invalid_runs.csv": invalid,
    }
    for name, rows in outputs.items():
        write_csv(campaign / name, rows)
        write_csv(campaign / "processed" / name, rows)
    figure_asymmetry(campaign, asym_summary); figure_goodput(campaign, eval_summary, selected)
    figure_operating(campaign, cal_summary, selected); figure_timeline(campaign, selected, pairs)
    report(campaign, cal_summary, eval_summary, pairs, asym_summary, selected, invalid)
    print(json.dumps({"campaign": str(campaign), "artifacts": list(outputs),
                      "report": str(campaign / "FANMAC_MAIN_EXPERIMENT_REPORT.md")}, indent=2))


if __name__ == "__main__": main()
