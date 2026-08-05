#!/usr/bin/env python3
"""Generate the bounded M2 mini replication tables, figures, reports, and bundle."""
from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "experiments/static_phaseaware/fanmac_m2mini_replication_20260805"


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def median(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def phase_times(run_key: str, stage: str) -> dict[str, float]:
    path = CAMPAIGN / stage / "raw/timelines" / f"{run_key}.json"
    samples = json.loads(path.read_text())["demand_samples"]
    totals = {"PREFILL": 0.0, "DECODE": 0.0, "IDLE": 0.0}
    for before, after in zip(samples, samples[1:]):
        if before["phase"] == after["phase"]:
            totals[before["phase"]] += max(0.0, float(after["timestamp"]) - float(before["timestamp"]))
    return totals


def cpu_report() -> None:
    rows = list(csv.DictReader((CAMPAIGN / "m2_cpu_scaling_summary.csv").open()))
    caps = [int(row["cap"]) for row in rows]
    qps = [float(row["median_qps"]) for row in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(caps, qps, marker="o"); ax.set_xticks(caps)
    ax.set_xlabel("External FAISS concurrency cap"); ax.set_ylabel("Retrieval QPS")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(CAMPAIGN / "figure_m2_cpu_scaling.pdf"); plt.close(fig)
    scale = qps[-1] / qps[0]
    (CAMPAIGN / "M2_CPU_SCALING_REPORT.md").write_text(
        "# M2 CPU-only HNSW Scaling\n\n"
        f"The median cap-4/cap-1 QPS ratio was **{scale:.3f}×** over three 60-second repeats. "
        "Each run used the same deterministic query stream within its repeat, one index-owner "
        "process, and FAISS internal thread count one.\n")


def mechanism_report() -> dict[str, Any]:
    rows = jsonl(CAMPAIGN / "m2_mechanism/raw/runs.jsonl")
    selected = [row for row in rows if row.get("status") == "valid"]
    by = {(int(row["repeat"]), row["policy"]): row for row in selected}
    details = []
    for repeat in range(3):
        base = by[(repeat, "llm-only")]
        for policy, cap in (("fixed1", 1), ("fixed2", 2), ("fixed4", 4)):
            row = by[(repeat, policy)]
            details.append({"repeat": repeat, "policy": policy, "cap": cap,
                "normalized_p95_tpot": float(row["p95_tpot_ms"]) / float(base["p95_tpot_ms"]),
                "normalized_p95_ttft": float(row["p95_ttft_ms"]) / float(base["p95_ttft_ms"]),
                "p99_inter_token_gap_ms": row["p99_inter_token_gap_ms"],
                "transition_gap_ms": row["p95_transition_gap_ms"],
                "qps": row["total_retrieval_goodput_qps"],
                "prefill_active_mean": row["prefill_active_retrieval_worker_mean"],
                "decode_active_mean": row["decode_active_retrieval_worker_mean"],
                "prefill_completions": row["completed_queries_prefill"],
                "decode_completions": row["completed_queries_decode"],
                "pageouts_delta": row["pageouts_delta"], "swap_delta": row["swap_used_delta_bytes"]})
    csv_write(CAMPAIGN / "m2_mechanism_runs.csv", details)
    summary = []
    for cap in (1, 2, 4):
        group = [row for row in details if row["cap"] == cap]
        summary.append({"cap": cap,
            "median_normalized_p95_tpot": median(group, "normalized_p95_tpot"),
            "median_normalized_p95_ttft": median(group, "normalized_p95_ttft"),
            "median_qps": median(group, "qps"),
            "median_p99_inter_token_gap_ms": median(group, "p99_inter_token_gap_ms"),
            "median_transition_gap_ms": median(group, "transition_gap_ms")})
    csv_write(CAMPAIGN / "m2_mechanism_summary.csv", summary)
    caps = [row["cap"] for row in summary]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.plot(caps, [row["median_normalized_p95_tpot"] for row in summary], marker="o", label="p95 TPOT")
    ax.plot(caps, [row["median_normalized_p95_ttft"] for row in summary], marker="s", label="p95 TTFT")
    ax.axhline(1, color="black", lw=.8); ax.set_xticks(caps); ax.set_xlabel("Fixed retrieval cap")
    ax.set_ylabel("Normalized latency"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(CAMPAIGN / "figure_m2_phase_asymmetry.pdf"); plt.close(fig)
    cap4 = next(row for row in summary if row["cap"] == 4)
    directional = (cap4["median_normalized_p95_tpot"] - 1
                   > cap4["median_normalized_p95_ttft"] - 1)
    (CAMPAIGN / "M2_MECHANISM_REPORT.md").write_text(
        "# M2 Phase-asymmetry Characterization\n\n"
        f"At cap 4, median normalized p95 TPOT was {cap4['median_normalized_p95_tpot']:.3f} "
        f"and TTFT was {cap4['median_normalized_p95_ttft']:.3f}. The directional criterion "
        f"(DECODE/TPOT degradation exceeding PREFILL/TTFT degradation) was **{'met' if directional else 'not met'}**.\n")
    return {"directional": directional, "cap4": cap4}


def bootstrap(values: list[float], seed: int = 926_806_501) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    draws = np.median(rng.choice(array, size=(10_000, len(array)), replace=True), axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float); order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float); start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]: end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0: return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def request_attainment(run_keys: set[str], tpot_limit: float, ttft_limit: float) -> dict[str, float]:
    records = jsonl(CAMPAIGN / "m2_heldout/raw/requests.jsonl")
    output = {}
    for record in records:
        if record["run_key"] in run_keys:
            requests = record["requests"]
            output[record["run_key"]] = float(np.mean([
                float(row["mean_tpot_ms"]) <= tpot_limit
                and float(row["user_visible_ttft_ms"]) <= ttft_limit for row in requests]))
    return output


def heldout_report(mechanism: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in jsonl(CAMPAIGN / "m2_heldout/raw/runs.jsonl") if row.get("status") == "valid"]
    selection = json.loads((CAMPAIGN / "m2_selection_freeze.json").read_text())
    baseline = json.loads((CAMPAIGN / "m2_baseline/baseline.json").read_text())
    budget = float(selection["m2_primary_B"] or selection["secondary_budget"])
    fixed, phase, timer = selection["fixed_policy"], selection["phasegate_policy"], selection["timegate_policy"]
    by = {(int(row["repeat"]), row["policy"]): row for row in rows}
    if any((repeat, policy) not in by for repeat in range(5)
           for policy in ("llm-only", fixed, phase, timer)):
        raise RuntimeError("held-out matrix incomplete")
    attainment = request_attainment({row["run_key"] for row in rows},
        baseline["p95_tpot_ms"] * budget, baseline["p95_ttft_ms"] * budget)
    run_table = []
    for row in rows:
        norm_tpot = float(row["p95_tpot_ms"]) / baseline["p95_tpot_ms"]
        norm_ttft = float(row["p95_ttft_ms"]) / baseline["p95_ttft_ms"]
        run_table.append({"repeat": row["repeat"], "policy": row["policy"], "run_key": row["run_key"],
            "retrieval_qps": row["total_retrieval_goodput_qps"], "normalized_p95_tpot": norm_tpot,
            "normalized_p95_ttft": norm_ttft, "joint_slo_pass": norm_tpot <= budget and norm_ttft <= budget,
            "request_joint_attainment": attainment[row["run_key"]],
            "p99_inter_token_gap_ms": row["p99_inter_token_gap_ms"],
            "transition_gap_ms": row["p95_transition_gap_ms"],
            "request_max_gap_p95_ms": row["per_request_max_gap_p95_ms"],
            "pageouts_delta": row["pageouts_delta"], "swap_delta": row["swap_used_delta_bytes"],
            "memory_free_percent_after": row["memory_free_percent_after"], "soft_flags": row["soft_flags"]})
    csv_write(CAMPAIGN / "m2_heldout_runs.csv", run_table)
    summaries = []
    for policy in ("llm-only", fixed, phase, timer):
        group = [row for row in run_table if row["policy"] == policy]
        summaries.append({"policy": policy, "valid_repeats": len(group), "frozen_slo_budget": budget,
            "median_qps": median(group, "retrieval_qps"), "median_normalized_p95_tpot": median(group, "normalized_p95_tpot"),
            "median_normalized_p95_ttft": median(group, "normalized_p95_ttft"),
            "joint_slo_pass_count": sum(str(row["joint_slo_pass"]).lower() == "true" if isinstance(row["joint_slo_pass"], str)
                                        else bool(row["joint_slo_pass"]) for row in group),
            "median_request_joint_attainment": median(group, "request_joint_attainment")})
    csv_write(CAMPAIGN / "m2_heldout_summary.csv", summaries)
    gains = []
    for repeat in range(5):
        f, p, t = by[(repeat, fixed)], by[(repeat, phase)], by[(repeat, timer)]
        gains.append({"repeat": repeat, "fixed_qps": f["total_retrieval_goodput_qps"],
            "phasegate_qps": p["total_retrieval_goodput_qps"], "timegate_qps": t["total_retrieval_goodput_qps"],
            "phasegate_vs_fixed_gain": p["total_retrieval_goodput_qps"] / f["total_retrieval_goodput_qps"] - 1,
            "phasegate_vs_timegate_gain": p["total_retrieval_goodput_qps"] / t["total_retrieval_goodput_qps"] - 1})
    pf = [float(row["phasegate_vs_fixed_gain"]) for row in gains]
    pt = [float(row["phasegate_vs_timegate_gain"]) for row in gains]
    pf_ci, pt_ci = bootstrap(pf), bootstrap(pt, seed=926_806_502)
    for row in gains:
        row.update({"phasegate_vs_fixed_bootstrap_low": pf_ci[0], "phasegate_vs_fixed_bootstrap_high": pf_ci[1],
                    "phasegate_vs_timegate_bootstrap_low": pt_ci[0], "phasegate_vs_timegate_bootstrap_high": pt_ci[1]})
    csv_write(CAMPAIGN / "m2_pairwise_gains.csv", gains)
    overlap = [{"repeat": repeat, "policy": policy,
        "prefill_high_cap_fraction": by[(repeat, policy)]["prefill_high_cap_fraction"],
        "prefill_low_cap_fraction": by[(repeat, policy)]["prefill_low_cap_fraction"],
        "decode_high_cap_fraction": by[(repeat, policy)]["decode_high_cap_fraction"],
        "decode_low_cap_fraction": by[(repeat, policy)]["decode_low_cap_fraction"],
        "high_cap_duty_fraction": by[(repeat, policy)]["high_cap_duty_fraction"],
        "cap_transitions_per_s": by[(repeat, policy)]["cap_transitions_per_s"],
        "prefill_active_mean": by[(repeat, policy)]["prefill_active_retrieval_worker_mean"],
        "decode_active_mean": by[(repeat, policy)]["decode_active_retrieval_worker_mean"]}
        for repeat in range(5) for policy in (fixed, phase, timer)]
    csv_write(CAMPAIGN / "m2_cap_phase_overlap.csv", overlap)
    pageout = []
    for repeat, gain in enumerate(gains):
        f, p, t = by[(repeat, fixed)], by[(repeat, phase)], by[(repeat, timer)]
        pageout.append({"repeat": repeat, "fixed_pageout_delta": f["pageouts_delta"],
            "phasegate_pageout_delta": p["pageouts_delta"], "timegate_pageout_delta": t["pageouts_delta"],
            "phasegate_fixed_pageout_imbalance": int(p["pageouts_delta"]) - int(f["pageouts_delta"]),
            "phasegate_timegate_pageout_imbalance": int(p["pageouts_delta"]) - int(t["pageouts_delta"]),
            "phasegate_vs_fixed_gain": gain["phasegate_vs_fixed_gain"],
            "phasegate_vs_timegate_gain": gain["phasegate_vs_timegate_gain"],
            "fixed_swap_delta": f["swap_used_delta_bytes"], "phasegate_swap_delta": p["swap_used_delta_bytes"],
            "timegate_swap_delta": t["swap_used_delta_bytes"],
            "fully_pageout_free": int(f["pageouts_delta"]) == int(p["pageouts_delta"]) == int(t["pageouts_delta"]) == 0})
    csv_write(CAMPAIGN / "m2_pageout_pair_audit.csv", pageout)
    imbalance = [float(row["phasegate_fixed_pageout_imbalance"]) for row in pageout]
    corr = spearman(imbalance, pf)
    loo = [{"excluded_repeat": excluded, "median_phasegate_fixed_gain": float(np.median([
        pf[index] for index in range(5) if index != excluded]))} for excluded in range(5)]
    csv_write(CAMPAIGN / "m2_pageout_leave_one_out.csv", loo)
    largest = int(np.argmax(np.abs(imbalance)))
    pageout_free = sum(bool(row["fully_pageout_free"]) for row in pageout)
    fig, ax = plt.subplots(figsize=(5.2, 3.4)); ax.scatter(imbalance, pf)
    ax.set_xlabel("Pageout imbalance (PhaseGate − Fixed)"); ax.set_ylabel("Paired PhaseGate/Fixed QPS gain")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(CAMPAIGN / "figure_m2_pageout_vs_gain.pdf"); plt.close(fig)
    (CAMPAIGN / "M2_PAGEOUT_REPORT.md").write_text(
        "# M2 Pageout Audit\n\n"
        f"Fully pageout-free three-policy pairs: **{pageout_free}/5**. Descriptive Spearman correlation "
        f"between PhaseGate/Fixed QPS gain and pageout imbalance: **{corr:.3f}**. We "
        f"{'did' if np.isfinite(corr) and abs(corr) >= .5 else 'did not'} observe a visible association in five paired runs. "
        "This descriptive check does not prove absence of confounding.\n"
        f"The median gain after excluding the largest absolute imbalance (repeat {largest}) was "
        f"{np.median([pf[i] for i in range(5) if i != largest]):+.2%}.\n")
    # Main comparison figure.
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(5); width = .24
    for offset, policy in zip((-width, 0, width), (fixed, phase, timer)):
        ax.bar(x + offset, [float(by[(r, policy)]["total_retrieval_goodput_qps"]) for r in range(5)],
               width, label=policy)
    ax.set_xlabel("Paired repeat"); ax.set_ylabel("Retrieval QPS"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(CAMPAIGN / "figure_m2_fixed_phasegate_timegate.pdf"); plt.close(fig)
    time_decode = float(np.median([float(by[(repeat, timer)]["decode_high_cap_fraction"]) for repeat in range(5)]))
    med_pf, med_pt = float(np.median(pf)), float(np.median(pt))
    (CAMPAIGN / "M2_REPLICATION_REPORT.md").write_text(
        "# M2 Cross-device Replication\n\n"
        f"Frozen test: `{fixed}` vs `{phase}` vs phase-blind `{timer}` at budget **{budget:.2f}×**. "
        f"Median paired PhaseGate/Fixed QPS gain was **{med_pf:+.2%}** (10,000-resample paired-run bootstrap "
        f"interval {pf_ci[0]:+.2%} to {pf_ci[1]:+.2%}). Median paired PhaseGate/TimeGate gain was "
        f"**{med_pt:+.2%}** ({pt_ci[0]:+.2%} to {pt_ci[1]:+.2%}). Median TimeGate high-cap overlap "
        f"with actual DECODE was **{time_decode:.2%}**. Device results are not pooled with M4 Pro or old M2 Pro runs.\n")
    return {"budget": budget, "fixed": fixed, "phase": phase, "timer": timer,
            "median_pf": med_pf, "median_pt": med_pt, "pf_ci": pf_ci, "pt_ci": pt_ci,
            "time_decode": time_decode, "pageout_free": pageout_free, "corr": corr,
            "summaries": summaries, "mechanism": mechanism}


def output512_report() -> dict[str, Any]:
    rows = [row for row in jsonl(CAMPAIGN / "m2_output512/raw/runs.jsonl") if row.get("status") == "valid"]
    base = json.loads((CAMPAIGN / "m2_output512_baseline/baseline.json").read_text())
    selection = json.loads((CAMPAIGN / "m2_selection_freeze.json").read_text())
    phase = selection["phasegate_policy"]
    by = {(int(row["repeat"]), row["policy"]): row for row in rows}
    details = []
    for repeat in range(3):
        for policy in ("fixed1", phase):
            row = by[(repeat, policy)]; times = phase_times(row["run_key"], "m2_output512")
            details.append({"repeat": repeat, "policy": policy, "qps": row["total_retrieval_goodput_qps"],
                "normalized_p95_tpot": row["p95_tpot_ms"] / base["p95_tpot_ms"],
                "normalized_p95_ttft": row["p95_ttft_ms"] / base["p95_ttft_ms"],
                "transition_gap_ms": row["p95_transition_gap_ms"], "prefill_wall_s": times["PREFILL"],
                "decode_wall_s": times["DECODE"],
                "prefill_fraction": times["PREFILL"] / (times["PREFILL"] + times["DECODE"])})
    csv_write(CAMPAIGN / "m2_output512_runs.csv", details)
    gains = [by[(r, phase)]["total_retrieval_goodput_qps"] / by[(r, "fixed1")]["total_retrieval_goodput_qps"] - 1
             for r in range(3)]
    summary = {"output_tokens": 512, "paired_repeats": 3, "median_qps_gain": float(np.median(gains)),
               "median_prefill_fraction": median(details, "prefill_fraction")}
    csv_write(CAMPAIGN / "m2_output512_summary.csv", [summary])
    (CAMPAIGN / "M2_OUTPUT512_REPORT.md").write_text(
        "# M2 Optional 512-token Check\n\n"
        f"With frozen caps and no recalibration, median paired QPS gain was **{summary['median_qps_gain']:+.2%}**; "
        f"the measured median PREFILL wall-time fraction was **{summary['median_prefill_fraction']:.2%}**. "
        "This is a directional secondary point.\n")
    return summary


def final_reports(result: dict[str, Any], output512: dict[str, Any]) -> None:
    machine = json.loads((CAMPAIGN / "m2_machine_manifest.json").read_text())
    mode = json.loads((CAMPAIGN / "memory_mode.json").read_text())["mode"]
    selection = json.loads((CAMPAIGN / "m2_selection_freeze.json").read_text())
    baseline = json.loads((CAMPAIGN / "m2_baseline/baseline.json").read_text())
    cpu = list(csv.DictReader((CAMPAIGN / "m2_cpu_scaling_summary.csv").open()))
    cap4_scale = float(cpu[-1]["median_qps"]) / float(cpu[0]["median_qps"])
    exact = bool(selection["exact_4to1_feasible"])
    text = f"""# M2 Final Report

1. **Device:** Mac mini `{machine['model_identifier']}`, Apple M2, 4P+4E CPU cores, {machine['gpu_cores']} GPU cores, {machine['unified_memory_bytes']/1024**3:.0f} GiB unified memory.
2. **Memory mode:** `{mode}`; the 100k preflight had no swap growth, pressure failure, or pageout delta in any policy.
3. **CPU-only scaling:** cap 4 delivered {cap4_scale:.3f}× cap-1 median HNSW QPS.
4. **Phase asymmetry:** the directional DECODE-sensitivity criterion was {'met' if result['mechanism']['directional'] else 'not met'}.
5. **Exact 4→1 feasibility:** {'feasible' if exact else 'infeasible; the predeclared 2→1 secondary was used'} at the frozen device budget.
6. **PhaseGate vs Fixed-1:** median paired QPS gain {result['median_pf']:+.2%}.
7. **PhaseGate vs TimeGate:** median paired QPS gain {result['median_pt']:+.2%}.
8. **TimeGate DECODE overlap:** median high-cap overlap {result['time_decode']:.2%}.
9. **Pageout-free pairs:** {result['pageout_free']}/5.
10. **Scope:** M2 metrics and intervals remain separate from M4 Pro and old M2 Pro. The optional 512-token point is directional and was not recalibrated.
"""
    (CAMPAIGN / "M2_FINAL_REPORT.md").write_text(text)
    summaries = {row["policy"]: row for row in result["summaries"]}
    comparison = f"""# M2 Cross-device Comparison Input

| Device/chip | Memory | Commit | Index | Caps | Baseline p95 TPOT / TTFT | Phase asymmetry | Frozen policies | Budget | Median QPS (Fixed / PhaseGate / TimeGate) | Paired gains (P/F; P/T) | Pageout-free |
|---|---:|---|---|---|---|---|---|---:|---|---|---:|
| Mac mini / Apple M2 (`Mac14,3`) | 16 GiB | `{machine['repository_commit']}` | {mode}, 100k×384 HNSW | 1 and {selection['prefill_cap']}→1 | {baseline['p95_tpot_ms']:.3f} ms / {baseline['p95_ttft_ms']:.3f} ms | {'DECODE-sensitive' if result['mechanism']['directional'] else 'criterion not met'} | `{result['fixed']}` / `{result['phase']}` / `{result['timer']}` | {result['budget']:.2f}× | {float(summaries[result['fixed']]['median_qps']):.1f} / {float(summaries[result['phase']]['median_qps']):.1f} / {float(summaries[result['timer']]['median_qps']):.1f} | {result['median_pf']:+.2%}; {result['median_pt']:+.2%} | {result['pageout_free']}/5 |

Scope caveat: independent fan-cooled base-M2 cross-device replication; no pooled confidence interval and no direct QPS equivalence claim across devices.
"""
    (CAMPAIGN / "M2_CROSS_DEVICE_COMPARISON_INPUT.md").write_text(comparison)
    recommendations = f"""# M2 Paper Update Recommendations

## Proposed replacement paragraph

On a fan-cooled 16 GiB Apple M2 Mac mini, we independently repeated the fixed-cap, phase-gated, and phase-blind duty-cycle control using a fresh device-local baseline and a frozen {result['budget']:.2f}× joint p95 TPOT/TTFT budget. PhaseGate's median paired retrieval-QPS gain was {result['median_pf']:+.2%} over Fixed-1 and {result['median_pt']:+.2%} over matched TimeGate across five held-out paired repeats. The TimeGate placed its high cap in actual DECODE for a median {result['time_decode']:.2%} of DECODE time. These M2 runs are reported independently and are not pooled with M4 Pro or earlier M2 Pro intervals.

## Scope sentence

The M2 result is a cross-device replication under the same 100k HNSW shape, not a replacement for the primary M4 Pro evaluation and not a clean remeasurement of the earlier M2 Pro campaign.

## Required qualification

{'All five held-out M2 pairs were pageout-free, strengthening the within-device interpretation.' if result['pageout_free'] == 5 else f'Only {result["pageout_free"]} of five M2 pairs were fully pageout-free; pageout imbalance analyses remain descriptive and do not prove absence of confounding.'}
"""
    (CAMPAIGN / "M2_PAPER_UPDATE_RECOMMENDATIONS.md").write_text(recommendations)
    repro = {"device": "Mac14,3 Apple M2", "memory_bytes": machine["unified_memory_bytes"],
             "harness_commit": machine["repository_commit"], "model": machine["model"],
             "index": {key: value for key, value in machine["index"].items() if key != "path"},
             "memory_mode": mode, "selection": selection, "pooled_cross_device_ci": False,
             "campaign_relative_path": "experiments/static_phaseaware/fanmac_m2mini_replication_20260805"}
    (CAMPAIGN / "m2_reproducibility_manifest.json").write_text(json.dumps(repro, indent=2) + "\n")


def bundle() -> None:
    names = [path for path in CAMPAIGN.iterdir() if path.is_file() and
             (path.suffix in {".csv", ".md", ".json", ".pdf"})]
    names += [REPO / "scripts/run_m2mini_replication.py",
              REPO / "scripts/analyze_m2mini_replication.py",
              REPO / "scripts/run_static_phaseaware_pilot.py",
              REPO / "src/phaseguard/policies.py"]
    target = CAMPAIGN / "m2_results_bundle.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in names:
            arcname = (f"scripts/{path.name}" if path.is_relative_to(REPO / "scripts") else
                       f"src/phaseguard/{path.name}" if path.is_relative_to(REPO / "src/phaseguard") else path.name)
            if path.suffix in {".md", ".json", ".csv", ".py"}:
                text = path.read_text().replace(str(REPO), "$REPO").replace(str(Path.home()), "$HOME")
                archive.writestr(arcname, text)
            else:
                archive.write(path, arcname)


def main() -> None:
    cpu_report()
    mechanism = mechanism_report()
    result = heldout_report(mechanism)
    output512 = output512_report()
    final_reports(result, output512)
    bundle()
    print(json.dumps({"analysis_complete": True, "campaign": str(CAMPAIGN)}, indent=2))


if __name__ == "__main__":
    main()
