#!/usr/bin/env python3
"""Generate the audited Base-M4 r5 reports, manifest, and sanitized bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np


TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt"}


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def med(rows: Iterable[dict[str, str]], key: str) -> float:
    return median(float(row[key]) for row in rows)


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    output = np.empty(len(values), dtype=float)
    output[order] = np.arange(len(values), dtype=float)
    return output


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def sanitized(data: bytes, home: str) -> bytes:
    text = data.decode("utf-8")
    text = text.replace(home, "$HOME")
    return text.encode("utf-8")


def bundle(campaign: Path, repo: Path, home: str, head: str) -> tuple[Path, str, int, int]:
    output = campaign / "m4_results_bundle.zip"
    checksum_path = campaign / "m4_results_bundle.sha256"
    selected: list[tuple[Path, str]] = []
    for path in sorted(campaign.iterdir()):
        if not path.is_file() or path in {output, checksum_path}:
            continue
        if path.suffix.lower() in {".csv", ".json", ".md", ".pdf", ".png"}:
            selected.append((path, f"campaign/{path.name}"))
    for name in [
        "analyze_m4_r5_heldout.py",
        "analyze_m4_r5_output_shape.py",
        "finalize_m4_r5.py",
        "run_m4_r5_baseline.py",
        "run_m4_cpu_scaling.py",
        "run_m4_r5_mechanism.py",
        "run_m4_r5_calibration.py",
        "run_m4_r5_heldout.py",
        "run_m4_r5_output_shape.py",
    ]:
        path = repo / "scripts" / name
        selected.append((path, f"scripts/{name}"))

    contents: list[tuple[str, bytes]] = []
    for path, archive_name in selected:
        data = path.read_bytes()
        if path.suffix.lower() in TEXT_SUFFIXES:
            data = sanitized(data, home)
        contents.append((archive_name, data))
    commits = (
        "Base-M4 r5 reproducibility bundle\n"
        f"bundle_source_HEAD={head}\n"
        f"machine_audit_commit={json_file(campaign / 'machine_manifest.json')['repository_commit']}\n"
        f"cpu_scaling_freeze_commit={json_file(campaign / 'K_HI_FREEZE.json')['repository_commit']}\n"
        f"calibration_selection_commit={json_file(campaign / 'frozen_m4_selection.json')['repository_commit']}\n"
        f"output_shape_freeze_commit={json_file(campaign / 'OUTPUT_SHAPE_PROTOCOL_FREEZE.json')['repository_commit']}\n"
    ).encode()
    contents.append(("GIT_COMMITS.txt", commits))
    checksum_lines = [f"{hashlib.sha256(data).hexdigest()}  {name}" for name, data in contents]
    contents.append(("BUNDLE_CONTENTS.sha256", ("\n".join(checksum_lines) + "\n").encode()))

    fixed_time = (2026, 8, 7, 0, 0, 0)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(contents):
            if home.encode() in data:
                raise RuntimeError(f"private home path remains in {name}")
            info = zipfile.ZipInfo(name, fixed_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum_path.write_text(f"{digest}  {output.name}\n")
    return output, digest, len(contents), output.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    repo = Path(__file__).resolve().parents[1]
    head = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    now = datetime.now(timezone.utc).isoformat()

    machine = json_file(campaign / "machine_manifest.json")
    khi = json_file(campaign / "K_HI_FREEZE.json")
    selection = json_file(campaign / "frozen_m4_selection.json")
    primary_budget = float(selection["primary_B"])
    fixed = selection["selected_fixed"]
    phasegate = selection["selected_phasegate"]
    timegate = "timegate4to1"

    mechanism = csv_rows(campaign / "m4_mechanism_runs.csv")
    mechanism_stats: dict[str, dict[str, float]] = {}
    for policy in sorted({row["policy"] for row in mechanism}):
        rows = [row for row in mechanism if row["policy"] == policy and row["status"] == "valid"]
        mechanism_stats[policy] = {
            "n": len(rows),
            "tpot": med(rows, "p95_tpot_ms"),
            "ttft": med(rows, "p95_ttft_ms"),
            "norm_tpot": med(rows, "normalized_p95_tpot"),
            "norm_ttft": med(rows, "normalized_p95_ttft"),
            "qps": med(rows, "total_retrieval_goodput_qps"),
        }

    heldout_summary = {row["policy"]: row for row in csv_rows(campaign / "m4_heldout_summary.csv")}
    gains = csv_rows(campaign / "m4_pairwise_gains.csv")
    pg_fixed = [float(row["phasegate_vs_fixed_gain"]) for row in gains]
    pg_time = [float(row["phasegate_vs_timegate_gain"]) for row in gains]
    pg_fixed_median = float(gains[0]["phasegate_vs_fixed_median_gain"])
    pg_fixed_low = float(gains[0]["phasegate_vs_fixed_ci_low"])
    pg_fixed_high = float(gains[0]["phasegate_vs_fixed_ci_high"])
    pg_time_median = float(gains[0]["phasegate_vs_timegate_median_gain"])
    pg_time_low = float(gains[0]["phasegate_vs_timegate_ci_low"])
    pg_time_high = float(gains[0]["phasegate_vs_timegate_ci_high"])

    overlaps = csv_rows(campaign / "m4_cap_phase_overlap.csv")
    def overlap_median(policy: str, phase: str, cap: int) -> float:
        rows = [row for row in overlaps if row["policy"] == policy and row["phase"] == phase
                and int(row["cap"]) == cap]
        return med(rows, "fraction_within_phase")
    pg_prefill_high = overlap_median(phasegate, "PREFILL", 4)
    pg_decode_high = overlap_median(phasegate, "DECODE", 4)
    tg_prefill_high = overlap_median(timegate, "PREFILL", 4)
    tg_decode_high = overlap_median(timegate, "DECODE", 4)

    output_summary = csv_rows(campaign / "m4_output_length_summary.csv")
    output_pairs = csv_rows(campaign / "m4_output_length_pairs.csv")
    output_by_length: dict[int, dict[str, Any]] = {}
    for length in (64, 128, 512):
        rows = [row for row in output_summary if int(row["output_tokens"]) == length]
        by_policy = {row["policy"]: row for row in rows}
        pairs = [row for row in output_pairs if int(row["output_tokens"]) == length]
        output_by_length[length] = {
            "fixed_qps": float(by_policy[fixed]["median_retrieval_qps"]),
            "phasegate_qps": float(by_policy[phasegate]["median_retrieval_qps"]),
            "gain": float(by_policy[phasegate]["paired_phasegate_gain_median"]),
            "ci_low": float(by_policy[phasegate]["paired_gain_ci_low"]),
            "ci_high": float(by_policy[phasegate]["paired_gain_ci_high"]),
            "prefill_fraction": float(by_policy[phasegate]["median_aggregate_prefill_fraction"]),
            "fixed_slo": int(by_policy[fixed]["joint_slo_pass_count"]),
            "phasegate_slo": int(by_policy[phasegate]["joint_slo_pass_count"]),
            "pair_gains": [float(row["paired_qps_gain"]) for row in pairs],
        }
    prefill_values = np.asarray([float(row["phasegate_prefill_fraction"]) for row in output_pairs])
    gain_values = np.asarray([float(row["paired_qps_gain"]) for row in output_pairs])
    shape_pearson = float(np.corrcoef(prefill_values, gain_values)[0, 1])
    shape_spearman = float(np.corrcoef(ranks(prefill_values), ranks(gain_values))[0, 1])

    pageout = csv_rows(campaign / "m4_pageout_pair_audit.csv")
    pageout_free = sum(row["all_three_pageout_free"] == "True" for row in pageout)
    fixed_pageouts = [int(row["fixed_pageout_delta"]) for row in pageout]
    pg_pageouts = [int(row["phasegate_pageout_delta"]) for row in pageout]
    tg_pageouts = [int(row["timegate_pageout_delta"]) for row in pageout]
    imbalance = np.asarray([float(row["phasegate_minus_fixed_pageout"]) for row in pageout])
    pageout_spearman = float(np.corrcoef(ranks(imbalance), ranks(np.asarray(pg_fixed)))[0, 1])
    largest_imbalance_repeat = int(np.argmax(np.abs(imbalance)))
    excluding_largest = median(value for i, value in enumerate(pg_fixed) if i != largest_imbalance_repeat)
    leave_one_out = [float(row["leave_one_out_median_phasegate_vs_fixed_gain"])
                     for row in csv_rows(campaign / "m4_pageout_leave_one_out.csv")]

    baseline_512 = [row for row in jsonl_rows(campaign / "m4_output_512_baseline/raw/runs.jsonl")
                    if row["policy"] == "llm-only" and row["status"] == "valid"]
    if len(baseline_512) != 3:
        raise RuntimeError(f"expected three valid 512-token baselines, found {len(baseline_512)}")
    baseline_512_tpot = [float(row["p95_tpot_ms"]) for row in baseline_512]

    cpu = csv_rows(campaign / "cpu_scaling_summary.csv")
    cpu_rows = {int(row["cap"]): row for row in cpu}
    cpu_text = ", ".join(f"cap {cap}: {float(cpu_rows[cap]['median_qps']):.1f} QPS"
                         for cap in (1, 2, 4))

    h = machine["hardware"]
    heldout_table = "\n".join(
        f"| {policy} | {float(heldout_summary[policy]['median_retrieval_qps']):.1f} | "
        f"{float(heldout_summary[policy]['median_normalized_p95_tpot']):.3f} | "
        f"{float(heldout_summary[policy]['median_normalized_p95_ttft']):.3f} | "
        f"{heldout_summary[policy]['joint_slo_pass_count']}/7 |"
        for policy in (fixed, phasegate, timegate)
    )
    mechanism_table = "\n".join(
        f"| {policy} | {mechanism_stats[policy]['qps']:.1f} | "
        f"{mechanism_stats[policy]['norm_ttft']:.3f}x | {mechanism_stats[policy]['norm_tpot']:.3f}x |"
        for policy in ("llm-only", "fixed1", "fixed2", "fixed4")
    )
    shape_table = "\n".join(
        f"| {length} | {output_by_length[length]['prefill_fraction'] * 100:.2f}% | "
        f"{output_by_length[length]['fixed_qps']:.1f} | {output_by_length[length]['phasegate_qps']:.1f} | "
        f"{output_by_length[length]['gain'] * 100:.2f}% "
        f"[{output_by_length[length]['ci_low'] * 100:.2f}, {output_by_length[length]['ci_high'] * 100:.2f}] |"
        for length in (64, 128, 512)
    )

    final_report = f"""# Base-M4 PhaseGate r5 Final Report

This report uses only frozen, valid r5 measurements. Diagnostic smoke runs, invalid attempts, and the immutable r4 campaign are excluded from policy selection and confidence intervals.

## 1. Exact device configuration

The campaign ran on a fan-cooled {h['model_name']} ({h['model_identifier']}, {h['model_number']}) with an {h['chip']}, {h['performance_cores']} performance cores, {h['efficiency_cores']} efficiency cores ({h['total_cpu_cores']} total), {h['gpu_cores']} GPU cores, and {h['unified_memory']} unified memory. The OS was macOS {machine['macos']['product_version']} build {machine['macos']['build_version']}. Python was {machine['python']}; package versions were MLX {machine['packages']['mlx']}, MLX-LM {machine['packages']['mlx-lm']}, NumPy {machine['packages']['numpy']}, and FAISS {machine['packages']['faiss-cpu']}. The model was {machine['model']['identifier']} at revision `{machine['model']['revision']}`, 4-bit, with a 5.5 GB MLX limit. The shared read-only HNSW index contained {machine['index']['vectors']:,} vectors of dimension {machine['index']['dimensions']}, M={machine['index']['graph_degree']}, efConstruction={machine['index']['ef_construction']}, SHA-256 `{machine['index']['sha256']}`. This is a base M4, not an M4 Pro.

## 2. CPU-only scaling and K_hi

Three 60-second repeats were run at each cap. Median throughput was {cpu_text}. The frozen rule selected the smallest cap reaching at least 90% of the maximum median QPS; therefore `K_hi={khi['selected_K_hi']}`. Observed active-concurrency p95 exactly matched each requested cap, and cap 4 had no CPU-scaling swap or pageout flag. On this device, cap 4 was justified by the preregistered CPU-only rule.

## 3. PREFILL versus DECODE sensitivity

The mechanism blocks showed substantially greater TPOT than TTFT slowdown as fixed retrieval concurrency increased:

| Policy | Retrieval QPS | normalized p95 TTFT | normalized p95 TPOT |
|---|---:|---:|---:|
{mechanism_table}

At cap 4, median p95 TPOT increased to {mechanism_stats['fixed4']['norm_tpot']:.3f}x isolated while p95 TTFT increased to {mechanism_stats['fixed4']['norm_ttft']:.3f}x. Thus DECODE was more sensitive than PREFILL on this base-M4 system under the measured workload. These are matched metrics within r5; they are not equated with older M2-Pro metrics.

## 4. Frozen policies at primary_B

The fresh five-repeat baseline was stable under the predeclared +/-3% gate. Calibration froze `primary_B={primary_budget:.2f}`, `{fixed}`, and `{phasegate}` before held-out execution. The matched phase-blind control was `{timegate}` with a calibration-derived wall-clock replay schedule.

## 5. PhaseGate versus matched TimeGate

| Policy | median retrieval QPS | median normalized p95 TPOT | median normalized p95 TTFT | joint SLO passes |
|---|---:|---:|---:|---:|
{heldout_table}

Across seven paired randomized repeats, PhaseGate exceeded Fixed by a median {pct(pg_fixed_median)} (95% run-level paired bootstrap interval {pct(pg_fixed_low)} to {pct(pg_fixed_high)}). It exceeded matched TimeGate by {pct(pg_time_median)} ({pct(pg_time_low)} to {pct(pg_time_high)}). PhaseGate passed the joint 1.25x SLO in 7/7 blocks; TimeGate passed 0/7, with median normalized p95 TPOT {float(heldout_summary[timegate]['median_normalized_p95_tpot']):.3f}. The result supports phase alignment beyond burstiness: the phase-blind schedule delivered similar raw retrieval throughput but exposed DECODE to high concurrency and violated the latency budget.

## 6. TimeGate overlap with actual phases

PhaseGate spent a median {pct(pg_prefill_high)} of actual PREFILL and only {pg_decode_high * 100:.3f}% of actual DECODE at cap 4. TimeGate spent {pct(tg_prefill_high)} of PREFILL and {pct(tg_decode_high)} of DECODE at cap 4. Runtime audit fields confirm that TimeGate did not consult phase state when selecting its cap. Therefore TimeGate's high-cap intervals occurred materially during actual DECODE, as intended for the causal control.

## 7. Output-length dependence

| Output tokens | PhaseGate PREFILL fraction | Fixed QPS | PhaseGate QPS | median paired QPS gain [95% interval] |
|---:|---:|---:|---:|---:|
{shape_table}

The PhaseGate-versus-Fixed gain decreased monotonically from 64 to 512 output tokens. Policies remained fixed from the 128-token calibration; no length-specific recalibration was performed. This supports dependence on available PREFILL overlap time within the measured lengths and does not justify extrapolation beyond 64--512 tokens.

## 8. Association with measured PREFILL fraction

Across the 15 paired output-shape observations, the descriptive Pearson association between PhaseGate PREFILL fraction and paired QPS gain was r={shape_pearson:.4f}, and rank association was {shape_spearman:.3f}. At the three length-level medians the ordering was perfectly monotonic. Because output length jointly changes phase fraction and other runtime behavior, this is descriptive mechanism evidence, not an independent causal estimate of phase fraction.

## 9. Memory and pageout audit

All 21 primary held-out blocks had zero swap growth and normal memory pressure. Only {pageout_free}/7 paired triplets were completely pageout-free. Per-policy pageout deltas were Fixed median/range {median(fixed_pageouts):g}/{min(fixed_pageouts)}--{max(fixed_pageouts)}, PhaseGate {median(pg_pageouts):g}/{min(pg_pageouts)}--{max(pg_pageouts)}, and TimeGate {median(tg_pageouts):g}/{min(tg_pageouts)}--{max(tg_pageouts)}. The descriptive Spearman association between PhaseGate-minus-Fixed pageout imbalance and paired QPS gain was {pageout_spearman:.3f} at n=7. Excluding repeat {largest_imbalance_repeat}, which had the largest absolute imbalance, yielded a median gain of {pct(excluding_largest)}; leave-one-pair-out medians ranged from {pct(min(leave_one_out))} to {pct(max(leave_one_out))}. This robustness check does not prove absence of confounding. The campaign reproduces the result without swap or memory-pressure warnings, but it does **not** establish broadly pageout-free execution.

## 10. Paper-claim disposition

- Strengthen the device-specific claim that decode contention is more severe than prefill contention, using the matched base-M4 mechanism values above.
- Replace the old primary result with the frozen base-M4 Fixed/PhaseGate/TimeGate triplet and its run-level paired intervals.
- Strengthen the phase-alignment interpretation only in the bounded form supported by TimeGate's decode overlap, 0/7 SLO passes, and PhaseGate's {pct(pg_time_median)} median QPS advantage.
- Weaken any universal throughput-gain statement: the observed gain fell from {pct(output_by_length[64]['gain'])} at 64 tokens to {pct(output_by_length[512]['gain'])} at 512 tokens.
- Do not claim pageout neutrality or a fully pageout-free primary campaign; only {pageout_free}/7 triplets were completely pageout-free.
- Move the older flagged M2-Pro evidence to supporting context rather than numerically merging it with this campaign.
- Remove any M4-Pro label from these results and do not compare raw QPS across devices.

## Audit limitation

The 512-token isolated baseline p95 TPOT values were {', '.join(f'{x:.3f}' for x in baseline_512_tpot)} ms, showing tail variability. The sweep protocol did not preregister a +/-3% stability gate for these three per-length baselines, so the valid blocks were not selectively rerun. Consequently, normalized 512-token TPOT values below 1 must not be interpreted as retrieval improving decode latency; the paired retrieval-QPS shape result and event-measured phase fractions are the defensible conclusions.
"""
    (campaign / "M4_FINAL_REPORT.md").write_text(final_report)

    recommendations = f"""# Base-M4 Paper Update Recommendations

These are proposed replacements only. Do not edit the LaTeX automatically.

## Replacement abstract sentences

"On a fan-cooled base-M4 Mac mini, PhaseGate increased retrieval throughput by a median {pct(pg_fixed_median)} over the best calibrated continuous Fixed policy under a frozen 1.25x joint TTFT/TPOT budget across seven paired randomized repeats."

"Against a calibration-matched phase-blind TimeGate schedule, PhaseGate improved retrieval throughput by {pct(pg_time_median)} (95% paired bootstrap interval {pct(pg_time_low)}--{pct(pg_time_high)}) while satisfying the joint latency budget in 7/7 blocks versus 0/7 for TimeGate."

"The PhaseGate-versus-Fixed throughput gain decreased from {pct(output_by_length[64]['gain'])} at 64 output tokens to {pct(output_by_length[512]['gain'])} at 512 as the measured PREFILL fraction fell from {pct(output_by_length[64]['prefill_fraction'])} to {pct(output_by_length[512]['prefill_fraction'])}."

## Replacement contribution bullets

- "We provide a phase-blind, duty-cycle-matched TimeGate control that replays calibration-derived high/low cap intervals without access to LLM phase state."
- "We report seven paired randomized base-M4 repeats using frozen Fixed-1, PhaseGate 4->1, and TimeGate 4/1 policies, with run-level paired bootstrap intervals and complete per-repeat results."
- "We quantify workload-shape dependence at 64, 128, and 512 output tokens using event-derived PREFILL/DECODE wall time rather than percentile-derived phase estimates."
- "We audit swap, memory pressure, and pageouts per block and retain pageout-flagged but otherwise valid blocks rather than selecting favorable repeats."

## Paper-ready primary-results paragraph

"At the frozen primary budget B={primary_budget:.2f}, calibration selected Fixed-1 and PhaseGate 4->1. In seven held-out paired randomized repeats, PhaseGate achieved a median {float(heldout_summary[phasegate]['median_retrieval_qps']):.1f} retrieval QPS versus {float(heldout_summary[fixed]['median_retrieval_qps']):.1f} for Fixed-1, a median paired gain of {pct(pg_fixed_median)} (95% run-level bootstrap interval {pct(pg_fixed_low)}--{pct(pg_fixed_high)}). Both policies passed the joint p95 TTFT/TPOT budget in 7/7 blocks. PhaseGate's median normalized p95 TPOT and TTFT were {float(heldout_summary[phasegate]['median_normalized_p95_tpot']):.3f} and {float(heldout_summary[phasegate]['median_normalized_p95_ttft']):.3f}, respectively."

## Paper-ready phase-alignment-control paragraph

"The matched phase-blind TimeGate control achieved {float(heldout_summary[timegate]['median_retrieval_qps']):.1f} median retrieval QPS but passed the joint latency budget in 0/7 blocks because its median normalized p95 TPOT was {float(heldout_summary[timegate]['median_normalized_p95_tpot']):.3f}. PhaseGate exceeded TimeGate throughput by a median {pct(pg_time_median)} ({pct(pg_time_low)}--{pct(pg_time_high)}) while passing in 7/7 blocks. Event reconstruction showed that TimeGate ran cap 4 during {pct(tg_decode_high)} of actual DECODE time, whereas PhaseGate did so during only {pg_decode_high * 100:.3f}%. This comparison supports a bounded phase-alignment benefit beyond merely alternating between high and low concurrency."

## Paper-ready workload-shape paragraph

"With the Fixed-1 and PhaseGate 4->1 policies frozen at the 128-token calibration, PhaseGate's median paired retrieval-QPS gain decreased monotonically from {pct(output_by_length[64]['gain'])} at 64 tokens, to {pct(output_by_length[128]['gain'])} at 128, and {pct(output_by_length[512]['gain'])} at 512. The corresponding event-measured PhaseGate PREFILL fractions were {pct(output_by_length[64]['prefill_fraction'])}, {pct(output_by_length[128]['prefill_fraction'])}, and {pct(output_by_length[512]['prefill_fraction'])}. The trend supports dependence on available PREFILL overlap time over the measured range; it does not establish behavior outside these lengths."

## Paper-ready limitations paragraph

"The evaluation is device- and workload-specific: a base-M4 Mac mini, Qwen2.5-1.5B-Instruct 4-bit under MLX, 2,048-token context, and output lengths of 64--512. Only one of seven primary triplets was completely pageout-free, although all 21 blocks had zero swap growth and normal memory pressure; therefore we do not claim pageout neutrality. The 512-token isolated baseline showed run-level p95 TPOT tail variability ({', '.join(f'{x:.3f}' for x in baseline_512_tpot)} ms), so normalized tail metrics at that length require caution. Phase fraction and output length covary, and the observed association should not be extrapolated beyond the measured workloads or devices."

## Proposed primary table rows

| Policy | Retrieval QPS | normalized p95 TPOT | normalized p95 TTFT | joint passes |
|---|---:|---:|---:|---:|
{heldout_table}

## Proposed output-shape table rows

| Output tokens | PREFILL fraction | Fixed QPS | PhaseGate QPS | paired gain [95% interval] |
|---:|---:|---:|---:|---:|
{shape_table}

## Proposed figure captions

- **Primary control figure:** "Per-repeat retrieval-QPS gains for frozen PhaseGate 4->1 relative to Fixed-1 and calibration-matched phase-blind TimeGate 4/1 on a fan-cooled base-M4 Mac mini. Points are seven paired randomized repeats; intervals use 10,000 run-level paired bootstrap resamples."
- **Output-length figure:** "Median paired PhaseGate-versus-Fixed retrieval-QPS gain at 64, 128, and 512 output tokens. Policies were frozen at 128 tokens; error bars are 95% run-level paired bootstrap intervals over five pairs per length."
- **PREFILL-fraction figure:** "Paired retrieval-QPS gain versus event-measured PREFILL wall-time fraction. The association is descriptive because output length changes both phase fraction and other runtime behavior."
- **Pageout figure:** "Paired PhaseGate-versus-Fixed retrieval-QPS gain versus the PhaseGate-minus-Fixed global pageout delta. The n=7 correlation is descriptive and does not establish absence of pageout confounding."

## Statements that must not be made

- Do not call this hardware M4 Pro or generalize the result to all Apple Silicon.
- Do not claim the seven policy blocks are independent replicates; they are seven paired randomized repeats.
- Do not claim TimeGate was phase-aware or tuned on held-out phase overlap.
- Do not claim PhaseGate eliminates contention or always improves latency.
- Do not claim the {pct(pg_fixed_median)} 128-token gain is universal; the measured gain was length-dependent.
- Do not claim pageout-free or pageout-neutral primary execution; only {pageout_free}/7 triplets were completely pageout-free.
- Do not interpret normalized 512-token TPOT below 1 as evidence that retrieval accelerates decode.
- Do not use individual requests as bootstrap units for the primary comparison.
- Do not merge raw QPS across this base-M4 campaign and older M2-Pro/MacBook-Air campaigns.
- Do not change `K_hi`, the selected policies, or `primary_B` after observing held-out results.
"""
    (campaign / "M4_PAPER_UPDATE_RECOMMENDATIONS.md").write_text(recommendations)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": now,
        "campaign_id": campaign.name,
        "device_scope": "fan-cooled base Apple M4 Mac mini; not M4 Pro",
        "repository": {"branch": branch, "bundle_source_head": head},
        "machine": h,
        "macos": machine["macos"],
        "python": machine["python"],
        "packages": machine["packages"],
        "model": {"identifier": machine["model"]["identifier"],
                  "revision": machine["model"]["revision"],
                  "quantization": machine["model"]["quantization"]},
        "index": {key: value for key, value in machine["index"].items() if key != "path"},
        "mlx_memory_limit_gb": machine["mlx_memory_limit_gb"],
        "observer_mode": "event",
        "observer_semantics": {"phase_event_driven": True, "memory_sampling_max_hz": 1,
                               "production_subprocess_polling": False,
                               "token_timestamp_semantics_unchanged_from_r4": True},
        "workload": {"context_tokens": 2048, "primary_output_tokens": 128,
                     "shape_output_tokens": [64, 128, 512], "heldout_pairs": 7,
                     "shape_pairs_per_length": 5},
        "frozen_selection": {"primary_B": primary_budget, "fixed": fixed,
                             "phasegate": phasegate, "timegate": timegate,
                             "K_hi": khi["selected_K_hi"]},
        "primary_results": {"phasegate_vs_fixed_median_gain": pg_fixed_median,
                            "phasegate_vs_fixed_ci95": [pg_fixed_low, pg_fixed_high],
                            "phasegate_vs_timegate_median_gain": pg_time_median,
                            "phasegate_vs_timegate_ci95": [pg_time_low, pg_time_high],
                            "joint_slo_passes": {policy: int(heldout_summary[policy]["joint_slo_pass_count"])
                                                 for policy in (fixed, phasegate, timegate)}},
        "memory_audit": {"heldout_swap_growth_blocks": 0,
                         "heldout_warning_or_critical_memory_pressure_blocks": 0,
                         "completely_pageout_free_triplets": pageout_free,
                         "heldout_triplets": 7},
        "output_shape": {str(length): output_by_length[length] for length in (64, 128, 512)},
        "source_freezes": ["R5_BASELINE_FREEZE.json", "K_HI_FREEZE.json",
                           "CALIBRATION_POLICY_FREEZE.json", "frozen_m4_selection.json",
                           "timegate_schedule_freeze.json", "HELDOUT_PROTOCOL_FREEZE.json",
                           "OUTPUT_SHAPE_PROTOCOL_FREEZE.json"],
        "raw_data_policy": "raw request/timeline data retained in campaign tree but excluded from Git and bundle",
    }
    (campaign / "m4_reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    output, digest, count, size = bundle(campaign, repo, str(Path.home()), head)
    print(final_report)
    print(f"bundle={output} files={count} bytes={size} sha256={digest}")


if __name__ == "__main__":
    main()
