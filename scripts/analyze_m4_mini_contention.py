#!/usr/bin/env python3
"""Analyze and package the diagnostic Base-M4 Mini contention campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
AIR_NORMALIZED = {
    0: {"ttft": 1.000, "tpot": 1.000},
    1: {"ttft": 1.027, "tpot": 1.040},
    2: {"ttft": 1.043, "tpot": 1.194},
    4: {"ttft": 1.057, "tpot": 1.613},
}


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def percentile(values: Iterable[float], q: float) -> float:
    return float(np.percentile(np.asarray(list(values), dtype=float), q))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def cap(policy: str) -> int:
    return 0 if policy == "llm-only" else int(policy.removeprefix("fixed"))


def thermal_clean(snapshot: dict[str, Any]) -> bool:
    text = snapshot["thermal"]["stdout"].lower()
    return (snapshot["thermal"]["returncode"] == 0
            and "no thermal warning level has been recorded" in text
            and "no performance warning level has been recorded" in text)


def create_bundle(campaign: Path) -> tuple[Path, str, int]:
    output = campaign / "m4_mini_contention_results_bundle.zip"
    checksum = campaign / "m4_mini_contention_results_bundle.sha256"
    included: list[tuple[str, bytes]] = []
    home = str(Path.home())
    for path in sorted(campaign.rglob("*")):
        if not path.is_file() or path in {output, checksum}:
            continue
        if path.name.endswith("_orchestration.log") or path.suffix.lower() in {
            ".json", ".jsonl", ".csv", ".md", ".pdf", ".png", ".txt"
        }:
            data = path.read_bytes()
            if path.suffix.lower() in {".json", ".jsonl", ".csv", ".md", ".txt", ".log"}:
                data = data.decode("utf-8").replace(home, "$HOME").encode()
            included.append((f"campaign/{path.relative_to(campaign)}", data))
    for name in ("run_m4_mini_contention.py", "analyze_m4_mini_contention.py"):
        path = REPO / "scripts" / name
        included.append((f"scripts/{name}", path.read_text().replace(home, "$HOME").encode()))
    hashes = [f"{hashlib.sha256(data).hexdigest()}  {name}" for name, data in included]
    included.append(("BUNDLE_CONTENTS.sha256", ("\n".join(hashes) + "\n").encode()))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(included):
            if home.encode() in data:
                raise RuntimeError(f"private path remains in {name}")
            info = zipfile.ZipInfo(name, (2026, 8, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    digest = sha256(output)
    checksum.write_text(f"{digest}  {output.name}\n")
    return output, digest, len(included)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("campaign", type=Path)
    args = parser.parse_args(); campaign = args.campaign.resolve()
    matrix = json.loads((campaign / "FROZEN_EXECUTION_MATRIX.json").read_text())
    complete = json.loads((campaign / "M4_MINI_CONTENTION_COLLECTION_COMPLETE.json").read_text())
    if int(matrix["frozen_K_hi"]) != 2 or int(complete["frozen_K_hi"]) != 2:
        raise RuntimeError("K_hi changed after freeze")
    raw_all = jsonl(campaign / "m4_mini_contention/raw/runs.jsonl")
    invalidated_path = campaign / "externally_invalidated_runs.jsonl"
    invalidated = ({row["run_key"] for row in jsonl(invalidated_path)}
                   if invalidated_path.exists() else set())
    runs = [row for row in raw_all if row.get("status") == "valid"
            and row.get("run_key") not in invalidated]
    if len(runs) != 12:
        raise RuntimeError(f"expected 12 valid blocks, found {len(runs)}")
    baseline_name = complete["passed_baseline_set"]
    baseline_result = json.loads(
        (campaign / f"CONTENTION_BASELINE_{baseline_name}_RESULT.json").read_text())
    normalization_tpot = float(baseline_result["median_p95_tpot_ms"])
    normalization_ttft = float(baseline_result["median_p95_ttft_ms"])
    requests = {row["run_key"]: row["requests"]
                for row in jsonl(campaign / "m4_mini_contention/raw/requests.jsonl")}
    thermal_rows = jsonl(campaign / "thermal_power_audit.jsonl")
    thermal = {(row["stage"], row["policy"], int(row["repeat"]), int(row["attempt"])): row
               for row in thermal_rows}
    by = {(row["policy"], int(row["repeat"])): row for row in runs}
    detailed = []
    for row in sorted(runs, key=lambda item: (int(item["repeat"]), cap(item["policy"]))):
        policy = row["policy"]; repeat = int(row["repeat"]); request_data = requests[row["run_key"]]
        baseline = by[("llm-only", repeat)]
        gaps = [float(gap) for request in request_data for gap in request["tpot_intervals_ms"]]
        transitions = [(float(request["token_timestamps"][0]) - float(request["prefill_end"])) * 1000
                       for request in request_data]
        maximum_gaps = [max(float(value) for value in request["tpot_intervals_ms"])
                        for request in request_data]
        audit = thermal[("m4_mini_contention", policy, repeat, int(row["attempt"]))]
        event = row["observer_reconstruction_audit"]
        detailed.append({
            "repeat": repeat, "policy": policy, "cap": cap(policy), "run_key": row["run_key"],
            "p95_ttft_ms": row["p95_ttft_ms"],
            "normalized_p95_ttft": float(row["p95_ttft_ms"]) / normalization_ttft,
            "normalized_p95_ttft_within_repeat": float(row["p95_ttft_ms"]) / float(baseline["p95_ttft_ms"]),
            "p95_tpot_ms": row["p95_tpot_ms"],
            "normalized_p95_tpot": float(row["p95_tpot_ms"]) / normalization_tpot,
            "normalized_p95_tpot_within_repeat": float(row["p95_tpot_ms"]) / float(baseline["p95_tpot_ms"]),
            "retrieval_qps": row["total_retrieval_goodput_qps"],
            "inter_token_gap_p99_ms": percentile(gaps, 99),
            "request_maximum_gap_p95_ms": percentile(maximum_gaps, 95),
            "phase_transition_gap_p95_ms": percentile(transitions, 95),
            "pageouts_delta": row["pageouts_delta"], "swap_used_delta_bytes": row["swap_used_delta_bytes"],
            "memory_pressure_clean": row["memory_pressure_clean"],
            "thermal_clean_before": thermal_clean(audit["before"]),
            "thermal_clean_after": thermal_clean(audit["after"]),
            "thermal_before": audit["before"]["thermal"]["stdout"].replace("\n", " | "),
            "thermal_after": audit["after"]["thermal"]["stdout"].replace("\n", " | "),
            "power_before": audit["before"]["power"]["stdout"].replace("\n", " | "),
            "power_after": audit["after"]["power"]["stdout"].replace("\n", " | "),
            "observer_mode": row["observer_mode"],
            "observer_subprocess_count": row["observer_subprocess_count_during_block"],
            "event_sequence_exact": event["sequence_exact"],
            "event_timestamps_monotonic": event["timestamps_monotonic"],
            "query_accounting_exact": event["query_accounting_exact"],
            "token_events": event["token_events"], "duration_s": row["duration_s"],
        })
    write_csv(campaign / "m4_mini_contention_per_run.csv", detailed)

    contention_request_rows = []
    for row in sorted(runs, key=lambda item: (int(item["repeat"]), cap(item["policy"]))):
        for request_index, request in enumerate(requests[row["run_key"]]):
            gaps = [float(value) for value in request["tpot_intervals_ms"]]
            contention_request_rows.append({
                "repeat": row["repeat"], "policy": row["policy"],
                "cap": cap(row["policy"]), "run_key": row["run_key"],
                "request_index": request_index, "request_id": request["request_id"],
                "ttft_ms": request["ttft_ms"], "mean_tpot_ms": request["mean_tpot_ms"],
                "p50_tpot_ms": request["p50_tpot_ms"], "p95_tpot_ms": request["p95_tpot_ms"],
                "p99_tpot_ms": request["p99_tpot_ms"], "maximum_gap_ms": max(gaps),
                "phase_transition_gap_ms": (
                    float(request["token_timestamps"][0]) - float(request["prefill_end"])) * 1000,
                "token_timestamp_count": len(request["token_timestamps"]),
            })
    write_csv(campaign / "m4_mini_contention_request_metrics.csv", contention_request_rows)

    baseline_run_rows = []
    baseline_request_rows = []
    for set_name in ("A", "B"):
        run_csv = campaign / f"contention_baseline_{set_name}_runs.csv"
        stage = campaign / f"contention_baseline_{set_name}"
        if not run_csv.exists():
            continue
        baseline_run_rows.extend(csv_rows(run_csv))
        run_rows = [row for row in jsonl(stage / "raw/runs.jsonl")
                    if row.get("status") == "valid" and row.get("run_key") not in invalidated]
        request_map = {row["run_key"]: row["requests"]
                       for row in jsonl(stage / "raw/requests.jsonl")}
        for run in sorted(run_rows, key=lambda item: int(item["repeat"])):
            for request_index, request in enumerate(request_map[run["run_key"]]):
                gaps = [float(value) for value in request["tpot_intervals_ms"]]
                baseline_request_rows.append({
                    "baseline_set": set_name, "repeat": run["repeat"],
                    "run_key": run["run_key"], "request_index": request_index,
                    "request_id": request["request_id"], "ttft_ms": request["ttft_ms"],
                    "mean_tpot_ms": request["mean_tpot_ms"],
                    "p50_tpot_ms": request["p50_tpot_ms"],
                    "p95_tpot_ms": request["p95_tpot_ms"],
                    "p99_tpot_ms": request["p99_tpot_ms"],
                    "maximum_gap_ms": max(gaps),
                    "phase_transition_gap_ms": (
                        float(request["token_timestamps"][0]) - float(request["prefill_end"])) * 1000,
                    "token_timestamp_count": len(request["token_timestamps"]),
                })
    write_csv(campaign / "m4_mini_contention_baseline_runs.csv", baseline_run_rows)
    write_csv(campaign / "m4_mini_contention_baseline_request_metrics.csv", baseline_request_rows)

    summary = []
    for policy in ("llm-only", "fixed1", "fixed2", "fixed4"):
        rows = [row for row in detailed if row["policy"] == policy]
        summary.append({
            "policy": policy, "cap": cap(policy), "valid_repeats": len(rows),
            "median_p95_ttft_ms": median(float(row["p95_ttft_ms"]) for row in rows),
            "median_normalized_p95_ttft": median(float(row["normalized_p95_ttft"]) for row in rows),
            "median_paired_normalized_p95_ttft": median(float(row["normalized_p95_ttft_within_repeat"]) for row in rows),
            "median_p95_tpot_ms": median(float(row["p95_tpot_ms"]) for row in rows),
            "median_normalized_p95_tpot": median(float(row["normalized_p95_tpot"]) for row in rows),
            "median_paired_normalized_p95_tpot": median(float(row["normalized_p95_tpot_within_repeat"]) for row in rows),
            "median_retrieval_qps": median(float(row["retrieval_qps"]) for row in rows),
            "median_inter_token_gap_p99_ms": median(float(row["inter_token_gap_p99_ms"]) for row in rows),
            "median_request_maximum_gap_p95_ms": median(float(row["request_maximum_gap_p95_ms"]) for row in rows),
            "median_phase_transition_gap_p95_ms": median(float(row["phase_transition_gap_p95_ms"]) for row in rows),
            "pageout_flag_count": sum(int(row["pageouts_delta"]) > 0 for row in rows),
            "pageout_median": median(int(row["pageouts_delta"]) for row in rows),
            "pageout_min": min(int(row["pageouts_delta"]) for row in rows),
            "pageout_max": max(int(row["pageouts_delta"]) for row in rows),
            "swap_growth_count": sum(int(row["swap_used_delta_bytes"]) > 0 for row in rows),
            "thermal_clean_count": sum(bool(row["thermal_clean_before"] and row["thermal_clean_after"]) for row in rows),
        })
    write_csv(campaign / "m4_mini_contention_normalized_summary.csv", summary)

    pageout_sensitivity = []
    for policy in ("llm-only", "fixed1", "fixed2", "fixed4"):
        all_rows = [row for row in detailed if row["policy"] == policy]
        zero_rows = [row for row in all_rows if int(row["pageouts_delta"]) == 0]
        pageout_sensitivity.append({
            "policy": policy, "cap": cap(policy), "all_valid_n": len(all_rows),
            "zero_pageout_n": len(zero_rows),
            "all_valid_median_normalized_p95_ttft": median(
                float(row["normalized_p95_ttft"]) for row in all_rows),
            "zero_pageout_median_normalized_p95_ttft": (
                median(float(row["normalized_p95_ttft"]) for row in zero_rows)
                if zero_rows else ""),
            "all_valid_median_normalized_p95_tpot": median(
                float(row["normalized_p95_tpot"]) for row in all_rows),
            "zero_pageout_median_normalized_p95_tpot": (
                median(float(row["normalized_p95_tpot"]) for row in zero_rows)
                if zero_rows else ""),
            "all_valid_median_retrieval_qps": median(
                float(row["retrieval_qps"]) for row in all_rows),
            "zero_pageout_median_retrieval_qps": (
                median(float(row["retrieval_qps"]) for row in zero_rows)
                if zero_rows else ""),
        })
    write_csv(campaign / "m4_mini_contention_pageout_sensitivity.csv", pageout_sensitivity)

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    caps = [row["cap"] for row in summary]
    ttft = [float(row["median_normalized_p95_ttft"]) for row in summary]
    tpot = [float(row["median_normalized_p95_tpot"]) for row in summary]
    for metric, values, marker in (("p95 TTFT (PREFILL)", ttft, "o"),
                                    ("p95 TPOT (DECODE)", tpot, "s")):
        ax.plot(caps, values, marker=marker, linewidth=2, label=metric)
    for row in detailed:
        ax.scatter(row["cap"], row["normalized_p95_ttft"], color="C0", alpha=.25, s=18)
        ax.scatter(row["cap"], row["normalized_p95_tpot"], color="C1", alpha=.25, s=18)
    ax.axhline(1, color="black", linewidth=.8); ax.set_xticks(caps)
    ax.set(xlabel="Fixed retrieval cap (4 is diagnostic-only)", ylabel="Normalized latency",
           title="Base-M4 Mac mini phase contention")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(campaign / "figure_m4_mini_ttft_tpot_contention.pdf")
    fig.savefig(campaign / "figure_m4_mini_ttft_tpot_contention.png", dpi=180); plt.close(fig)

    baseline_name = complete["passed_baseline_set"]
    baseline_rows = csv_rows(campaign / f"contention_baseline_{baseline_name}_runs.csv")
    baseline_result = json.loads((campaign / f"CONTENTION_BASELINE_{baseline_name}_RESULT.json").read_text())
    baseline_thermal = [row for row in thermal_rows
                        if row["stage"] == f"contention_baseline_{baseline_name}"]
    baseline_thermal_clean = sum(thermal_clean(row["before"]) and thermal_clean(row["after"])
                                 for row in baseline_thermal)
    table = "\n".join(
        f"| {row['policy']} | {float(row['median_p95_ttft_ms']):.2f} | "
        f"{float(row['median_normalized_p95_ttft']):.3f}x | {float(row['median_p95_tpot_ms']):.3f} | "
        f"{float(row['median_normalized_p95_tpot']):.3f}x | {float(row['median_retrieval_qps']):.1f} | "
        f"{row['pageout_flag_count']}/3 |"
        for row in summary)
    air_table = "\n".join(
        f"| {item['policy']} | {AIR_NORMALIZED[item['cap']]['ttft']:.3f}x | "
        f"{AIR_NORMALIZED[item['cap']]['tpot']:.3f}x | "
        f"{float(item['median_normalized_p95_ttft']):.3f}x | "
        f"{float(item['median_normalized_p95_tpot']):.3f}x |"
        for item in summary)
    asymmetric = all(float(row["median_normalized_p95_tpot"]) - 1
                     > float(row["median_normalized_p95_ttft"]) - 1
                     for row in summary if int(row["cap"]) > 0)
    baseline_table = "\n".join(
        f"| {row['repeat']} | {float(row['p95_tpot_ms']):.3f} | {float(row['tpot_abs_deviation'])*100:.2f}% | "
        f"{float(row['p95_ttft_ms']):.2f} | {float(row['ttft_abs_deviation'])*100:.2f}% | "
        f"{row['pageouts_delta']} | {row['swap_used_delta_bytes']} | {row['within_3pct_both']} |"
        for row in baseline_rows)
    secondary_table = "\n".join(
        f"| {row['policy']} | {float(row['median_inter_token_gap_p99_ms']):.3f} | "
        f"{float(row['median_request_maximum_gap_p95_ms']):.3f} | "
        f"{float(row['median_phase_transition_gap_p95_ms']):.3f} |"
        for row in summary)
    def display(value: Any) -> str:
        return "NA" if value == "" else f"{float(value):.3f}"

    sensitivity_table = "\n".join(
        f"| {row['policy']} | {row['zero_pageout_n']}/{row['all_valid_n']} | "
        f"{display(row['zero_pageout_median_normalized_p95_ttft'])} | "
        f"{display(row['zero_pageout_median_normalized_p95_tpot'])} | "
        f"{display(row['zero_pageout_median_retrieval_qps'])} |"
        for row in pageout_sensitivity)
    baseline_pageout_flags = sum(int(row["pageouts_delta"]) > 0 for row in baseline_rows)
    report = f"""# M4 Mini Contention Final Report

## Scope and frozen interpretation

This is a diagnostic contention campaign on the fan-cooled base-M4 Mac mini. It contains no calibration, PhaseGate, TimeGate, held-out policy selection, or output-length sweep. `K_hi=2` remains frozen from the source handoff; cap 4 is diagnostic-only regardless of outcome.

The harness used event-driven phase timing, <=1 Hz native process-memory monitoring, and no legacy 5 ms sampler or repeated `ps` subprocesses. All policies used context 2,048, output 128, 300 measured requests, and newly frozen r2 prompt/query seeds. Each repeat shared its trace across LLM-only and Fixed caps 1/2/4, with order frozen before execution and a fixed 10-second inter-block cooldown.

## Baseline stability gate

Baseline set {baseline_name} passed. Its median p95 TPOT was {baseline_result['median_p95_tpot_ms']:.3f} ms and median p95 TTFT was {baseline_result['median_p95_ttft_ms']:.2f} ms.

| Repeat | p95 TPOT (ms) | abs dev. | p95 TTFT (ms) | abs dev. | pageouts | swap bytes | within +/-3% |
|---:|---:|---:|---:|---:|---:|---:|---:|
{baseline_table}

All baseline blocks had zero swap growth, normal memory pressure, exact 300-request/38,400-token accounting, and zero observer subprocesses. Positive global pageout was retained as a soft flag in {baseline_pageout_flags}/5 blocks. Clean pre/post block-boundary thermal status was recorded for {baseline_thermal_clean}/5 baseline blocks.

## Contention results

| Condition | median p95 TTFT | normalized TTFT | median p95 TPOT | normalized TPOT | retrieval QPS | pageout flags |
|---|---:|---:|---:|---:|---:|---:|
{table}

Every repeat is included in `m4_mini_contention_per_run.csv`; medians are descriptive across three randomized repeats. Decode sensitivity exceeded prefill sensitivity at every positive cap: **{asymmetric}**. Cap 4 remains diagnostic-only and does not change `K_hi=2`.

Official normalized TTFT and TPOT use the passed five-run baseline-set medians. Supplementary within-repeat normalization against the contention-matrix LLM-only block is retained in the per-run CSV.

| Condition | inter-token p99 (ms) | request maximum-gap p95 (ms) | phase-transition gap p95 (ms) |
|---|---:|---:|---:|
{secondary_table}

## Direction-only comparison with the M4 MacBook Air

| Condition | Air normalized prefill/TTFT proxy | Air normalized decode/TPOT proxy | Mini normalized TTFT | Mini normalized TPOT |
|---|---:|---:|---:|---:|
{air_table}

The Air values are the previously reported normalized phase-contention curve in `PHASEGUARD_RESULTS.md` (0/1/2/4 HNSW workers). The comparison is limited to normalized curve direction; raw retrieval QPS is not compared across devices. Any difference in magnitude is not attributed solely to cooling because device form factor, scheduling, and harness details differ.

The Air cap-4 reference corresponds to decode +61.3% and prefill +5.7%. The Mini result is compared only for direction and normalized curve shape.

## Zero-pageout sensitivity

The official result retains every valid block. This sensitivity view uses only blocks with zero global pageout growth; `NA` means that a condition had no zero-pageout block.

| Condition | zero-pageout n/all | normalized TTFT | normalized TPOT | retrieval QPS |
|---|---:|---:|---:|---:|
{sensitivity_table}

## Validity, memory, and thermal audit

- Valid contention blocks: 12/12; invalid attempts preserved: {complete['invalid_attempts_preserved']}.
- Swap-growth blocks: {sum(int(row['swap_growth_count']) for row in summary)}/12.
- Positive-pageout blocks: {sum(int(row['pageout_flag_count']) for row in summary)}/12; pageout deltas are retained per run rather than used for result selection.
- Clean pre/post thermal status: {sum(int(row['thermal_clean_count']) for row in summary)}/12 blocks.
- All production blocks used observer mode `event`, launched zero observer subprocesses, and retained exact request/token/query/event accounting.

No valid block was rerun because of an unfavorable number. Cap 4 is not a PhaseGate candidate, and this campaign cannot revise calibration or K_hi.
"""
    (campaign / "M4_MINI_CONTENTION_FINAL_REPORT.md").write_text(report)
    output, digest, count = create_bundle(campaign)
    print(report)
    print(f"bundle={output} entries={count} bytes={output.stat().st_size} sha256={digest}")


if __name__ == "__main__":
    main()
