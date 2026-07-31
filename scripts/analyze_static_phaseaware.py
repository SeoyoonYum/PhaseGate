#!/usr/bin/env python3
"""Select frozen static policies and summarize held-out evaluation."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/static_phaseaware"
PROCESSED = ROOT / "processed"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def median(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def summarize(rows: list[dict[str, Any]], expected_repeats: int,
              allow_invalid_measured: bool = False) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        # Preserve legacy/early failed attempts in raw JSONL; rows that failed
        # before policy metadata was written cannot contribute to a policy group.
        if "policy" in row:
            groups[str(row["policy"])].append(row)
    output: list[dict[str, Any]] = []
    for policy, attempts in sorted(groups.items()):
        valid = [row for row in attempts if row.get("status") == "valid"]
        measured = [row for row in attempts if "p95_tpot_ms" in row]
        source = valid or (measured[-1:] if allow_invalid_measured else [])
        if not source:
            continue
        row = {
            "policy": policy,
            "policy_arg": source[0]["policy_arg"],
            "prefill_cap": source[0]["prefill_cap"],
            "decode_cap": source[0]["decode_cap"],
            "attempts": len(attempts),
            "valid_repeats": len(valid),
            "expected_repeats": expected_repeats,
            "p95_tpot_ms": median(source, "p95_tpot_ms"),
            "p95_ttft_ms": median(source, "p95_ttft_ms"),
            "p95_tpot_norm": median(source, "normalized_p95_tpot"),
            "p95_ttft_norm": median(source, "normalized_p95_ttft"),
            "joint_slo_pass_count": sum(bool(item["joint_slo_pass"]) for item in valid),
            "joint_slo_pass": (
                len(valid) >= expected_repeats
                and all(bool(item["joint_slo_pass"]) for item in valid[:expected_repeats])
            ),
            "retrieval_qps": median(source, "total_retrieval_goodput_qps"),
            "retrieval_qps_min": min(float(item["total_retrieval_goodput_qps"]) for item in source),
            "retrieval_qps_max": max(float(item["total_retrieval_goodput_qps"]) for item in source),
            "raw_19ms_pass_count": sum(bool(item["raw_19ms_tpot_pass"]) for item in valid),
            "queue_nonempty_fraction": median(source, "queue_nonempty_fraction"),
            "phase_transition_to_cap_ms": median(source, "phase_transition_to_cap_ms"),
            "decode_cap_overshoot_fraction": median(source, "decode_cap_overshoot_fraction"),
            "decode_cap_overshoot_worker_mean": median(
                source, "decode_cap_overshoot_worker_mean"),
            "decode_cap_overshoot_worker_max": max(
                float(item["decode_cap_overshoot_worker_max"]) for item in source),
            "admitted_queries_prefill": median(source, "admitted_queries_prefill"),
            "admitted_queries_decode": median(source, "admitted_queries_decode"),
            "completed_queries_prefill": median(source, "completed_queries_prefill"),
            "completed_queries_decode": median(source, "completed_queries_decode"),
            "prefill_retrieval_qps": median(source, "prefill_retrieval_qps"),
            "decode_retrieval_qps": median(source, "decode_retrieval_qps"),
            "pageout_contaminated_count": sum(
                int(item.get("pageouts_delta", 0)) != 0 for item in attempts),
            "swap_contaminated_count": sum(
                int(item.get("swap_used_delta_bytes", 0) or 0) != 0 for item in attempts),
            "thermal_contaminated_count": sum(
                "sentinel" in str(item.get("invalid_reason", ""))
                or "within_block_drift" in str(item.get("invalid_reason", ""))
                for item in attempts),
        }
        output.append(row)
    return output


def select_with_tie_rule(candidates: list[dict[str, Any]], conservative_key: str) -> dict[str, Any] | None:
    feasible = [row for row in candidates if row["joint_slo_pass"]]
    feasible.sort(key=lambda row: int(row[conservative_key]))
    selected = None
    for candidate in feasible:
        if selected is None or candidate["retrieval_qps"] > selected["retrieval_qps"] * 1.03:
            selected = candidate
    return selected


def calibration(smoke: bool) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    name = "smoke_runs.jsonl" if smoke else "runs.jsonl"
    rows = read_jsonl(ROOT / "calibration/raw" / name)
    baseline_file = ROOT / "calibration" / ("baseline_smoke.json" if smoke else "baseline.json")
    expected = int(json.loads(baseline_file.read_text())["valid_repeats"]) if baseline_file.exists() else 3
    summary = summarize(rows, expected, allow_invalid_measured=smoke)
    fixed = [row for row in summary if row["policy_arg"] == "fixed"]
    phase = [row for row in summary if row["policy_arg"] == "phasegate"]
    best_fixed = select_with_tie_rule(fixed, "decode_cap")
    best_phase = select_with_tie_rule(phase, "decode_cap")
    selection = None
    if best_fixed is not None and best_phase is not None:
        selection = {
            "smoke": smoke,
            "selection_data": "calibration only",
            "tie_rule": "require >3% QPS improvement to select a less conservative cap",
            "best_fixed": best_fixed,
            "best_phasegate": best_phase,
        }
        PROCESSED.mkdir(parents=True, exist_ok=True)
        (PROCESSED / ("selection_smoke.json" if smoke else "selection.json")).write_text(
            json.dumps(selection, indent=2) + "\n")
    return summary, selection


def pooled_request_p95(stage: str, smoke: bool, run_keys: set[str],
                       metric: str) -> float | None:
    name = "smoke_requests.jsonl" if smoke else "requests.jsonl"
    records = read_jsonl(ROOT / stage / "raw" / name)
    values = [float(request[metric]) for record in records if record.get("run_key") in run_keys
              for request in record.get("requests", [])]
    return float(np.percentile(values, 95)) if values else None


def evaluation(smoke: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    name = "smoke_runs.jsonl" if smoke else "runs.jsonl"
    rows = read_jsonl(ROOT / "evaluation/raw" / name)
    baseline_file = ROOT / "evaluation" / ("baseline_smoke.json" if smoke else "baseline.json")
    expected = int(json.loads(baseline_file.read_text())["valid_repeats"]) if baseline_file.exists() else 5
    baseline = json.loads(baseline_file.read_text()) if baseline_file.exists() else None
    summary = summarize(rows, expected, allow_invalid_measured=smoke)
    valid_rows = [row for row in rows if row.get("status") == "valid"]
    for group in summary:
        keys = {row["run_key"] for row in valid_rows if row["policy"] == group["policy"]}
        pooled_tpot = pooled_request_p95("evaluation", smoke, keys, "p95_tpot_ms")
        pooled_ttft = pooled_request_p95("evaluation", smoke, keys, "ttft_ms")
        group["aggregate_request_p95_tpot_ms"] = pooled_tpot
        group["aggregate_request_p95_ttft_ms"] = pooled_ttft
        group["aggregate_joint_slo_pass"] = (
            baseline is not None and pooled_tpot is not None and pooled_ttft is not None
            and pooled_tpot <= float(baseline["p95_tpot_ms"]) * 1.10
            and pooled_ttft <= float(baseline["p95_ttft_ms"]) * 1.10
        )
    details: dict[str, Any] = {"paired_wins": None, "paired_repeats": 0, "gain": None}
    selection_path = PROCESSED / ("selection_smoke.json" if smoke else "selection.json")
    if selection_path.exists():
        selected = json.loads(selection_path.read_text())
        fixed_name = selected["best_fixed"]["policy"]
        phase_name = selected["best_phasegate"]["policy"]
        fixed = next((row for row in summary if row["policy"] == fixed_name), None)
        phase = next((row for row in summary if row["policy"] == phase_name), None)
        fixed_runs = {int(row["repeat"]): row for row in valid_rows if row["policy"] == fixed_name}
        phase_runs = {int(row["repeat"]): row for row in valid_rows if row["policy"] == phase_name}
        paired = sorted(set(fixed_runs) & set(phase_runs))
        details = {
            "best_fixed": fixed_name, "best_phasegate": phase_name,
            "paired_repeats": len(paired),
            "paired_wins": sum(
                float(phase_runs[index]["total_retrieval_goodput_qps"])
                > float(fixed_runs[index]["total_retrieval_goodput_qps"]) for index in paired),
            "gain": (
                float(phase["retrieval_qps"]) / float(fixed["retrieval_qps"]) - 1
                if fixed and phase else None),
            "both_joint_slo": bool(
                fixed and phase and fixed["joint_slo_pass"] and phase["joint_slo_pass"]
                and fixed["aggregate_joint_slo_pass"] and phase["aggregate_joint_slo_pass"]),
        }
    return summary, details


FIELDS = [
    "policy", "policy_arg", "prefill_cap", "decode_cap", "attempts", "valid_repeats",
    "expected_repeats", "p95_tpot_ms", "p95_ttft_ms", "p95_tpot_norm", "p95_ttft_norm",
    "joint_slo_pass_count", "joint_slo_pass", "retrieval_qps", "retrieval_qps_min",
    "retrieval_qps_max", "raw_19ms_pass_count", "queue_nonempty_fraction",
    "phase_transition_to_cap_ms", "decode_cap_overshoot_fraction",
    "decode_cap_overshoot_worker_mean", "decode_cap_overshoot_worker_max",
    "admitted_queries_prefill", "admitted_queries_decode", "completed_queries_prefill",
    "completed_queries_decode", "prefill_retrieval_qps", "decode_retrieval_qps",
    "pageout_contaminated_count", "swap_contaminated_count", "thermal_contaminated_count",
    "aggregate_request_p95_tpot_ms", "aggregate_request_p95_ttft_ms",
    "aggregate_joint_slo_pass",
]


def report(cal_smoke: list[dict[str, Any]], selection: dict[str, Any] | None,
           eval_smoke: list[dict[str, Any]], eval_details: dict[str, Any]) -> None:
    if eval_details.get("gain") is None:
        first = ("Under the same TPOT and TTFT SLO, it is not yet known whether the best "
                 "static phase-aware policy achieves higher total retrieval goodput than "
                 "the best fixed policy; only functional smoke validation is complete.")
    else:
        direction = "did" if eval_details["gain"] > 0 else "did not"
        first = (f"Under the same TPOT and TTFT SLO, the smoke-selected static phase-aware "
                 f"policy {direction} achieve higher retrieval goodput, but this is not a "
                 f"research result.")
    lines = [first, "", "# Static Phase-Aware Scheduler Evaluation", "",
             "All data currently summarized below are fanless functional smoke data. "
             "They validate policy semantics and the analysis path, not the research claim.",
             "", "## Calibration smoke", "",
             "| Policy | Prefill cap | Decode cap | p95 TPOT norm | p95 TTFT norm | Joint SLO | Retrieval QPS |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in cal_smoke:
        lines.append(f"| {row['policy']} | {row['prefill_cap']} | {row['decode_cap']} | "
                     f"{row['p95_tpot_norm']:.3f} | {row['p95_ttft_norm']:.3f} | "
                     f"{row['joint_slo_pass_count']}/{row['expected_repeats']} | "
                     f"{row['retrieval_qps']:.1f} |")
    lines += ["", "## Frozen smoke selection", ""]
    if selection:
        lines += [f"- Best Fixed-k: {selection['best_fixed']['policy']}",
                  f"- Best PhaseGate: {selection['best_phasegate']['policy']}"]
    else:
        lines.append("- No selection: the smoke data did not yield both feasible policy classes.")
    lines += ["", "## Held-out smoke", ""]
    if not eval_smoke:
        lines.append("Not run. The handoff requested implementation and smoke validation first.")
    else:
        lines += ["| Policy | p95 TPOT | p95 TTFT | Joint SLO | Retrieval QPS |",
                  "|---|---:|---:|---:|---:|"]
        for row in eval_smoke:
            lines.append(f"| {row['policy']} | {row['p95_tpot_ms']:.2f} | "
                         f"{row['p95_ttft_ms']:.2f} | {row['joint_slo_pass']} | "
                         f"{row['retrieval_qps']:.1f} |")
    lines += ["", "## Interpretation", "",
              "No scheduler benefit is claimed from smoke-scale timings. The primary next step "
              "is three valid calibration repeats per candidate, followed by freezing the two "
              "winners and five paired held-out repeats on new prompt and HNSW traces."]
    (REPO / "STATIC_PHASEAWARE_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cal_primary, selection_primary = calibration(False)
    eval_primary, primary_details = evaluation(False)
    cal_smoke, selection_smoke = calibration(True)
    eval_smoke, smoke_details = evaluation(True)
    write_csv(PROCESSED / "calibration_summary.csv", cal_primary, FIELDS)
    write_csv(PROCESSED / "evaluation_summary.csv", eval_primary, FIELDS)
    write_csv(PROCESSED / "calibration_smoke_summary.csv", cal_smoke, FIELDS)
    write_csv(PROCESSED / "evaluation_smoke_summary.csv", eval_smoke, FIELDS)
    report(cal_smoke, selection_smoke, eval_smoke, smoke_details)
    print(json.dumps({"primary_calibration_groups": len(cal_primary),
                      "primary_evaluation_groups": len(eval_primary),
                      "smoke_calibration_groups": len(cal_smoke),
                      "smoke_evaluation_groups": len(eval_smoke),
                      "primary_selection": selection_primary,
                      "primary_evaluation": primary_details,
                      "smoke_selection": selection_smoke,
                      "smoke_evaluation": smoke_details}, indent=2))


if __name__ == "__main__":
    main()
