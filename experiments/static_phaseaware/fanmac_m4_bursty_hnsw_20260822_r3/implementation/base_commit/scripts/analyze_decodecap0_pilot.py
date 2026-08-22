#!/usr/bin/env python3
"""Analyze the strict-SLO decode-cap-zero paired pilot without policy selection."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/static_phaseaware/decodecap0_pilot"
RAW = ROOT / "raw/runs.jsonl"
MANIFEST = ROOT / "logs/decodecap0_pilot_manifest.json"
PROCESSED = ROOT / "processed"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def median(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def main() -> None:
    if not RAW.exists() or not MANIFEST.exists():
        raise SystemExit("decode-cap-zero pilot is incomplete")
    runs = read_jsonl(RAW)
    by_key = {row["run_key"]: row for row in runs}
    manifest = json.loads(MANIFEST.read_text())
    blocks: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    policies: dict[str, list[dict[str, Any]]] = {"fixed0": []}

    for session in manifest["pairs"]:
        comparison = session["pair"]
        fixed_label, gate_label = comparison.split("_vs_")
        policies.setdefault(gate_label, [])
        for accepted in session["clean_pairs"]:
            selected = [by_key[key] for key in accepted["run_keys"]]
            fixed = next(row for row in selected if row["policy"] == fixed_label)
            gate = next(row for row in selected if row["policy"] == gate_label)
            gain = float(gate["total_retrieval_goodput_qps"]) - float(fixed["total_retrieval_goodput_qps"])
            paired.append({
                "comparison": comparison, "repeat": accepted["repeat"],
                "attempt": accepted["attempt"], "order": " then ".join(accepted["order"]),
                "fixed0_qps": fixed["total_retrieval_goodput_qps"],
                "phasegate_qps": gate["total_retrieval_goodput_qps"],
                "additional_qps": gain,
                "fixed0_joint_slo": fixed["joint_slo_pass"],
                "phasegate_joint_slo": gate["joint_slo_pass"],
                "fixed0_decode_admitted": fixed["admitted_queries_decode"],
                "phasegate_decode_admitted": gate["admitted_queries_decode"],
            })
            for row in selected:
                policies.setdefault(row["policy"], []).append(row)
                blocks.append({
                    "comparison": comparison, "repeat": accepted["repeat"],
                    "attempt": accepted["attempt"], "order": " then ".join(accepted["order"]),
                    "policy": row["policy"], "total_retrieval_qps": row["total_retrieval_goodput_qps"],
                    "p95_tpot_ms": row["p95_tpot_ms"], "normalized_p95_tpot": row["normalized_p95_tpot"],
                    "p95_ttft_ms": row["p95_ttft_ms"], "normalized_p95_ttft": row["normalized_p95_ttft"],
                    "joint_slo_pass": row["joint_slo_pass"],
                    "queue_nonempty_fraction": row["queue_nonempty_fraction"],
                    "prefill_active_worker_mean": row["prefill_active_retrieval_worker_mean"],
                    "prefill_active_worker_p95": row["prefill_active_retrieval_worker_p95"],
                    "decode_active_worker_mean": row["decode_active_retrieval_worker_mean"],
                    "decode_active_worker_p95": row["decode_active_retrieval_worker_p95"],
                    "admitted_queries_prefill": row["admitted_queries_prefill"],
                    "admitted_queries_decode": row["admitted_queries_decode"],
                    "completed_queries_prefill": row["completed_queries_prefill"],
                    "completed_queries_decode": row["completed_queries_decode"],
                    "decode_transition_to_zero_ms": row["phase_transition_to_cap_ms"],
                    "decode_nonzero_worker_fraction": row["decode_overlap_fraction"],
                    "decode_overshoot_worker_mean": row["decode_cap_overshoot_worker_mean"],
                    "decode_overshoot_worker_max": row["decode_cap_overshoot_worker_max"],
                    "pre_sentinel_deviation": row["sentinel_before_initial_deviation"],
                    "post_sentinel_initial_deviation": row["sentinel_after_initial_deviation"],
                    "post_sentinel_recovered_deviation": row["sentinel_after_deviation"],
                    "tpot_slope_ms_per_s": row["tpot_within_block_slope_ms_per_s"],
                    "retrieval_qps_slope_qps_per_s": row["retrieval_qps_within_block_slope_qps_per_s"],
                    "pageout_delta": row["pageouts_delta"], "swap_delta_bytes": row["swap_used_delta_bytes"],
                    "resident_memory_bytes": row["resident_memory_bytes"],
                    "peak_resident_memory_bytes": row["peak_resident_memory_bytes"],
                    "invalid_reason": row["invalid_reason"],
                })

    summaries: list[dict[str, Any]] = []
    for policy, rows in policies.items():
        if not rows:
            continue
        policy_pairs = [row for row in paired if row["comparison"].endswith(f"_vs_{policy}")]
        additions = [float(row["additional_qps"]) for row in policy_pairs]
        feasible = (len(rows) >= 3 and all(bool(row["joint_slo_pass"]) for row in rows)
                    and all(int(row["admitted_queries_decode"]) == 0 for row in rows))
        summaries.append({
            "policy": policy, "clean_blocks": len(rows),
            "median_normalized_p95_tpot": median(rows, "normalized_p95_tpot"),
            "median_normalized_p95_ttft": median(rows, "normalized_p95_ttft"),
            "joint_slo_pass_count": sum(bool(row["joint_slo_pass"]) for row in rows),
            "median_total_retrieval_qps": median(rows, "total_retrieval_goodput_qps"),
            "median_additional_qps_vs_fixed0": float(np.median(additions)) if additions else 0.0,
            "all_decode_admission_zero": all(int(row["admitted_queries_decode"]) == 0 for row in rows),
            "feasible": feasible,
            "thermal_risk_negative_qps_slope_all": all(
                float(row["retrieval_qps_within_block_slope_qps_per_s"]) < 0 for row in rows),
        })

    invalid = [{"run_key": row["run_key"], "policy": row["policy"], "repeat": row["repeat"],
                "attempt": row["attempt"], "invalid_reason": row.get("invalid_reason"),
                "pageout_delta": row.get("pageouts_delta"), "swap_delta_bytes": row.get("swap_used_delta_bytes")}
               for row in runs if row.get("status") != "valid"]
    write_csv(PROCESSED / "decodecap0_policy_blocks.csv", blocks)
    write_csv(PROCESSED / "decodecap0_pair_summary.csv", paired)
    write_csv(PROCESSED / "decodecap0_policy_summary.csv", summaries)
    write_csv(PROCESSED / "decodecap0_invalid_blocks.csv", invalid)

    gates = [row for row in summaries if row["policy"].startswith("phasegate")]
    positive = [row for row in gates if row["feasible"] and row["median_additional_qps_vs_fixed0"] > 0]
    answer = ("Yes." if positive else "No.")
    lines = ["# Decode-Cap-0 Strict-SLO Pilot", "",
             f"{answer} Under the strict 1.10× TPOT and TTFT SLO, prefill-only CPU "
             "execution did " + ("provide" if positive else "not provide") +
             " useful retrieval goodput beyond Fixed-0 in this fanless-M4 pilot.", "",
             "This is a focused paired go/no-go experiment: no calibration matrix, held-out "
             "evaluation, adaptive controller, or decode-cap-positive retest was run.", "",
             "| Policy | TPOT norm | TTFT norm | Joint SLO | Retrieval QPS | Additional QPS vs Fixed-0 |",
             "|---|---:|---:|---:|---:|---:|"]
    order = ["fixed0", "phasegate1to0", "phasegate2to0", "phasegate4to0"]
    by_policy = {row["policy"]: row for row in summaries}
    for policy in order:
        row = by_policy.get(policy)
        if row is None:
            continue
        additional = "baseline" if policy == "fixed0" else f"{row['median_additional_qps_vs_fixed0']:+.1f}"
        lines.append(f"| {policy} | {row['median_normalized_p95_tpot']:.3f} | "
                     f"{row['median_normalized_p95_ttft']:.3f} | "
                     f"{row['joint_slo_pass_count']}/{row['clean_blocks']} | "
                     f"{row['median_total_retrieval_qps']:.1f} | {additional} |")
    lines += ["", "## Interpretation", ""]
    for row in gates:
        lines.append(f"- {row['policy']}: feasible={row['feasible']}; decode admission zero="
                     f"{row['all_decode_admission_zero']}; paired median additional QPS="
                     f"{row['median_additional_qps_vs_fixed0']:+.1f}; persistent negative QPS "
                     f"slope thermal-risk flag={row['thermal_risk_negative_qps_slope_all']}.")
    lines += ["", "## Separation from the prior decode-cap-positive evidence", "",
              "1. Latency-matched: the prior `4→1` and `4→2` policies improved retrieval "
              "QPS at nearly identical TPOT relative to their Fixed counterparts, but both "
              "operating points were outside the strict 1.10× TPOT SLO.",
              "2. Strict-SLO: this report evaluates only decode cap zero and does not combine "
              "its result with the decode-cap-positive comparison."]
    (REPO / "DECODECAP0_PILOT_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"positive_policies": [row["policy"] for row in positive],
                      "policy_summaries": summaries, "accepted_blocks": len(blocks)}, indent=2))


if __name__ == "__main__":
    main()
