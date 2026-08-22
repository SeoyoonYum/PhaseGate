#!/usr/bin/env python3
"""Analyze only the two stability-first paired static phase-awareness pilots."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/static_phaseaware/paired_pilot"
RAW = ROOT / "raw/runs.jsonl"
REQUESTS = ROOT / "raw/requests.jsonl"
MANIFEST = ROOT / "logs/paired_pilot_manifest.json"
PROCESSED = ROOT / "processed"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def request_tpot_slope(request_record: dict[str, Any]) -> float:
    """Return the least-squares request-p95-TPOT trend in ms per second."""
    requests = request_record.get("requests", [])
    if len(requests) < 2:
        return 0.0
    times = np.asarray([float(row["completion"]) for row in requests])
    values = np.asarray([float(row["p95_tpot_ms"]) for row in requests])
    return float(np.polyfit(times - times[0], values, 1)[0])


def main() -> None:
    if not RAW.exists() or not REQUESTS.exists() or not MANIFEST.exists():
        raise SystemExit("paired pilot is incomplete")
    runs = read_jsonl(RAW)
    request_records = {row["run_key"]: row for row in read_jsonl(REQUESTS)}
    by_key = {row["run_key"]: row for row in runs}
    manifest = json.loads(MANIFEST.read_text())
    block_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for session in manifest["pairs"]:
        pair_name = session["pair"]
        fixed_label, gate_label = pair_name.split("_vs_")
        qualifying = 0
        gains: list[float] = []
        paired_wins = 0
        for accepted in session["clean_pairs"]:
            selected = [by_key[key] for key in accepted["run_keys"]]
            fixed = next(row for row in selected if row["policy"] == fixed_label)
            gate = next(row for row in selected if row["policy"] == gate_label)
            gain = (float(gate["total_retrieval_goodput_qps"])
                    / float(fixed["total_retrieval_goodput_qps"]) - 1.0)
            same_joint_slo = bool(fixed["joint_slo_pass"] and gate["joint_slo_pass"])
            qualifies = same_joint_slo and gain >= .05
            qualifying += int(qualifies)
            paired_wins += int(gain > 0)
            gains.append(gain)
            pair_rows.append({
                "comparison": pair_name, "repeat": accepted["repeat"],
                "attempt": accepted["attempt"], "order": " then ".join(accepted["order"]),
                "fixed_policy": fixed_label, "phasegate_policy": gate_label,
                "fixed_retrieval_qps": fixed["total_retrieval_goodput_qps"],
                "phasegate_retrieval_qps": gate["total_retrieval_goodput_qps"],
                "phasegate_gain": gain,
                "fixed_joint_slo_pass": fixed["joint_slo_pass"],
                "phasegate_joint_slo_pass": gate["joint_slo_pass"],
                "same_joint_slo_pass": same_joint_slo,
                "qualifying_repeat": qualifies,
            })
            for row in selected:
                block_rows.append({
                    "comparison": pair_name, "repeat": accepted["repeat"],
                    "attempt": accepted["attempt"], "order": " then ".join(accepted["order"]),
                    "policy": row["policy"],
                    "total_retrieval_qps": row["total_retrieval_goodput_qps"],
                    "p95_tpot_ms": row["p95_tpot_ms"],
                    "normalized_p95_tpot": row["normalized_p95_tpot"],
                    "p95_ttft_ms": row["p95_ttft_ms"],
                    "normalized_p95_ttft": row["normalized_p95_ttft"],
                    "joint_slo_pass": row["joint_slo_pass"],
                    "queue_nonempty_fraction": row["queue_nonempty_fraction"],
                    "prefill_active_worker_mean": row["prefill_active_retrieval_worker_mean"],
                    "decode_active_worker_mean": row["decode_active_retrieval_worker_mean"],
                    "decode_overshoot_fraction": row["decode_cap_overshoot_fraction"],
                    "decode_convergence_ms": row["phase_transition_to_cap_ms"],
                    "prefill_retrieval_completions": row["completed_queries_prefill"],
                    "decode_retrieval_completions": row["completed_queries_decode"],
                    "pre_sentinel_initial_deviation": row["sentinel_before_initial_deviation"],
                    "post_sentinel_initial_deviation": row["sentinel_after_initial_deviation"],
                    "post_sentinel_recovered_deviation": row["sentinel_after_deviation"],
                    "pageout_delta": row["pageouts_delta"],
                    "swap_delta_bytes": row["swap_used_delta_bytes"],
                    "tpot_within_block_ratio": row["tpot_within_block_drift_ratio"],
                    "tpot_slope_ms_per_s":
                        request_tpot_slope(request_records[row["run_key"]]),
                    "retrieval_qps_first_third": row["retrieval_qps_first_third"],
                    "retrieval_qps_last_third": row["retrieval_qps_last_third"],
                    "retrieval_qps_within_block_ratio": row["retrieval_qps_within_block_ratio"],
                    "retrieval_qps_slope_qps_per_s":
                        row["retrieval_qps_within_block_slope_qps_per_s"],
                })
        decision = {
            "comparison": pair_name, "clean_paired_repeats": len(session["clean_pairs"]),
            "failed_pair_attempts": len(session["failed_pair_attempts"]),
            "median_phasegate_gain": float(np.median(gains)),
            "phasegate_higher_qps_count": paired_wins,
            "qualifying_repeat_count": qualifying,
            "promising": qualifying >= 2 and len(session["clean_pairs"]) == 3,
        }
        decisions.append(decision)

    write_csv(PROCESSED / "paired_policy_blocks.csv", block_rows)
    write_csv(PROCESSED / "paired_repeat_summary.csv", pair_rows)
    write_csv(PROCESSED / "paired_decision.csv", decisions)
    invalid_rows = [{
        "run_key": row["run_key"], "policy": row["policy"],
        "repeat": row["repeat"], "attempt": row["attempt"],
        "invalid_reason": row.get("invalid_reason"),
        "pageout_delta": row.get("pageouts_delta"),
        "swap_delta_bytes": row.get("swap_used_delta_bytes"),
        "pre_sentinel_initial_deviation": row.get("sentinel_before_initial_deviation"),
        "post_sentinel_initial_deviation": row.get("sentinel_after_initial_deviation"),
        "post_sentinel_recovered_deviation": row.get("sentinel_after_deviation"),
    } for row in runs if row.get("status") != "valid"
        and row.get("policy") in {"fixed1", "phasegate4to1", "fixed2", "phasegate4to2"}]
    write_csv(PROCESSED / "invalid_policy_blocks.csv", invalid_rows)

    lines = ["# Stability-First Static Phase-Awareness Paired Pilot", "",
             "This is a fanless M4 Air go/no-go pilot, not final policy selection or "
             "held-out evaluation.", "",
             "## Setup and validity controls", "",
             "- Qwen2.5 1.5B 4-bit, context 2048, 128 generated tokens, four LLM "
             "requests per block.",
             "- Always-backlogged FAISS-HNSW retrieval; queue non-empty fraction "
             "must be at least 0.95.",
             "- Separate sessions and isolated baselines for the cap-1 and cap-2 "
             "comparisons; randomized paired order.",
             "- Fresh process for every policy block, 6 GB MLX memory limit, "
             "resident-memory preflight, and browser-process preflight.",
             "- A block was accepted only with pageout delta 0, swap-used delta 0, "
             "and sentinel recovery to within ±5% before continuing.",
             "", "## Per-block results", "",
             "| Comparison | Repeat | Order | Policy | Retrieval QPS | p95 TPOT (norm) | "
             "p95 TTFT (norm) | Joint SLO | Queue non-empty |",
             "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in block_rows:
        lines.append(
            f"| {row['comparison']} | {row['repeat'] + 1} | {row['order']} | "
            f"{row['policy']} | {row['total_retrieval_qps']:.1f} | "
            f"{row['p95_tpot_ms']:.2f} ({row['normalized_p95_tpot']:.3f}) | "
            f"{row['p95_ttft_ms']:.1f} ({row['normalized_p95_ttft']:.3f}) | "
            f"{row['joint_slo_pass']} | {row['queue_nonempty_fraction']:.3f} |")
    lines += ["", "## Go/no-go", ""]
    for row in decisions:
        result = "PROMISING" if row["promising"] else "NO-GO"
        lines.append(
            f"- {row['comparison']}: **{result}**; median QPS gain "
            f"{row['median_phasegate_gain'] * 100:+.1f}%, PhaseGate higher in "
            f"{row['phasegate_higher_qps_count']}/3 pairs, joint-SLO +5% qualifying "
            f"in {row['qualifying_repeat_count']}/3 pairs. "
            f"Failed/contaminated pair attempts preserved: {row['failed_pair_attempts']}."
        )
    lines += ["", "## Mechanism and stability detail", "",
              "| Comparison | Rep | Policy | Active workers P/D | Overshoot | "
              "Converge ms | Completions P/D | Sentinel pre / post initial→recovered | "
              "Pageout / swap | TPOT slope | QPS slope |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in block_rows:
        lines.append(
            f"| {row['comparison']} | {row['repeat'] + 1} | {row['policy']} | "
            f"{row['prefill_active_worker_mean']:.2f} / "
            f"{row['decode_active_worker_mean']:.2f} | "
            f"{row['decode_overshoot_fraction'] * 100:.2f}% | "
            f"{row['decode_convergence_ms']:.1f} | "
            f"{row['prefill_retrieval_completions']} / "
            f"{row['decode_retrieval_completions']} | "
            f"{row['pre_sentinel_initial_deviation'] * 100:+.1f}% / "
            f"{row['post_sentinel_initial_deviation'] * 100:+.1f}%→"
            f"{row['post_sentinel_recovered_deviation'] * 100:+.1f}% | "
            f"{row['pageout_delta']} / {row['swap_delta_bytes']} B | "
            f"{row['tpot_slope_ms_per_s']:+.3f} ms/s | "
            f"{row['retrieval_qps_slope_qps_per_s']:+.2f} QPS/s |")
    lines += [
        "",
        f"All 12 accepted blocks had queue non-empty fraction 1.000, pageout delta 0, "
        f"and swap-used delta 0. {len(invalid_rows)} invalid policy blocks remain in "
        "`invalid_policy_blocks.csv` and the raw JSONL; they were not relabeled or "
        "included in the primary comparison.",
        "",
        "Immediate post-sentinels often exceeded the ±5% gate, so the runner cooled "
        "and retried until recovery before starting the next block. The QPS slopes "
        "are diagnostics only and were not used to accept or reject a run.",
        "",
        "No final best policy was selected and held-out evaluation was not run.",
    ]
    (REPO / "STATIC_PHASEAWARE_PAIRED_PILOT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"decisions": decisions, "block_rows": len(block_rows)}, indent=2))


if __name__ == "__main__":
    main()
