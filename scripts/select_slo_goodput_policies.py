#!/usr/bin/env python3
"""Select and freeze SLO-goodput policies using fresh calibration only."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from fanmac_main_common import CALIBRATION_POLICIES, REPO, analysis_eligible, clean_run
from slo_goodput_common import CAMPAIGN_SEEDS, add_args, configure_root, resolve

CONT_FIXED = {"fixed1", "fixed2", "fixed3", "fixed4"}
CONT_GATE = {"phasegate2to1", "phasegate3to1", "phasegate4to1",
             "phasegate3to2", "phasegate4to2", "phasegate4to3"}
ZERO_GATE = {"phasegate1to0", "phasegate2to0", "phasegate3to0", "phasegate4to0"}


def med(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def choose(rows: list[dict[str, Any]], family: str) -> dict[str, Any] | None:
    if not rows:
        return None
    best = max(float(row["median_retrieval_qps"]) for row in rows)
    near = [row for row in rows if float(row["median_retrieval_qps"]) >= best / 1.03]
    if family == "fixed":
        return min(near, key=lambda row: int(row["decode_cap"]))
    return min(near, key=lambda row: (int(row["decode_cap"]), int(row["prefill_cap"]),
                                     str(row["policy"])))


def selection(summary: list[dict[str, Any]], budget: float, tpot_base: float,
              ttft_base: float) -> dict[str, Any]:
    feasible = []
    for row in summary:
        ok = all(float(run["p95_tpot_ms"]) / tpot_base <= budget and
                 float(run["p95_ttft_ms"]) / ttft_base <= budget for run in row["runs"])
        if ok:
            feasible.append(row)
    fixed = choose([row for row in feasible if row["policy"] in CONT_FIXED], "fixed")
    gate = choose([row for row in feasible if row["policy"] in CONT_GATE], "gate")
    zero = choose([row for row in feasible if row["policy"] in ZERO_GATE], "gate")
    fixed0 = next((row for row in feasible if row["policy"] == "fixed0"), None)
    return {"continuous_fixed": fixed, "continuous_phasegate": gate,
            "decode_zero_phasegate": zero, "fixed0_feasible": fixed0 is not None}


def slim(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in ("policy", "policy_arg", "prefill_cap", "decode_cap",
                                      "valid_repeats", "median_normalized_p95_tpot",
                                      "max_normalized_p95_tpot", "median_normalized_p95_ttft",
                                      "max_normalized_p95_ttft", "median_retrieval_qps")}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def matrix_rows(rows: list[dict[str, Any]], policy: str, clean_only: bool = False) -> list[dict[str, Any]]:
    """One recorded block per paired repeat; prefer clean over soft-flagged retries."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("policy") == policy and analysis_eligible(row):
            grouped.setdefault(int(row["repeat"]), []).append(row)
    selected = []
    for repeat in sorted(grouped):
        options = grouped[repeat]
        clean = [row for row in options if clean_run(row)]
        if clean: selected.append(max(clean, key=lambda row: int(row.get("attempt", 0))))
        elif not clean_only: selected.append(max(options, key=lambda row: int(row.get("attempt", 0))))
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); add_args(ap)
    args = ap.parse_args(); campaign = resolve(args.campaign_dir)
    pilot = configure_root(campaign)
    grid_info = json.loads((campaign / "slo_grid_frozen.json").read_text())
    baseline = json.loads((campaign / "isolated_baseline/baseline.json").read_text())
    all_calibration = pilot.read_jsonl(pilot.raw_path("calibration", False))
    summary: list[dict[str, Any]] = []
    for label, policy_arg, prefill, decode in CALIBRATION_POLICIES:
        runs = matrix_rows(all_calibration, label)
        if len(runs) not in (3, 5):
            raise RuntimeError(f"{label} requires 3 or 5 valid repeats, found {len(runs)}")
        summary.append({"policy": label, "policy_arg": policy_arg, "prefill_cap": prefill,
                        "decode_cap": decode, "valid_repeats": len(runs),
                        "median_normalized_p95_tpot": med(runs, "normalized_p95_tpot"),
                        "max_normalized_p95_tpot": max(float(r["normalized_p95_tpot"]) for r in runs),
                        "median_normalized_p95_ttft": med(runs, "normalized_p95_ttft"),
                        "max_normalized_p95_ttft": max(float(r["normalized_p95_ttft"]) for r in runs),
                        "median_retrieval_qps": med(runs, "total_retrieval_goodput_qps"),
                        "runs": runs})
    tpot_base, ttft_base = float(baseline["p95_tpot_ms"]), float(baseline["p95_ttft_ms"])
    tu, fu = float(baseline["tpot_uncertainty_fraction"]), float(baseline["ttft_uncertainty_fraction"])
    perturbations = {
        "nominal": (tpot_base, ttft_base),
        "tpot_baseline_decreased": (tpot_base * (1-tu), ttft_base),
        "tpot_baseline_increased": (tpot_base * (1+tu), ttft_base),
        "ttft_baseline_decreased": (tpot_base, ttft_base * (1-fu)),
        "ttft_baseline_increased": (tpot_base, ttft_base * (1+fu)),
    }
    table, stability = [], []
    for budget in grid_info["frozen_slo_grid"]:
        selected = {name: selection(summary, float(budget), *bases)
                    for name, bases in perturbations.items()}
        nominal = selected["nominal"]
        def name(which: str, family: str) -> str | None:
            row = selected[which][family]
            return None if row is None else str(row["policy"])
        stable = all(name(case, family) == name("nominal", family)
                     for case in selected for family in
                     ("continuous_fixed", "continuous_phasegate", "decode_zero_phasegate"))
        stability.append({"B": budget,
            "nominal_fixed": name("nominal", "continuous_fixed"),
            "nominal_phasegate": name("nominal", "continuous_phasegate"),
            "nominal_decode_zero": name("nominal", "decode_zero_phasegate"),
            **{f"{case}_{family}": name(case, family) for case in selected if case != "nominal"
               for family in ("continuous_fixed", "continuous_phasegate", "decode_zero_phasegate")},
            "selection_stable": stable})
        table.append({"B": budget, "continuous_fixed": slim(nominal["continuous_fixed"]),
                      "continuous_phasegate": slim(nominal["continuous_phasegate"]),
                      "decode_zero_phasegate": slim(nominal["decode_zero_phasegate"]),
                      "fixed0_feasible": nominal["fixed0_feasible"]})
    primary = next((row["B"] for row in table if row["continuous_fixed"] is not None and
                    row["continuous_phasegate"] is not None), None)
    primary_audit = next((row for row in stability if row["B"] == primary), None)
    unstable_names: set[str] = set()
    if primary_audit and not primary_audit["selection_stable"]:
        for key, value in primary_audit.items():
            if (key.endswith("_fixed") or key.endswith("_phasegate")) and value:
                unstable_names.add(str(value))
    write_csv(campaign / "selection_stability_audit.csv", stability)
    run_fields = ("run_key", "status", "invalid_reason", "hard_failure_flags", "soft_flags", "clean_run",
                  "policy", "repeat", "attempt",
                  "llm_requests", "output_tokens", "total_generated_tokens",
                  "inter_token_gap_count", "normalized_p95_tpot", "normalized_p95_ttft",
                  "total_retrieval_goodput_qps", "queue_nonempty_fraction", "pageouts_delta",
                  "swap_used_delta_bytes", "p95_inter_token_gap_ms", "p99_inter_token_gap_ms",
                  "p95_transition_gap_ms", "p99_transition_gap_ms",
                  "tpot_first_last_quarter_p95_ratio", "ttft_first_last_quarter_p95_ratio",
                  "retrieval_qps_first_last_quarter_ratio")
    write_csv(campaign / "calibration_runs.csv",
              [{key: row.get(key) for key in run_fields} for row in all_calibration])
    write_csv(campaign / "calibration_summary.csv",
              [{key: value for key, value in row.items() if key != "runs"} for row in summary])
    write_csv(campaign / "calibration_slo_feasibility.csv", [{
        "B": row["B"], "continuous_fixed": None if row["continuous_fixed"] is None else row["continuous_fixed"]["policy"],
        "continuous_phasegate": None if row["continuous_phasegate"] is None else row["continuous_phasegate"]["policy"],
        "decode_zero_phasegate": None if row["decode_zero_phasegate"] is None else row["decode_zero_phasegate"]["policy"],
        "fixed0_feasible": row["fixed0_feasible"]} for row in table])
    extras_done = all(len(next(row for row in summary if row["policy"] == name)["runs"]) == 5
                      for name in unstable_names)
    if unstable_names and not extras_done:
        request = {"reason": "primary selection changes under observed baseline uncertainty",
                   "primary_continuous_budget": primary, "policies": sorted(unstable_names),
                   "additional_repeats": [3, 4]}
        (campaign / "selection/needs_additional_calibration.json").write_text(
            json.dumps(request, indent=2) + "\n")
        print(json.dumps(request, indent=2)); raise SystemExit(2)
    if unstable_names and extras_done and primary_audit and not primary_audit["selection_stable"]:
        raise SystemExit("primary selection remains fragile after two additional calibration repeats")
    payload = {"frozen_at": datetime.now().astimezone().isoformat(),
        "repository_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
            check=True, capture_output=True, text=True).stdout.strip(),
        "selection_data": "fresh calibration only", "frozen_slo_grid": grid_info["frozen_slo_grid"],
        "B_max": grid_info["B_max"], "primary_continuous_budget": primary,
        "selection_by_slo": table, "fixed0_strict_baseline": "fixed0",
        "baseline": baseline, "baseline_uncertainty": {"tpot_fraction": tu, "ttft_fraction": fu},
        "selection_stability": stability,
        "calibration_seed_base": CAMPAIGN_SEEDS["calibration_prompt_trace"],
        "calibration_policy_order_seed_base": CAMPAIGN_SEEDS["calibration_policy_order"],
        "evaluation_seed_base": CAMPAIGN_SEEDS["evaluation_prompt_trace"],
        "evaluation_policy_order_seed_base": CAMPAIGN_SEEDS["evaluation_policy_order"],
        "tie_rules": {"within_qps_fraction": .03, "fixed": "smaller k",
                      "phasegate": "smaller decode cap, then prefill cap, then name"},
        "calibration_summary": [{key: value for key, value in row.items() if key != "runs"}
                                for row in summary]}
    frozen = campaign / "frozen_slo_selection.json"
    frozen.write_text(json.dumps(payload, indent=2) + "\n")
    (campaign / "selection/frozen_slo_selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
