#!/usr/bin/env python3
"""Freeze the token-tail campaign SLO grid from the prior clean calibration."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / "experiments/static_phaseaware/fanmac_main_apple_m2_pro_20260801/calibration_runs.csv"
POLICIES = ("fixed1", "phasegate4to1", "fixed2", "phasegate4to2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-dir", type=Path, required=True)
    ap.add_argument("--old-calibration", type=Path, default=OLD)
    args = ap.parse_args()
    campaign = args.campaign_dir if args.campaign_dir.is_absolute() else REPO / args.campaign_dir
    source = args.old_calibration if args.old_calibration.is_absolute() else REPO / args.old_calibration
    campaign.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(source.open()))
    extracted: dict[str, dict[str, object]] = {}
    all_values: list[float] = []
    for policy in POLICIES:
        valid = [row for row in rows if row["policy"] == policy and row["status"] == "valid"]
        tpot = [float(row["normalized_p95_tpot"]) for row in valid]
        ttft = [float(row["normalized_p95_ttft"]) for row in valid]
        if not valid:
            raise RuntimeError(f"no valid old calibration rows for {policy}")
        all_values.extend(tpot + ttft)
        extracted[policy] = {
            "valid_runs": len(valid), "normalized_tpot": tpot, "normalized_ttft": ttft,
            "tpot_median": float(np.median(tpot)), "tpot_max": max(tpot),
            "ttft_median": float(np.median(ttft)), "ttft_max": max(ttft),
        }
    relevant_max = max(all_values)
    rounded = math.ceil((relevant_max - 1e-12) / .05) * .05
    b_max = round(min(1.50, max(1.30, rounded)), 2)
    if relevant_max > 1.50:
        raise SystemExit(f"relevant_max={relevant_max:.9f} exceeds predeclared 1.50 cap")
    grid = [round(1.10 + .05 * i, 2) for i in range(round((b_max - 1.10) / .05) + 1)]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                            capture_output=True, text=True).stdout.strip()
    frozen_at = datetime.now().astimezone().isoformat()
    payload = {"source": str(source), "source_campaign_commit":
               "ff1c4e8af36df40b0ac6b744fbf9162ded5f7857", "audit_commit": commit,
               "frozen_at": frozen_at, "extracted": extracted,
               "relevant_max": relevant_max, "B_max": b_max, "frozen_slo_grid": grid,
               "frozen_before_new_calibration": True, "frozen_before_new_evaluation": True}
    (campaign / "slo_grid_frozen.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# SLO Grid Audit", "", f"- Timestamp: `{frozen_at}`",
             f"- Repository commit: `{commit}`", f"- Source: `{source.relative_to(REPO)}`",
             "- The grid was frozen before the new calibration and held-out evaluation.", "",
             "| Policy | Valid runs | Normalized TPOT values | TPOT median/max | "
             "Normalized TTFT values | TTFT median/max |", "|---|---:|---|---|---|---|"]
    for policy in POLICIES:
        row = extracted[policy]
        tv = ", ".join(f"{x:.9f}" for x in row["normalized_tpot"])
        fv = ", ".join(f"{x:.9f}" for x in row["normalized_ttft"])
        lines.append(f"| {policy} | {row['valid_runs']} | {tv} | "
                     f"{row['tpot_median']:.9f} / {row['tpot_max']:.9f} | {fv} | "
                     f"{row['ttft_median']:.9f} / {row['ttft_max']:.9f} |")
    lines += ["", f"`relevant_max = {relevant_max:.12f}`", "",
              f"`B_max = {b_max:.2f}`", "",
              "Frozen grid: `{" + ", ".join(f"{x:.2f}" for x in grid) + "}`", "",
              "The extension is mechanical from old calibration extrema; no new held-out result "
              "was inspected or used to choose a favorable budget."]
    (campaign / "slo_grid_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
