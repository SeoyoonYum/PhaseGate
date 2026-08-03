#!/usr/bin/env python3
"""Run the fresh single-index campaign through held-out analysis."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from slo_goodput_common import DEFAULT_CAMPAIGN, REPO


def run(command: list[str], campaign: Path, stage: str) -> None:
    status_path = campaign / "logs/orchestration_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "status": "running",
        "updated": datetime.now().astimezone().isoformat(),
        "command": command,
    }
    status_path.write_text(json.dumps(payload, indent=2) + "\n")
    result = subprocess.run(command, cwd=REPO, check=False)
    payload.update({
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "updated": datetime.now().astimezone().isoformat(),
    })
    status_path.write_text(json.dumps(payload, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--evaluation-repeats", type=int, choices=(5, 7), default=7)
    args = parser.parse_args()
    campaign = args.campaign_dir.resolve()
    common = ["--campaign-dir", str(campaign), "--mem-limit-gb", "5.5"]
    run([sys.executable, str(REPO / "scripts/run_slo_goodput_calibration.py"), *common],
        campaign, "smoke_overhead_baseline_calibration_selection")
    run([sys.executable, str(REPO / "scripts/run_slo_goodput_evaluation.py"), *common,
         "--repeats", str(args.evaluation_repeats)], campaign,
        "baseline_revalidation_and_held_out_evaluation")
    run([sys.executable, str(REPO / "scripts/analyze_slo_goodput_token_tails.py"), *common],
        campaign, "analysis_figures_and_report")
    (campaign / "logs/orchestration_status.json").write_text(json.dumps({
        "stage": "complete",
        "status": "completed",
        "updated": datetime.now().astimezone().isoformat(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
