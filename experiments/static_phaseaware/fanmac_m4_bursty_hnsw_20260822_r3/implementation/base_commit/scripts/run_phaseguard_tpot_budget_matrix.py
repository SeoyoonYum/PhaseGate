#!/usr/bin/env python3
"""Calibrate and evaluate a PhaseGuard absolute-TPOT CPU-progress budget."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/phaseguard/backlog_validation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("calibration", "evaluation"), required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--target-tpot-ms", type=float, default=19.0)
    ap.add_argument("--selection", type=Path, default=Path(
        "experiments/phaseguard/backlog_validation/processed/tpot_budget_cap.json"))
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--allow-battery", action="store_true")
    args, extra = ap.parse_known_args()
    if args.repeats < 1 or args.target_tpot_ms <= 0:
        raise SystemExit("repeats and target TPOT must be positive")
    selection_path = args.selection if args.selection.is_absolute() else REPO / args.selection

    blocks: list[dict[str, object]] = []
    if args.stage == "calibration":
        for repeat in range(args.repeats):
            for cap in range(1, 5):
                blocks.append({"repeat": repeat, "demand": "high",
                               "policy": "static", "decode_cap": cap})
    else:
        if not selection_path.exists():
            raise SystemExit(
                f"missing {selection_path}; run calibration and analyze_phasegate_backlog.py")
        selection = json.loads(selection_path.read_text())
        if abs(float(selection["target_tpot_ms"]) - args.target_tpot_ms) > 1e-9:
            raise SystemExit("selection target differs from requested target")
        cap = int(selection["decode_cap"])
        for repeat in range(args.repeats):
            for demand in ("low", "medium", "high"):
                blocks.extend((
                    {"repeat": repeat, "demand": demand, "policy": "uncoordinated"},
                    {"repeat": repeat, "demand": demand, "policy": "phaseguard"},
                    {"repeat": repeat, "demand": demand, "policy": "phaseguard",
                     "target_tpot_ms": args.target_tpot_ms, "decode_cap": cap},
                ))
    random.Random(args.seed + (1_000_000 if args.stage == "evaluation" else 0)).shuffle(blocks)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    manifest = {"stage": args.stage, "seed": args.seed,
                "target_tpot_ms": args.target_tpot_ms,
                "started": datetime.now().isoformat(), "blocks": blocks}
    (OUT / "logs" / f"tpot_budget_matrix_{args.stage}.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    for index, block in enumerate(blocks, 1):
        command = [sys.executable, str(REPO / "scripts/run_phasegate_backlog.py"),
                   "--split", args.stage, "--repeat", str(block["repeat"]),
                   "--demand", str(block["demand"]), "--policy", str(block["policy"])]
        if block.get("policy") == "static":
            command += ["--decode-workers", str(block["decode_cap"])]
        if "target_tpot_ms" in block:
            command += ["--target-tpot-ms", str(block["target_tpot_ms"]),
                        "--phaseguard-decode-cap", str(block["decode_cap"])]
        if args.allow_battery:
            command.append("--allow-battery")
        command += extra
        print(f"[{index}/{len(blocks)}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
