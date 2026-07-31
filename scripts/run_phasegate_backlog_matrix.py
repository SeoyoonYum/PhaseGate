#!/usr/bin/env python3
"""Randomize and execute calibration or held-out PhaseGate backlog blocks."""
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
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--fixed-map", type=Path,
                    default=Path("experiments/phaseguard/backlog_validation/processed/best_fixed_k.json"))
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--allow-battery", action="store_true")
    args, extra = ap.parse_known_args()
    if args.repeats < 1:
        raise SystemExit("repeats must be positive")
    fixed_path = args.fixed_map if args.fixed_map.is_absolute() else REPO / args.fixed_map
    demands = ("low", "medium", "high")
    blocks: list[tuple[int, str, str, int | None]] = []
    if args.stage == "calibration":
        for repeat in range(args.repeats):
            for demand in demands:
                blocks.append((repeat, demand, "uncoordinated", None))
                # fixed-k=0 cannot complete a retrieval-first request pipeline.
                for workers in range(1, 5):
                    blocks.append((repeat, demand, "fixed", workers))
    else:
        if not fixed_path.exists():
            raise SystemExit(f"missing calibration selection: {fixed_path}")
        fixed = json.loads(fixed_path.read_text())
        for repeat in range(args.repeats):
            for demand in demands:
                for policy in ("serialized", "fixed", "uncoordinated", "static0", "phaseguard"):
                    workers = int(fixed[demand]) if policy == "fixed" else None
                    blocks.append((repeat, demand, policy, workers))
    random.Random(args.seed + (0 if args.stage == "calibration" else 1_000_000)).shuffle(blocks)
    OUT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    manifest = {"stage": args.stage, "seed": args.seed, "started": datetime.now().isoformat(),
                "blocks": [{"repeat": r, "demand": d, "policy": p, "fixed_workers": k}
                           for r, d, p, k in blocks]}
    OUT.joinpath("logs", f"matrix_{args.stage}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for index, (repeat, demand, policy, workers) in enumerate(blocks, 1):
        command = [sys.executable, str(REPO / "scripts/run_phasegate_backlog.py"),
                   "--split", args.stage, "--repeat", str(repeat), "--demand", demand,
                   "--policy", policy]
        if workers is not None:
            command += ["--fixed-workers", str(workers)]
        if args.allow_battery:
            command.append("--allow-battery")
        command += extra
        print(f"[{index}/{len(blocks)}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
