#!/usr/bin/env python3
"""Run held-out paired blocks after calibration choices are frozen."""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from run_static_phaseaware_pilot import (ROOT, baseline_path, block_command,  # noqa: E402
    max_attempt, parser, valid_count, write_baseline)


def main() -> None:
    ap = parser()
    ap.description = __doc__
    ap.set_defaults(stage="evaluation", repeats=5, seed=20270731)
    ap.add_argument("--selection", type=Path)
    args = ap.parse_args()
    args.stage = "evaluation"
    if args.selection is None:
        args.selection = ROOT / "processed" / (
            "selection_smoke.json" if args.smoke else "selection.json")
    elif not args.selection.is_absolute():
        args.selection = REPO / args.selection
    args.index = args.index if args.index.is_absolute() else REPO / args.index
    if args.smoke:
        args.context = min(args.context, 512)
        args.output_tokens = min(args.output_tokens, 16)
        args.llm_requests = min(args.llm_requests, 3)
        args.memory_idle_seconds = min(args.memory_idle_seconds, 1.0)
        args.sentinel_cooldown = min(args.sentinel_cooldown, 1.0)
        args.sentinel_reps = 1
        args.min_duration_s = min(args.min_duration_s, 0.1)
        args.min_completed_queries = min(args.min_completed_queries, 64)
        args.min_headroom_gb = min(args.min_headroom_gb, 5.5)
        args.within_block_drift_tolerance = max(args.within_block_drift_tolerance, 0.30)
    if not args.selection.exists():
        raise SystemExit(f"missing frozen calibration selection: {args.selection}")
    selected = json.loads(args.selection.read_text())

    for repeat in range(args.repeats):
        while valid_count(args.stage, args.smoke, "llm-only", repeat) < 1:
            attempt = max_attempt(args.stage, args.smoke, "llm-only", repeat) + 1
            if attempt > args.max_attempts:
                raise RuntimeError(f"could not obtain evaluation baseline repeat {repeat}")
            subprocess.run(block_command(args, "llm-only", 0, 0, repeat, attempt),
                           cwd=REPO, check=True)
    baseline = write_baseline(args.stage, args.smoke, args.repeats)

    fixed_cap = int(selected["best_fixed"]["decode_cap"])
    phase_prefill = int(selected["best_phasegate"]["prefill_cap"])
    phase_decode = int(selected["best_phasegate"]["decode_cap"])
    policy_items = [
        (f"fixed{fixed_cap}", "fixed", fixed_cap, fixed_cap),
        (f"phasegate{phase_prefill}to{phase_decode}", "phasegate",
         phase_prefill, phase_decode),
        ("fixed4", "fixed", 4, 4),
    ]
    policy_items = list(dict.fromkeys(policy_items))
    blocks = [(repeat, item) for repeat in range(args.repeats) for item in policy_items]
    random.Random(args.seed).shuffle(blocks)
    for repeat, (label, policy, prefill, decode) in blocks:
        while valid_count(args.stage, args.smoke, label, repeat) < 1:
            attempt = max_attempt(args.stage, args.smoke, label, repeat) + 1
            if attempt > args.max_attempts:
                break
            subprocess.run(block_command(args, policy, prefill, decode, repeat, attempt),
                           cwd=REPO, check=True)

    log_dir = ROOT / "evaluation" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / ("smoke_matrix.json" if args.smoke else "matrix.json")).write_text(
        json.dumps({"seed": args.seed, "repeats": args.repeats,
                    "selection_file": str(args.selection), "selection": selected,
                    "baseline": baseline,
                    "randomized_blocks": [{"repeat": repeat, "policy": item[0]}
                                          for repeat, item in blocks]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
