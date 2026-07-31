#!/usr/bin/env python3
"""Randomized, resume-safe driver for the single-SLO preliminary check."""
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


def existing_valid(label: str, stage: str) -> int:
    raw = OUT / "raw" / ("single_slo_19ms_cpu_only.jsonl" if stage == "cpu-only"
                          else "single_slo_19ms_runs.jsonl")
    if not raw.exists():
        return 0
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    def is_current_campaign(row: dict[str, object]) -> bool:
        if stage == "cpu-only":
            return True
        resident = row.get("resident_memory_preflight")
        return isinstance(resident, dict) and resident.get("passed") is True

    return sum(
        row.get("status") == "valid"
        and row.get("run_key", "").split("_")[-2] == label
        and is_current_campaign(row)
        for row in rows
    )


def existing_attempts(label: str, stage: str) -> int:
    raw = OUT / "raw" / ("single_slo_19ms_cpu_only.jsonl" if stage == "cpu-only"
                          else "single_slo_19ms_runs.jsonl")
    if not raw.exists():
        return 0
    attempts = []
    for row in (json.loads(line) for line in raw.read_text().splitlines() if line.strip()):
        parts = row.get("run_key", "").split("_")
        if len(parts) >= 2 and parts[-2] == label and parts[-1].startswith("a"):
            try:
                attempts.append(int(parts[-1][1:]))
            except ValueError:
                pass
    return max(attempts, default=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("cpu-only", "main"), required=True)
    ap.add_argument("--valid-repeats", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--allow-fanless-pilot", action="store_true")
    args, extra = ap.parse_known_args()
    if args.valid_repeats < 1 or args.max_attempts < args.valid_repeats:
        raise SystemExit("max-attempts must be at least valid-repeats")
    settings = (["cap1", "cap2", "cap3", "cap4"] if args.stage == "cpu-only"
                else ["cap1", "cap2", "cap3", "cap4", "uncoordinated"])
    attempts = {setting: existing_attempts(setting, args.stage) for setting in settings}
    manifest = {"stage": args.stage, "seed": args.seed, "valid_repeats": args.valid_repeats,
                "max_attempts": args.max_attempts, "started": datetime.now().isoformat(), "blocks": []}
    randomizer = random.Random(args.seed)
    while True:
        pending = [setting for setting in settings if existing_valid(setting, args.stage) < args.valid_repeats
                   and attempts[setting] < args.max_attempts]
        if not pending:
            break
        randomizer.shuffle(pending)
        for setting in pending:
            attempts[setting] += 1
            policy = "uncoordinated" if setting == "uncoordinated" else "phaseguard"
            cap = 4 if setting == "uncoordinated" else int(setting.removeprefix("cap"))
            command = [sys.executable, str(REPO / "scripts/run_single_slo_19ms.py"),
                       "--mode", args.stage, "--policy", policy, "--decode-cap", str(cap),
                       "--attempt", str(attempts[setting]), "--trace-seed",
                       str(args.seed + attempts[setting])]
            if args.allow_battery:
                command.append("--allow-battery")
            if args.allow_fanless_pilot:
                command.append("--allow-fanless-pilot")
            command += extra
            manifest["blocks"].append({"setting": setting, "command": command})
            print("[single-slo]", " ".join(command), flush=True)
            subprocess.run(command, cwd=REPO, check=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    manifest["finished"] = datetime.now().isoformat()
    manifest["valid_counts"] = {setting: existing_valid(setting, args.stage) for setting in settings}
    (OUT / "logs/single_slo_19ms_matrix.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["valid_counts"], indent=2))


if __name__ == "__main__":
    main()
