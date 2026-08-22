#!/usr/bin/env python3
"""Strict-SLO paired pilot for the decode-cap-zero operating region only."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from run_static_phaseaware_pilot import (ROOT, block_command, max_attempt, parser,
    raw_path, read_jsonl, valid_count)  # noqa: E402

PAIRS = (
    ("fixed0_vs_phasegate1to0", ("fixed0", "fixed0", 0, 0),
     ("phasegate1to0", "phasegate", 1, 0)),
    ("fixed0_vs_phasegate2to0", ("fixed0", "fixed0", 0, 0),
     ("phasegate2to0", "phasegate", 2, 0)),
    ("fixed0_vs_phasegate4to0", ("fixed0", "fixed0", 0, 0),
     ("phasegate4to0", "phasegate", 4, 0)),
)


def assert_no_browser() -> dict[str, Any]:
    command = ["pgrep", "-ifl",
               r"Google Chrome.app|Chromium.app|Arc.app|Firefox.app|Safari.app/Contents/MacOS/Safari"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    matches = [line for line in result.stdout.splitlines() if line.strip()]
    if matches:
        raise RuntimeError("browser processes must be closed:\n" + "\n".join(matches))
    return {"checked": datetime.now().isoformat(), "browser_matches": matches}


def exact_row(stage: str, label: str, repeat: int, attempt: int) -> dict[str, Any]:
    expected = f"_{label}_r{repeat:02d}_a{attempt:02d}"
    rows = [row for row in read_jsonl(raw_path(stage, False))
            if expected in str(row.get("run_key", ""))]
    if not rows:
        raise RuntimeError(f"missing durable result row for {expected}")
    return rows[-1]


def session_baseline(args: argparse.Namespace, pair_index: int) -> Path:
    repeat, label = 200 + pair_index, "llm-only"
    while valid_count(args.stage, False, label, repeat) < 1:
        attempt = max_attempt(args.stage, False, label, repeat) + 1
        if attempt > args.max_attempts:
            raise RuntimeError(f"could not obtain a clean baseline for pair {pair_index}")
        subprocess.run(block_command(args, label, 0, 0, repeat, attempt), cwd=REPO, check=True)
    rows = [row for row in read_jsonl(raw_path(args.stage, False))
            if row.get("policy") == label and int(row.get("repeat", -1)) == repeat
            and row.get("status") == "valid"]
    row = rows[-1]
    path = ROOT / args.stage / f"baseline_pair{pair_index}.json"
    path.write_text(json.dumps({
        "stage": args.stage, "pair_index": pair_index, "valid_repeats": 1,
        "p95_tpot_ms": row["p95_tpot_ms"], "p95_ttft_ms": row["p95_ttft_ms"],
        "tpot_slo_multiplier": 1.10, "ttft_slo_multiplier": 1.10,
        "run_keys": [row["run_key"]],
    }, indent=2) + "\n")
    return path


def orders(seed: int, first: tuple[str, str, int, int],
           second: tuple[str, str, int, int]) -> list[list[tuple[str, str, int, int]]]:
    rng = random.Random(seed)
    result = []
    for _ in range(3):
        order = [first, second]
        rng.shuffle(order)
        result.append(order)
    if len({order[0][0] for order in result}) == 1:
        result[-1] = list(reversed(result[-1]))
    return result


def prior_attempts(stage: str, labels: tuple[str, str], repeat: int) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in read_jsonl(raw_path(stage, False)):
        label = str(row.get("policy", ""))
        if label in labels and int(row.get("repeat", -1)) == repeat:
            grouped.setdefault(int(row.get("attempt", -1)), {})[label] = row
    return [{"repeat": repeat, "attempt": attempt,
             "run_keys": [group[label]["run_key"] for label in labels],
             "accepted": all(group[label].get("status") == "valid" for label in labels)}
            for attempt, group in sorted(grouped.items()) if all(label in group for label in labels)]


def save_session(stage: str, pair_name: str, session: dict[str, Any]) -> None:
    path = ROOT / stage / "logs" / f"{pair_name}_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, indent=2) + "\n")


def main() -> None:
    ap = parser()
    ap.description = __doc__
    ap.set_defaults(stage="decodecap0_pilot", repeats=3, max_attempts=16,
                    seed=20260801, output_tokens=128, llm_requests=4,
                    sentinel_attempts=10, sentinel_cooldown=30.0,
                    memory_idle_seconds=30.0, min_headroom_gb=5.5,
                    within_block_drift_tolerance=0.10)
    ap.add_argument("--between-session-cooldown", type=float, default=30.0)
    args = ap.parse_args()
    args.stage, args.smoke, args.allow_fanless_pilot = "decodecap0_pilot", False, True
    args.index = args.index if args.index.is_absolute() else REPO / args.index
    if args.repeats != 3 or args.output_tokens != 128 or args.mem_limit_gb != 6.0:
        raise SystemExit("pilot requires exactly 3 repeats, 128 output tokens, and a 6 GB MLX limit")
    manifest: dict[str, Any] = {
        "started": datetime.now().isoformat(), "repeats": 3, "pairs": [],
        "fresh_process_per_policy": True, "memory_limit_gb": args.mem_limit_gb,
        "environment_check": assert_no_browser(),
    }
    for pair_index, (pair_name, fixed, gate) in enumerate(PAIRS, 1):
        if pair_index > 1:
            time.sleep(args.between_session_cooldown)
        args.seed = 20260801 + pair_index * 1_000_000
        args.baseline_file = session_baseline(args, pair_index)
        session: dict[str, Any] = {
            "pair": pair_name, "started": datetime.now().isoformat(),
            "baseline_file": str(args.baseline_file), "environment_check": assert_no_browser(),
            "orders": [], "clean_pairs": [], "failed_pair_attempts": [],
        }
        for repeat, order in enumerate(orders(args.seed, fixed, gate)):
            session["orders"].append([item[0] for item in order])
            labels = (fixed[0], gate[0])
            previous = prior_attempts(args.stage, labels, repeat)
            accepted = [record for record in previous if record["accepted"]]
            if accepted:
                record = accepted[-1]
                record["order"] = [item[0] for item in order]
                session["clean_pairs"].append(record)
                save_session(args.stage, pair_name, session)
                continue
            session["failed_pair_attempts"].extend(
                {**record, "order": [item[0] for item in order]}
                for record in previous if not record["accepted"])
            start = max(max_attempt(args.stage, False, fixed[0], repeat),
                        max_attempt(args.stage, False, gate[0], repeat)) + 1
            for attempt in range(start, args.max_attempts + 1):
                rows: dict[str, dict[str, Any]] = {}
                for label, policy, prefill, decode in order:
                    subprocess.run(block_command(args, policy, prefill, decode, repeat, attempt),
                                   cwd=REPO, check=True)
                    rows[label] = exact_row(args.stage, label, repeat, attempt)
                record = {"repeat": repeat, "attempt": attempt,
                          "order": [item[0] for item in order],
                          "run_keys": [rows[item[0]]["run_key"] for item in order],
                          "accepted": all(row.get("status") == "valid" for row in rows.values())}
                if record["accepted"]:
                    session["clean_pairs"].append(record)
                    save_session(args.stage, pair_name, session)
                    break
                session["failed_pair_attempts"].append(record)
                save_session(args.stage, pair_name, session)
            else:
                raise RuntimeError(f"could not obtain clean {pair_name} repeat {repeat}")
        session["finished"] = datetime.now().isoformat()
        manifest["pairs"].append(session)
        save_session(args.stage, pair_name, session)
    manifest["finished"] = datetime.now().isoformat()
    path = ROOT / args.stage / "logs" / "decodecap0_pilot_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pairs": [{"pair": item["pair"], "clean_pairs": len(item["clean_pairs"])}
                                  for item in manifest["pairs"]]}, indent=2))


if __name__ == "__main__":
    main()
