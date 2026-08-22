#!/usr/bin/env python3
"""Stability-first paired pilot for two fixed-versus-PhaseGate comparisons."""
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
from run_static_phaseaware_pilot import (ROOT, block_command, max_attempt,  # noqa: E402
    parser, raw_path, read_jsonl, valid_count)

PAIRS = (
    ("fixed1_vs_phasegate4to1", ("fixed1", "fixed", 1, 1),
     ("phasegate4to1", "phasegate", 4, 1)),
    ("fixed2_vs_phasegate4to2", ("fixed2", "fixed", 2, 2),
     ("phasegate4to2", "phasegate", 4, 2)),
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
    matches = [row for row in read_jsonl(raw_path(stage, False))
               if expected in str(row.get("run_key", ""))]
    if not matches:
        raise RuntimeError(f"runner did not preserve row for {expected}")
    return matches[-1]


def write_session_baseline(args: argparse.Namespace, pair_index: int) -> Path:
    repeat = 100 + pair_index
    label = "llm-only"
    while valid_count(args.stage, False, label, repeat) < 1:
        attempt = max_attempt(args.stage, False, label, repeat) + 1
        if attempt > args.max_attempts:
            raise RuntimeError(f"could not obtain clean isolated baseline for pair {pair_index}")
        subprocess.run(block_command(args, "llm-only", 0, 0, repeat, attempt),
                       cwd=REPO, check=True)
    rows = [row for row in read_jsonl(raw_path(args.stage, False))
            if row.get("policy") == label and int(row.get("repeat", -1)) == repeat
            and row.get("status") == "valid"]
    row = rows[-1]
    path = ROOT / args.stage / f"baseline_pair{pair_index}.json"
    path.write_text(json.dumps({
        "stage": args.stage, "pair_index": pair_index, "valid_repeats": 1,
        "p95_tpot_ms": row["p95_tpot_ms"], "p95_ttft_ms": row["p95_ttft_ms"],
        "tpot_slo_multiplier": 1.10, "ttft_slo_multiplier": 1.10,
        "run_keys": [row["run_key"]]}, indent=2) + "\n")
    return path


def paired_orders(seed: int, repeats: int,
                  first: tuple[str, str, int, int],
                  second: tuple[str, str, int, int]) -> list[list[tuple[str, str, int, int]]]:
    rng = random.Random(seed)
    orders: list[list[tuple[str, str, int, int]]] = []
    for _ in range(repeats):
        order = [first, second]
        rng.shuffle(order)
        orders.append(order)
    if repeats > 1 and len({order[0][0] for order in orders}) == 1:
        orders[-1] = list(reversed(orders[-1]))
    return orders


def prior_pair_attempts(stage: str, labels: tuple[str, str], repeat: int) -> list[dict[str, Any]]:
    """Reconstruct paired attempts from durable per-block JSONL rows."""
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in read_jsonl(raw_path(stage, False)):
        label = str(row.get("policy", ""))
        if label not in labels or int(row.get("repeat", -1)) != repeat:
            continue
        attempt = int(row.get("attempt", -1))
        grouped.setdefault(attempt, {})[label] = row
    records: list[dict[str, Any]] = []
    for attempt, rows in sorted(grouped.items()):
        if not all(label in rows for label in labels):
            continue
        records.append({
            "repeat": repeat,
            "attempt": attempt,
            "run_keys": [rows[label]["run_key"] for label in labels],
            "accepted": all(rows[label].get("status") == "valid" for label in labels),
        })
    return records


def save_session(session: dict[str, Any], pair_name: str) -> None:
    """Checkpoint session state after every paired attempt."""
    log_dir = ROOT / "paired_pilot" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{pair_name}_manifest.json").write_text(
        json.dumps(session, indent=2) + "\n")


def main() -> None:
    ap = parser()
    ap.description = __doc__
    ap.set_defaults(stage="paired_pilot", repeats=3, max_attempts=8,
                    seed=20260731, output_tokens=128, llm_requests=4,
                    sentinel_attempts=10, sentinel_cooldown=30.0,
                    memory_idle_seconds=30.0, min_headroom_gb=5.5,
                    within_block_drift_tolerance=0.10,
                    within_block_qps_drift_tolerance=0.20)
    ap.add_argument("--between-session-cooldown", type=float, default=30.0)
    args = ap.parse_args()
    args.stage = "paired_pilot"
    args.smoke = False
    args.index = args.index if args.index.is_absolute() else REPO / args.index
    args.allow_fanless_pilot = True
    if args.repeats != 3:
        raise SystemExit("this focused pilot requires exactly three clean paired repeats")
    if args.output_tokens != 128:
        raise SystemExit("the paired pilot fixes output length at 128 tokens")
    if args.mem_limit_gb != 6.0:
        raise SystemExit("the paired pilot preserves the 6 GB MLX limit")
    environment = assert_no_browser()
    manifest: dict[str, Any] = {
        "started": datetime.now().isoformat(), "environment_check": environment,
        "repeats": args.repeats, "pairs": [], "fresh_process_per_policy": True,
        "memory_limit_gb": args.mem_limit_gb,
    }

    for pair_index, (pair_name, fixed, gate) in enumerate(PAIRS, 1):
        if pair_index > 1:
            time.sleep(args.between_session_cooldown)
        environment = assert_no_browser()
        args.seed = 20260731 + pair_index * 1_000_000
        args.baseline_file = write_session_baseline(args, pair_index)
        orders = paired_orders(args.seed, args.repeats, fixed, gate)
        session: dict[str, Any] = {
            "pair": pair_name, "started": datetime.now().isoformat(),
            "baseline_file": str(args.baseline_file), "environment_check": environment,
            "orders": [[item[0] for item in order] for order in orders],
            "clean_pairs": [], "failed_pair_attempts": [],
        }
        for repeat, order in enumerate(orders):
            labels = (fixed[0], gate[0])
            prior = prior_pair_attempts(args.stage, labels, repeat)
            accepted_prior = [record for record in prior if record["accepted"]]
            if accepted_prior:
                record = accepted_prior[-1]
                record["order"] = [item[0] for item in order]
                session["clean_pairs"].append(record)
                save_session(session, pair_name)
                continue
            session["failed_pair_attempts"].extend(
                {**record, "order": [item[0] for item in order]}
                for record in prior if not record["accepted"])
            next_attempt = max(
                max_attempt(args.stage, False, fixed[0], repeat),
                max_attempt(args.stage, False, gate[0], repeat),
            ) + 1
            clean = False
            for attempt in range(next_attempt, args.max_attempts + 1):
                rows: dict[str, dict[str, Any]] = {}
                for label, policy, prefill, decode in order:
                    subprocess.run(block_command(
                        args, policy, prefill, decode, repeat, attempt
                    ), cwd=REPO, check=True)
                    rows[label] = exact_row(args.stage, label, repeat, attempt)
                accepted = all(row.get("status") == "valid" for row in rows.values())
                record = {"repeat": repeat, "attempt": attempt,
                          "order": [item[0] for item in order],
                          "run_keys": [rows[item[0]]["run_key"] for item in order],
                          "accepted": accepted}
                if accepted:
                    session["clean_pairs"].append(record)
                    save_session(session, pair_name)
                    clean = True
                    break
                session["failed_pair_attempts"].append(record)
                save_session(session, pair_name)
            if not clean:
                save_session(session, pair_name)
                raise RuntimeError(f"could not obtain clean {pair_name} repeat {repeat}")
        session["finished"] = datetime.now().isoformat()
        manifest["pairs"].append(session)
        save_session(session, pair_name)
    manifest["finished"] = datetime.now().isoformat()
    log_dir = ROOT / args.stage / "logs"
    (log_dir / "paired_pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pairs": [
        {"pair": item["pair"], "clean_pairs": len(item["clean_pairs"]),
         "failed_pair_attempts": len(item["failed_pair_attempts"])}
        for item in manifest["pairs"]]}, indent=2))


if __name__ == "__main__":
    main()
