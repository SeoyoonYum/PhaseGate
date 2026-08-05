#!/usr/bin/env python3
"""Freeze TimeGate, validate semantics, and run the r5 held-out triplet."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BLOCK = REPO / "scripts/run_static_phaseaware_pilot.py"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.strip()


def parse_policy(name: str) -> tuple[str, int, int]:
    if match := re.fullmatch(r"fixed(\d+)", name):
        cap = int(match.group(1)); return "fixed", cap, cap
    match = re.fullmatch(r"phasegate(\d+)to(\d+)", name)
    if not match: raise ValueError(name)
    return "phasegate", int(match.group(1)), int(match.group(2))


def freeze_schedule(campaign: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    schedule_path = campaign / "timegate_schedule_freeze.json"
    heldout_path = campaign / "HELDOUT_PROTOCOL_FREEZE.json"
    if schedule_path.exists() and heldout_path.exists():
        return json.loads(schedule_path.read_text()), json.loads(heldout_path.read_text())
    selection = json.loads((campaign / "frozen_m4_selection.json").read_text())
    _, high, low = parse_policy(selection["selected_phasegate"])
    timelines = sorted((campaign / "m4_calibration/raw/timelines").glob(
        f"*{selection['selected_phasegate']}*.json"))
    intervals = []
    for path in timelines:
        events = sorted(json.loads(path.read_text())["events"], key=lambda x: x["timestamp"])
        phases = [event for event in events if event["event_type"] in
                  ("phase_enter_prefill", "phase_enter_decode", "phase_exit_or_idle")]
        for current, following in zip(phases, phases[1:]):
            phase = current["event_type"]
            if phase not in ("phase_enter_prefill", "phase_enter_decode"): continue
            duration = float(following["timestamp"]) - float(current["timestamp"])
            if duration > 0:
                intervals.append({"cap": high if phase == "phase_enter_prefill" else low,
                                  "duration_s": duration,
                                  "calibration_source_phase": "PREFILL" if phase.endswith("prefill") else "DECODE"})
    if not intervals: raise RuntimeError("no calibration phase intervals for TimeGate")
    period = sum(item["duration_s"] for item in intervals)
    rng = random.Random(2026088107); offsets = [rng.random() * period for _ in range(7)]
    schedule = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_head(), "construction": "calibration interval replay with frozen circular shift",
        "selected_phasegate": selection["selected_phasegate"], "high_cap": high, "low_cap": low,
        "phase_access_for_cap_selection": False, "phase_state_consulted": False,
        "intervals": intervals, "period_s": period,
        "high_duty_fraction": sum(item["duration_s"] for item in intervals if item["cap"] == high) / period,
        "transition_rate_hz": len(intervals) / period, "offset_seed": 2026088107,
        "heldout_offsets_s": offsets}
    schedule_path.write_text(json.dumps(schedule, indent=2) + "\n")
    fixed_family, fixed_high, fixed_low = parse_policy(selection["selected_fixed"])
    policies = [{"name": selection["selected_fixed"], "family": fixed_family,
                 "high": fixed_high, "low": fixed_low},
                {"name": selection["selected_phasegate"], "family": "phasegate",
                 "high": high, "low": low},
                {"name": f"timegate{high}to{low}", "family": "timegate",
                 "high": high, "low": low}]
    repeats = []
    for repeat in range(7):
        order = list(policies); random.Random(2026088200 + repeat).shuffle(order)
        repeats.append({"repeat": repeat, "prompt_seed": 3820000013 + repeat * 10007,
                        "query_seed": 3820500016 + repeat * 10007,
                        "timegate_offset_s": offsets[repeat], "order": order})
    heldout = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_head(), "primary_B": selection["primary_B"],
        "policies": policies, "paired_randomized_repeats": 7,
        "measured_requests_per_block": 250, "context_tokens": 2048,
        "output_tokens": 128, "observer_mode": "event", "repeats": repeats}
    heldout_path.write_text(json.dumps(heldout, indent=2) + "\n")
    return schedule, heldout


def run(campaign: Path, stage: str, spec: dict[str, Any], repeat: int,
        requests: int, prompt_seed: int, query_seed: int, index: Path, model: Path,
        log: Any, schedule: Path | None = None, offset: float = 0.0) -> dict[str, Any]:
    name = str(spec["name"]); raw = campaign / stage / "raw/runs.jsonl"
    existing = [row for row in read_jsonl(raw)
                if row.get("policy") == name and int(row.get("repeat", -1)) == repeat]
    valid = [row for row in existing if row.get("status") == "valid"]
    if valid: return valid[-1]
    attempt = max([int(row.get("attempt", 0)) for row in existing], default=0) + 1
    if attempt > 2: raise RuntimeError(f"retry exhausted: {stage}/{name}/r{repeat}")
    family = str(spec["family"]); high = int(spec["high"]); low = int(spec["low"])
    command = [sys.executable, str(BLOCK), "--run-one", "--stage", stage,
        "--policy", family, "--observer-mode", "event", "--repeat", str(repeat),
        "--attempt", str(attempt), "--fixed-workers", str(low), "--prefill-cap", str(high),
        "--decode-cap", str(low), "--prompt-seed", str(prompt_seed), "--query-seed", str(query_seed),
        "--model", str(model), "--index", str(index), "--context", "2048",
        "--output-tokens", "128", "--llm-requests", str(requests), "--max-workers", "4",
        "--feeders", "8", "--queries-per-task", "4096", "--chunk", "16",
        "--ef-search", "128", "--top-k", "10", "--memory-sample-interval-s", "1",
        "--warmup-s", "2", "--mem-limit-gb", "5.5", "--min-headroom-gb", "3.0",
        "--memory-idle-seconds", "2", "--sentinel-tolerance", "0.03",
        "--sentinel-cooldown", "2", "--sentinel-attempts", "2", "--sentinel-reps", "2",
        "--sentinel-reference-warmup-s", "120", "--within-block-drift-tolerance", "0.10",
        "--within-block-qps-drift-tolerance", "0.20", "--min-duration-s", "5",
        "--min-completed-queries", "1000", "--baseline-file",
        str(campaign / "r5_normalization_baseline.json")]
    if family == "timegate":
        if schedule is None: raise RuntimeError("missing TimeGate schedule")
        command += ["--timegate-schedule", str(schedule), "--timegate-offset-s", str(offset)]
    env = {**os.environ, "PHASEGATE_CAMPAIGN_ROOT": str(campaign),
           "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = [row for row in read_jsonl(raw)
            if row.get("policy") == name and int(row.get("repeat", -1)) == repeat]
    if not rows or rows[-1].get("status") != "valid":
        if attempt == 1: return run(campaign, stage, spec, repeat, requests, prompt_seed,
            query_seed, index, model, log, schedule, offset)
        raise RuntimeError(f"invalid twice: {stage}/{name}/r{repeat}")
    return rows[-1]


def isolated(campaign: Path, repeat: int, index: Path, model: Path, log: Any) -> dict[str, Any]:
    spec = {"name": "llm-only", "family": "llm-only", "high": 0, "low": 0}
    return run(campaign, "m4_pre_eval_baseline", spec, repeat, 150,
               3810000017 + repeat * 10007, 3810500020 + repeat * 10007,
               index, model, log)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True); parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args(); campaign = args.campaign.resolve(); index = args.index.resolve(); model = args.model.resolve()
    schedule, heldout = freeze_schedule(campaign)
    if git_head() != heldout["repository_commit"]: raise SystemExit("commit differs from held-out freeze")
    schedule_path = campaign / "timegate_schedule_freeze.json"
    with (campaign / "m4_heldout_orchestration.log").open("a") as log:
        high, low = schedule["high_cap"], schedule["low_cap"]
        semantic = [{"name": heldout["policies"][0]["name"], **heldout["policies"][0]},
                    {"name": heldout["policies"][1]["name"], **heldout["policies"][1]},
                    {"name": f"timegate{high}to{low}", "family": "timegate", "high": high, "low": low}]
        semantic_rows = [run(campaign, "m4_timegate_semantic", spec, 0, 10,
            3815000011, 3815500014, index, model, log,
            schedule_path if spec["family"] == "timegate" else None,
            schedule["heldout_offsets_s"][0]) for spec in semantic]
        timegate_row = next(row for row in semantic_rows if row["policy_arg"] == "timegate")
        if timegate_row["timegate_audit"]["phase_state_consulted"]:
            raise RuntimeError("TimeGate consulted phase state")
        pre = [isolated(campaign, repeat, index, model, log) for repeat in range(3)]
        old = json.loads((campaign / "r5_normalization_baseline.json").read_text())
        fresh_tpot = median(float(row["p95_tpot_ms"]) for row in pre)
        fresh_ttft = median(float(row["p95_ttft_ms"]) for row in pre)
        drift = {"fresh_median_p95_tpot_ms": fresh_tpot, "fresh_median_p95_ttft_ms": fresh_ttft,
                 "tpot_relative_drift": fresh_tpot / old["p95_tpot_ms"] - 1,
                 "ttft_relative_drift": fresh_ttft / old["p95_ttft_ms"] - 1}
        (campaign / "PRE_EVAL_BASELINE_DRIFT.json").write_text(json.dumps(drift, indent=2) + "\n")
        if abs(drift["tpot_relative_drift"]) > .03 or abs(drift["ttft_relative_drift"]) > .03:
            raise RuntimeError("pre-evaluation baseline drift exceeds 3%; calibration must be rerun")
        for repeat_spec in heldout["repeats"]:
            for spec in repeat_spec["order"]:
                run(campaign, "m4_heldout", spec, int(repeat_spec["repeat"]), 250,
                    int(repeat_spec["prompt_seed"]), int(repeat_spec["query_seed"]), index, model, log,
                    schedule_path if spec["family"] == "timegate" else None,
                    float(repeat_spec["timegate_offset_s"]))
    (campaign / "M4_HELDOUT_COLLECTION_COMPLETE.json").write_text(json.dumps(
        {"completed_utc": datetime.now(timezone.utc).isoformat(), "repository_commit": git_head(),
         "valid_paired_repeats": 7, "policies": heldout["policies"]}, indent=2) + "\n")


if __name__ == "__main__": main()
