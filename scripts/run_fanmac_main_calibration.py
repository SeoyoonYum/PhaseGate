#!/usr/bin/env python3
"""Run smoke, isolated baseline, phase characterization, and main calibration."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime

from fanmac_main_common import (CALIBRATION_POLICIES, REPO, SMOKE_POLICIES,
    add_common_args, assert_environment, base_namespace, capture_hardware,
    configure_root, ensure_block, resolve_campaign, run_isolated_baseline,
    summarize_and_freeze)


def append_command(campaign, command: list[str]) -> None:
    with (campaign / "commands.log").open("a") as handle:
        handle.write(f"{datetime.now().isoformat()} " + " ".join(command) + "\n")


def smoke(pilot, user, campaign, seed: int) -> None:
    args = base_namespace(pilot, user, "smoke", seed)
    args.smoke = True
    args.memory_idle_seconds = min(args.memory_idle_seconds, 1.0)
    args.sentinel_cooldown = min(args.sentinel_cooldown, 1.0)
    args.min_headroom_gb = min(args.min_headroom_gb, 5.5)
    smoke_baseline = ensure_block(
        pilot, args, ("llm-only", "llm-only", 0, 0), 0, smoke=True)
    baseline = {
        "stage": "smoke", "smoke": True, "performance_result": False,
        "valid_repeats": int(smoke_baseline.get("status") == "valid"),
        "measured_repeats": 1, "p95_tpot_ms": smoke_baseline["p95_tpot_ms"],
        "p95_ttft_ms": smoke_baseline["p95_ttft_ms"],
        "tpot_slo_multiplier": 1.10, "ttft_slo_multiplier": 1.10,
        "run_keys": [smoke_baseline["run_key"]],
    }
    pilot.baseline_path("smoke", True).write_text(json.dumps(baseline, indent=2) + "\n")
    args.baseline_file = pilot.baseline_path("smoke", True)
    results = [ensure_block(pilot, args, item, 0, smoke=True) for item in SMOKE_POLICIES]
    checks = []
    for row in results:
        label = row["policy"]
        passed = bool(row["validity"]["policy_cap_applied"])
        if label == "phasegate4to0":
            passed = passed and int(row["admitted_queries_decode"]) == 0
        checks.append({
            "policy": label, "passed": passed,
            "prefill_cap": row["prefill_cap"], "decode_cap": row["decode_cap"],
            "prefill_active_mean": row["prefill_active_retrieval_worker_mean"],
            "decode_active_mean": row["decode_active_retrieval_worker_mean"],
            "queue_nonempty_fraction": row["queue_nonempty_fraction"],
            "decode_admitted": row["admitted_queries_decode"],
            "transition_to_cap_ms": row["phase_transition_to_cap_ms"],
            "overshoot_fraction": row["decode_cap_overshoot_fraction"],
        })
    payload = {"purpose": "functional semantics only", "baseline": baseline,
               "checks": checks, "all_passed": all(item["passed"] for item in checks)}
    (campaign / "smoke" / "functional_smoke_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    if not payload["all_passed"]:
        raise RuntimeError("functional smoke policy semantics failed")


def characterization(user, campaign, seed: int) -> None:
    env = os.environ.copy()
    env["PHASEGUARD_PROFILE_ROOT"] = str(campaign / "characterization")
    command = [sys.executable, str(REPO / "scripts/run_phaseguard_profile.py"),
               "--index", str(REPO / "experiments/phaseguard/index/hnsw_100k_d384.faiss"),
               "--model", user.model, "--context", str(user.context),
               "--workers", "0,1,2,3,4", "--max-workers", "4",
               "--ef-search", "128", "--top-k", "10", "--decode-steps", "128",
               "--repetitions", "3", "--queries-per-task", "32", "--cooldown", "4",
               "--mem-limit-gb", str(user.mem_limit_gb), "--seed", str(seed),
               "--tag", "fanmac_main"]
    expected = {(repeat, workers, phase) for repeat in range(3)
                for workers in range(5) for phase in ("PREFILL", "DECODE")}
    raw = campaign / "characterization/raw/profile_fanmac_main.jsonl"
    for _ in range(user.max_attempts):
        append_command(campaign, command)
        subprocess.run(command, cwd=REPO, env=env, check=True)
        rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
        valid = {(int(row["repeat"]), int(row["workers"]), row["phase"])
                 for row in rows if row.get("status") == "ok"}
        if valid == expected:
            return
    raise RuntimeError(f"characterization incomplete: {len(valid)}/{len(expected)} valid")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--seed", type=int, default=20260801)
    user = ap.parse_args()
    campaign = resolve_campaign(user.campaign_dir)
    assert_environment()
    pilot = configure_root(campaign)
    capture_hardware(campaign, user.model, user.context, user.output_tokens)
    append_command(campaign, sys.argv)

    smoke(pilot, user, campaign, user.seed - 1)
    baseline = run_isolated_baseline(pilot, user, campaign, user.seed)
    characterization(user, campaign, user.seed + 1_000_000)

    args = base_namespace(pilot, user, "calibration", user.seed + 2_000_000, baseline)
    orders = []
    for repeat in range(3):
        order = list(CALIBRATION_POLICIES)
        random.Random(args.seed + repeat).shuffle(order)
        orders.append([item[0] for item in order])
        for item in order:
            ensure_block(pilot, args, item, repeat)
    matrix = {"seed": args.seed, "repeats": 3, "orders": orders,
              "baseline_file": str(baseline), "policy_grid_frozen_before_run": True}
    (campaign / "calibration/logs/matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    selection = summarize_and_freeze(pilot, campaign, args.seed, baseline)
    print(json.dumps({"campaign": str(campaign),
                      "best_fixed": (selection["best_fixed"] or {}).get("policy"),
                      "best_phasegate": (selection["best_phasegate"] or {}).get("policy")}, indent=2))


if __name__ == "__main__":
    main()
