#!/usr/bin/env python3
"""Run smoke, logging gate, fresh baseline, and full SLO-goodput calibration."""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from fanmac_main_common import CALIBRATION_POLICIES, SMOKE_POLICIES, REPO, ensure_block
from slo_goodput_common import (add_args, append_command, assert_environment, capture_hardware,
                                CAMPAIGN_SEEDS, configure_root, item_by_name, namespace, resolve,
                                write_fresh_baseline)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def old_ci_audit(campaign: Path) -> None:
    source = REPO / "experiments/static_phaseaware/fanmac_main_apple_m2_pro_20260801/paired_evaluation_details.csv"
    rows = list(csv.DictReader(source.open()))
    lines = ["# Previous Confidence-Interval Audit", "",
             "The prior analyzer formed one QPS gain per paired policy-block repeat and bootstrapped "
             "those five run-level paired gains (10,000 samples). It did not resample requests, "
             "queries, or tokens; therefore the prior CIs do not contain pseudoreplication.", ""]
    for comparison in ("fixed1_vs_phasegate4to1", "fixed2_vs_phasegate4to2"):
        gains = [float(row["paired_gain"]) for row in rows if row["comparison"] == comparison]
        lines += [f"- `{comparison}`: {len(gains)} paired run-level gains: " +
                  ", ".join(f"{gain:+.6%}" for gain in gains)]
    lines += ["", "No old raw result was modified or deleted."]
    (campaign / "previous_ci_audit.md").write_text("\n".join(lines) + "\n")


def smoke(pilot, user, campaign: Path) -> None:
    args = namespace(pilot, user, "smoke", CAMPAIGN_SEEDS["smoke_prompt_trace"], 3)
    args.smoke = True; args.memory_idle_seconds = 1; args.sentinel_cooldown = 1
    args.min_headroom_gb = min(args.min_headroom_gb, 5.5)
    baseline_row = ensure_block(pilot, args, ("llm-only", "llm-only", 0, 0), 0, smoke=True)
    smoke_baseline = {"p95_tpot_ms": baseline_row["p95_tpot_ms"],
                      "p95_ttft_ms": baseline_row["p95_ttft_ms"],
                      "median_inter_token_gap_ms": baseline_row["p50_tpot_ms"]}
    path = campaign / "smoke/baseline_smoke.json"
    path.write_text(json.dumps(smoke_baseline, indent=2) + "\n"); args.baseline_file = path
    checks = []
    for item in SMOKE_POLICIES:
        row = ensure_block(pilot, args, item, 0, smoke=True)
        expected_tokens = int(row["output_tokens"])
        passed = (bool(row["validity"]["policy_cap_applied"])
                  and bool(row["validity"]["token_timestamps_valid"])
                  and int(row["token_timestamp_count"]) == int(row["llm_requests"]) * expected_tokens)
        if item[0] == "phasegate4to0": passed &= int(row["admitted_queries_decode"]) == 0
        checks.append({"policy": item[0], "passed": bool(passed),
                       "tokens_per_request": expected_tokens,
                       "token_timestamp_count": row["token_timestamp_count"],
                       "prefill_active_mean": row["prefill_active_retrieval_worker_mean"],
                       "decode_active_mean": row["decode_active_retrieval_worker_mean"],
                       "decode_admitted": row["admitted_queries_decode"],
                       "queue_nonempty_fraction": row["queue_nonempty_fraction"]})
    payload = {"purpose": "functional semantics and token schema only", "checks": checks,
               "all_passed": all(row["passed"] for row in checks),
               "per_token_synchronous_io": False,
               "persistence": "one gzip JSONL batch flush after each policy block"}
    (campaign / "smoke/functional_smoke_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    if not payload["all_passed"]: raise RuntimeError("functional smoke failed")


def overhead(pilot, user, campaign: Path) -> None:
    all_rows = []
    for cohort, requests in ((0, 24), (1, 50)):
        paired = []
        for offset in range(3):
            repeat = cohort * 3 + offset
            variants = [("logging-disabled", False), ("logging-enabled", True)]
            if repeat % 2: variants.reverse()
            for label, enabled in variants:
                args = namespace(pilot, user, "token_logging_overhead",
                                 CAMPAIGN_SEEDS["logging_prompt_trace"], requests)
                args.run_label = label; args.token_timestamp_logging = enabled
                args.within_block_drift_tolerance = .10
                row = ensure_block(pilot, args, (label, "llm-only", 0, 0), repeat)
                paired.append(row)
        by = {(int(row["repeat"]), row["policy"]): row for row in paired}
        details = []
        for repeat in range(cohort * 3, cohort * 3 + 3):
            off, on = by[(repeat, "logging-disabled")], by[(repeat, "logging-enabled")]
            details.append({"cohort": cohort, "repeat": repeat, "requests": requests,
                "disabled_p95_tpot_ms": off["p95_tpot_ms"], "enabled_p95_tpot_ms": on["p95_tpot_ms"],
                "tpot_overhead": float(on["p95_tpot_ms"])/float(off["p95_tpot_ms"])-1,
                "disabled_p95_ttft_ms": off["p95_ttft_ms"], "enabled_p95_ttft_ms": on["p95_ttft_ms"],
                "ttft_overhead": float(on["p95_ttft_ms"])/float(off["p95_ttft_ms"])-1,
                "disabled_throughput": off["llm_request_throughput_rps"],
                "enabled_throughput": on["llm_request_throughput_rps"],
                "throughput_overhead": 1-float(on["llm_request_throughput_rps"])/float(off["llm_request_throughput_rps"]),
                "disabled_wall_s": off["wall_clock_with_logging_s"], "enabled_wall_s": on["wall_clock_with_logging_s"],
                "wall_overhead": float(on["wall_clock_with_logging_s"])/float(off["wall_clock_with_logging_s"])-1,
                "enabled_flush_ms": on["token_log_flush_ms"]})
        all_rows.extend(details)
        median_wall = float(np.median([row["wall_overhead"] for row in details]))
        if median_wall <= .01:
            write_csv(campaign / "token_logging_overhead.csv", all_rows)
            (campaign / "token_logging_overhead.md").write_text(
                "# Token Logging Overhead\n\n"
                f"The accepted paired cohort used {requests} requests and 3 paired repeats. "
                f"Median wall-clock overhead was {median_wall:+.3%}, within the predeclared 1% limit.\n\n"
                "Token timestamps were accumulated in memory and flushed once per block as compressed JSONL; "
                "there was no synchronous per-token disk I/O.\n")
            return
    write_csv(campaign / "token_logging_overhead.csv", all_rows)
    raise RuntimeError("token logging median wall-clock overhead exceeded 1% in both cohorts")


def baseline(pilot, user, campaign: Path) -> Path:
    for cohort in range(3):
        repeat_ids = list(range(cohort * 7, cohort * 7 + 7))
        args = namespace(pilot, user, "isolated_baseline",
                         CAMPAIGN_SEEDS["baseline_prompt_trace"], 200)
        for repeat in repeat_ids:
            ensure_block(pilot, args, ("llm-only", "llm-only", 0, 0), repeat)
        payload = write_fresh_baseline(pilot, campaign, repeat_ids)
        if payload["variation_within_3pct"]:
            return campaign / "isolated_baseline/baseline.json"
        time.sleep(user.sentinel_cooldown)
    raise RuntimeError("isolated baseline remained unstable after three seven-run cohorts")


def calibration(pilot, user, campaign: Path, baseline_path: Path) -> None:
    args = namespace(pilot, user, "calibration",
                     CAMPAIGN_SEEDS["calibration_prompt_trace"], 100, baseline_path)
    (campaign / "calibration/logs").mkdir(parents=True, exist_ok=True)
    orders = []
    for repeat in range(3):
        order = list(CALIBRATION_POLICIES)
        random.Random(CAMPAIGN_SEEDS["calibration_policy_order"] + repeat).shuffle(order)
        orders.append([item[0] for item in order])
    (campaign / "calibration/logs/matrix.json").write_text(json.dumps({
        "prompt_seed_base": args.seed,
        "query_seed_offset": 5000,
        "policy_order_seed_base": CAMPAIGN_SEEDS["calibration_policy_order"],
        "repeats": 3, "orders": orders,
        "requests_per_block": 100, "tokens_per_request": 128,
        "policy_grid_frozen_before_run": True}, indent=2) + "\n")
    for repeat, order_names in enumerate(orders):
        for name in order_names:
            ensure_block(pilot, args, item_by_name(name), repeat)
            valid = pilot.read_jsonl(pilot.raw_path("calibration", False))
            valid_count = sum(row.get("status") == "valid" for row in valid)
            (campaign / "calibration/progress.json").write_text(json.dumps({
                "valid_blocks": valid_count, "target_valid_blocks": 45,
                "last_policy": name, "last_repeat": repeat,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "resume_source": "calibration/raw/runs.jsonl",
            }, indent=2) + "\n")
    select = [sys.executable, str(REPO / "scripts/select_slo_goodput_policies.py"),
              "--campaign-dir", str(campaign)]
    result = subprocess.run(select, cwd=REPO)
    if result.returncode == 2:
        request = json.loads((campaign / "selection/needs_additional_calibration.json").read_text())
        extra_orders = []
        for repeat in request["additional_repeats"]:
            items = [item_by_name(name) for name in request["policies"]]
            random.Random(CAMPAIGN_SEEDS["calibration_policy_order"] + repeat).shuffle(items)
            extra_orders.append([x[0] for x in items])
            for item in items: ensure_block(pilot, args, item, int(repeat))
        (campaign / "calibration/logs/additional_matrix.json").write_text(json.dumps({
            "reason": request["reason"], "orders": extra_orders}, indent=2) + "\n")
        subprocess.run(select, cwd=REPO, check=True)
    elif result.returncode:
        raise subprocess.CalledProcessError(result.returncode, select)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); add_args(ap); user = ap.parse_args()
    campaign = resolve(user.campaign_dir); assert_environment(); pilot = configure_root(campaign)
    if not (campaign / "slo_grid_frozen.json").exists():
        subprocess.run([sys.executable, str(REPO / "scripts/audit_slo_grid.py"),
                        "--campaign-dir", str(campaign)], cwd=REPO, check=True)
    capture_hardware(campaign, user); append_command(campaign, sys.argv); old_ci_audit(campaign)
    smoke(pilot, user, campaign); overhead(pilot, user, campaign)
    baseline_path = baseline(pilot, user, campaign); calibration(pilot, user, campaign, baseline_path)
    print(json.dumps({"campaign": str(campaign), "calibration_complete": True,
                      "selection": str(campaign / "frozen_slo_selection.json")}, indent=2))


if __name__ == "__main__":
    main()
