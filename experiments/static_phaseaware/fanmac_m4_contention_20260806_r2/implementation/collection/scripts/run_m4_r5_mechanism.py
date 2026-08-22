#!/usr/bin/env python3
"""Freeze and run base-M4 r5 phase-asymmetry characterization."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BLOCK = REPO / "scripts/run_static_phaseaware_pilot.py"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.strip()


def label(policy: str, cap: int) -> str:
    return "llm-only" if policy == "llm-only" else f"fixed{cap}"


def freeze(campaign: Path, stage: str) -> dict[str, Any]:
    path = campaign / "M4_MECHANISM_CLEAN_FREEZE.json"
    if path.exists():
        return json.loads(path.read_text())
    baseline_result = json.loads((campaign / "R5_BASELINE_A_RESULT.json").read_text())
    if not baseline_result.get("passed"):
        raise SystemExit("r5 baseline A did not pass")
    k_hi = json.loads((campaign / "K_HI_FREEZE.json").read_text())
    policies = [{"policy": "llm-only", "cap": 0},
                {"policy": "fixed", "cap": 1},
                {"policy": "fixed", "cap": 2},
                {"policy": "fixed", "cap": 4}]
    repeats = []
    for repeat in range(3):
        order = list(policies)
        random.Random(2026086100 + repeat).shuffle(order)
        repeats.append({"repeat": repeat, "prompt_seed": 3610000019 + repeat * 10007,
                        "query_seed": 3610500022 + repeat * 10007, "order": order})
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
        "repository_commit": git_head(), "observer_mode": "event",
        "baseline": baseline_result, "selected_cpu_only_K_hi": k_hi["selected_K_hi"],
        "workload": {"context_tokens": 2048, "output_tokens": 128,
                     "measured_requests": 100, "valid_repeats": 3,
                     "fresh_process_per_block": True, "max_workers": 4},
        "policies": policies, "repeats": repeats,
        "interpretation": "mechanism characterization only; smoke and invalid attempts excluded",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    normalization = {"stage": "r5_baseline_A", "valid_repeats": 5,
                     "p95_tpot_ms": baseline_result["median_official_p95_tpot_ms"],
                     "p95_ttft_ms": baseline_result["median_official_p95_ttft_ms"],
                     "run_keys": baseline_result["run_keys"]}
    (campaign / "r5_normalization_baseline.json").write_text(
        json.dumps(normalization, indent=2) + "\n")
    return payload


def run_block(campaign: Path, stage: str, spec: dict[str, Any], repeat_spec: dict[str, Any],
              index: Path, model: Path, log: Any) -> None:
    policy, cap = str(spec["policy"]), int(spec["cap"])
    repeat = int(repeat_spec["repeat"]); expected = label(policy, cap)
    raw = campaign / stage / "raw/runs.jsonl"
    existing = [row for row in read_jsonl(raw)
                if row.get("policy") == expected and int(row.get("repeat", -1)) == repeat]
    if any(row.get("status") == "valid" for row in existing):
        return
    attempt = max([int(row.get("attempt", 0)) for row in existing], default=0) + 1
    if attempt > 2:
        raise RuntimeError(f"mechanism hard-invalid retry exhausted: {expected}/r{repeat}")
    command = [
        sys.executable, str(BLOCK), "--run-one", "--stage", stage,
        "--policy", policy, "--observer-mode", "event", "--repeat", str(repeat),
        "--attempt", str(attempt), "--fixed-workers", str(cap),
        "--prefill-cap", str(cap), "--decode-cap", str(cap),
        "--prompt-seed", str(repeat_spec["prompt_seed"]),
        "--query-seed", str(repeat_spec["query_seed"]), "--model", str(model),
        "--index", str(index), "--context", "2048", "--output-tokens", "128",
        "--llm-requests", "100", "--max-workers", "4", "--feeders", "8",
        "--queries-per-task", "4096", "--chunk", "16", "--ef-search", "128",
        "--top-k", "10", "--memory-sample-interval-s", "1", "--warmup-s", "2",
        "--mem-limit-gb", "5.5", "--min-headroom-gb", "3.0",
        "--memory-idle-seconds", "2", "--sentinel-tolerance", "0.03",
        "--sentinel-cooldown", "2", "--sentinel-attempts", "2", "--sentinel-reps", "2",
        "--sentinel-reference-warmup-s", "120", "--within-block-drift-tolerance", "0.10",
        "--within-block-qps-drift-tolerance", "0.20", "--min-duration-s", "5",
        "--min-completed-queries", "1000",
    ]
    if policy != "llm-only":
        command += ["--baseline-file", str(campaign / "r5_normalization_baseline.json")]
    env = {**os.environ, "PHASEGATE_CAMPAIGN_ROOT": str(campaign),
           "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                   check=True)
    rows = [row for row in read_jsonl(raw)
            if row.get("policy") == expected and int(row.get("repeat", -1)) == repeat]
    if not rows or rows[-1].get("status") != "valid":
        if attempt == 1:
            run_block(campaign, stage, spec, repeat_spec, index, model, log)
            return
        raise RuntimeError(f"mechanism block invalid twice: {expected}/r{repeat}")


def export(campaign: Path, stage: str) -> None:
    rows = read_jsonl(campaign / stage / "raw/runs.jsonl")
    if len([row for row in rows if row.get("status") == "valid"]) != 12:
        raise RuntimeError("mechanism stage is incomplete")
    fields = ["policy", "repeat", "attempt", "status", "p95_tpot_ms", "p95_ttft_ms",
              "normalized_p95_tpot", "normalized_p95_ttft", "total_retrieval_goodput_qps",
              "prefill_retrieval_qps", "decode_retrieval_qps",
              "prefill_active_retrieval_worker_mean", "decode_active_retrieval_worker_mean",
              "retrieval_latency_p50_ms", "retrieval_latency_p95_ms", "pageouts_delta",
              "swap_used_delta_bytes", "memory_pressure_clean", "observer_mode",
              "observer_subprocess_count_during_block", "invalid_reason", "run_key"]
    with (campaign / "m4_mechanism_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--stage", default="m4_mechanism_clean")
    args = parser.parse_args(); campaign = args.campaign.resolve()
    protocol = freeze(campaign, args.stage)
    if git_head() != protocol["repository_commit"]:
        raise SystemExit("repository commit differs from M4_MECHANISM_FREEZE.json")
    with (campaign / f"{args.stage}_orchestration.log").open("a") as log:
        for repeat_spec in protocol["repeats"]:
            for spec in repeat_spec["order"]:
                run_block(campaign, args.stage, spec, repeat_spec, args.index.resolve(),
                          args.model.resolve(), log)
    export(campaign, args.stage)


if __name__ == "__main__":
    main()
