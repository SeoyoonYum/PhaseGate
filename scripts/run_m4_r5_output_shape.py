#!/usr/bin/env python3
"""Freeze and collect the base-M4 output-length shape sweep."""
from __future__ import annotations

import argparse
import json
import os
import random
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


def create_freeze(campaign: Path) -> dict[str, Any]:
    path = campaign / "OUTPUT_SHAPE_PROTOCOL_FREEZE.json"
    if path.exists(): return json.loads(path.read_text())
    selection = json.loads((campaign / "frozen_m4_selection.json").read_text())
    if selection["selected_fixed"] != "fixed1" or selection["selected_phasegate"] != "phasegate4to1":
        raise SystemExit("unexpected frozen r5 policies")
    lengths = []
    for index, output_tokens in enumerate((64, 128, 512)):
        base = 3900000017 + index * 10_000_000
        baseline = [{"repeat": repeat, "prompt_seed": base + repeat * 10007,
                     "query_seed": base + 500003 + repeat * 10007} for repeat in range(3)]
        pairs = []
        policies = [{"name": "fixed1", "family": "fixed", "high": 1, "low": 1},
                    {"name": "phasegate4to1", "family": "phasegate", "high": 4, "low": 1}]
        for repeat in range(5):
            order = list(policies); random.Random(base + 700000 + repeat).shuffle(order)
            pairs.append({"repeat": repeat, "prompt_seed": base + 1000000 + repeat * 10007,
                          "query_seed": base + 1500003 + repeat * 10007, "order": order})
        lengths.append({"output_tokens": output_tokens, "baseline": baseline, "pairs": pairs})
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_head(), "context_tokens": 2048, "observer_mode": "event",
        "fixed_policy": "fixed1", "phasegate_policy": "phasegate4to1",
        "baseline_repeats": 3, "paired_repeats": 5, "measured_requests": 100,
        "fresh_process_per_block": True, "reuse_128_heldout": False,
        "lengths": lengths, "interpretation": "policies frozen at 128-token calibration; no per-length recalibration"}
    path.write_text(json.dumps(payload, indent=2) + "\n"); return payload


def run_block(campaign: Path, stage: str, name: str, family: str, high: int, low: int,
              repeat: int, prompt_seed: int, query_seed: int, requests: int,
              output_tokens: int, index: Path, model: Path, log: Any,
              baseline: Path | None = None) -> dict[str, Any]:
    raw = campaign / stage / "raw/runs.jsonl"
    existing = [row for row in read_jsonl(raw)
                if row.get("policy") == name and int(row.get("repeat", -1)) == repeat]
    valid = [row for row in existing if row.get("status") == "valid"]
    if valid: return valid[-1]
    attempt = max([int(row.get("attempt", 0)) for row in existing], default=0) + 1
    if attempt > 2: raise RuntimeError(f"output-shape retry exhausted: {stage}/{name}/r{repeat}")
    command = [sys.executable, str(BLOCK), "--run-one", "--stage", stage,
        "--policy", family, "--observer-mode", "event", "--repeat", str(repeat),
        "--attempt", str(attempt), "--fixed-workers", str(low), "--prefill-cap", str(high),
        "--decode-cap", str(low), "--prompt-seed", str(prompt_seed), "--query-seed", str(query_seed),
        "--model", str(model), "--index", str(index), "--context", "2048",
        "--output-tokens", str(output_tokens), "--llm-requests", str(requests),
        "--max-workers", "4", "--feeders", "8", "--queries-per-task", "4096",
        "--chunk", "16", "--ef-search", "128", "--top-k", "10",
        "--memory-sample-interval-s", "1", "--warmup-s", "2", "--mem-limit-gb", "5.5",
        "--min-headroom-gb", "3.0", "--memory-idle-seconds", "2",
        "--sentinel-tolerance", "0.03", "--sentinel-cooldown", "2",
        "--sentinel-attempts", "2", "--sentinel-reps", "2",
        "--sentinel-reference-warmup-s", "120", "--within-block-drift-tolerance", "0.10",
        "--within-block-qps-drift-tolerance", "0.20", "--min-duration-s", "5",
        "--min-completed-queries", "1000"]
    if baseline is not None: command += ["--baseline-file", str(baseline)]
    env = {**os.environ, "PHASEGATE_CAMPAIGN_ROOT": str(campaign),
           "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = [row for row in read_jsonl(raw)
            if row.get("policy") == name and int(row.get("repeat", -1)) == repeat]
    if not rows or rows[-1].get("status") != "valid":
        if attempt == 1: return run_block(campaign, stage, name, family, high, low, repeat,
            prompt_seed, query_seed, requests, output_tokens, index, model, log, baseline)
        raise RuntimeError(f"output-shape invalid twice: {stage}/{name}/r{repeat}")
    return rows[-1]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True); parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args(); campaign = args.campaign.resolve(); protocol = create_freeze(campaign)
    if git_head() != protocol["repository_commit"]: raise SystemExit("commit differs from output-shape freeze")
    with (campaign / "m4_output_shape_orchestration.log").open("a") as log:
        for length in protocol["lengths"]:
            output_tokens = int(length["output_tokens"]); baseline_stage = f"m4_output_{output_tokens}_baseline"
            baseline_rows = [run_block(campaign, baseline_stage, "llm-only", "llm-only", 0, 0,
                int(item["repeat"]), int(item["prompt_seed"]), int(item["query_seed"]), 100,
                output_tokens, args.index.resolve(), args.model.resolve(), log) for item in length["baseline"]]
            baseline_payload = {"stage": baseline_stage, "valid_repeats": 3,
                "p95_tpot_ms": median(float(row["p95_tpot_ms"]) for row in baseline_rows),
                "p95_ttft_ms": median(float(row["p95_ttft_ms"]) for row in baseline_rows),
                "run_keys": [row["run_key"] for row in baseline_rows]}
            baseline_path = campaign / f"m4_output_{output_tokens}_baseline.json"
            baseline_path.write_text(json.dumps(baseline_payload, indent=2) + "\n")
            stage = f"m4_output_{output_tokens}"
            for pair in length["pairs"]:
                for spec in pair["order"]:
                    run_block(campaign, stage, spec["name"], spec["family"], int(spec["high"]),
                        int(spec["low"]), int(pair["repeat"]), int(pair["prompt_seed"]),
                        int(pair["query_seed"]), 100, output_tokens, args.index.resolve(),
                        args.model.resolve(), log, baseline_path)
    (campaign / "M4_OUTPUT_SHAPE_COLLECTION_COMPLETE.json").write_text(json.dumps(
        {"completed_utc": datetime.now(timezone.utc).isoformat(), "repository_commit": git_head(),
         "lengths": [64, 128, 512], "valid_pairs_per_length": 5}, indent=2) + "\n")


if __name__ == "__main__": main()
