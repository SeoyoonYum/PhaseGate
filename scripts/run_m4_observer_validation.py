#!/usr/bin/env python3
"""Run the frozen base-M4 observer intervention in paired randomized order."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RUN_ONE = REPO / "scripts/run_static_phaseaware_pilot.py"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.strip()


def run_condition(root: Path, freeze: dict[str, Any], stage: str, mode: str,
                  repeat: int, prompt_seed: int, query_seed: int, log: Any) -> None:
    raw = root / stage / "raw/runs.jsonl"
    existing = [row for row in read_jsonl(raw)
                if row.get("observer_mode") == mode and int(row.get("repeat", -1)) == repeat]
    if any(row.get("status") == "valid" for row in existing):
        return
    attempt = max([int(row.get("attempt", 0)) for row in existing], default=0) + 1
    if attempt > 2:
        raise RuntimeError(f"hard-invalid retry exhausted: {stage}/{mode}/r{repeat}")
    workload = freeze["workload"]
    artifact = freeze["artifacts"]
    command = [
        sys.executable, str(RUN_ONE), "--run-one", "--stage", stage,
        "--policy", "llm-only", "--observer-mode", mode,
        "--repeat", str(repeat), "--attempt", str(attempt),
        "--prompt-seed", str(prompt_seed), "--query-seed", str(query_seed),
        "--model", artifact["model_local_path"], "--index", artifact["index_path"],
        "--context", str(workload["context_tokens"]),
        "--output-tokens", str(workload["output_tokens"]),
        "--llm-requests", str(workload["measured_requests"]),
        "--max-workers", "4", "--feeders", "8", "--queries-per-task", "4096",
        "--chunk", "16", "--ef-search", "128", "--top-k", "10",
        "--sample-ms", "5", "--rss-sample-stride", "20",
        "--memory-sample-interval-s", "1", "--warmup-s", "2",
        "--mem-limit-gb", "5.5", "--min-headroom-gb", "3.0",
        "--memory-idle-seconds", "2", "--sentinel-tolerance", "0.03",
        "--sentinel-cooldown", "2", "--sentinel-attempts", "2",
        "--sentinel-reps", "2", "--sentinel-reference-warmup-s", "120",
        "--within-block-drift-tolerance", "0.10",
        "--within-block-qps-drift-tolerance", "0.20",
        "--min-duration-s", "5", "--min-completed-queries", "1000",
    ]
    env = os.environ.copy()
    env.update({"PHASEGATE_CAMPAIGN_ROOT": str(root), "OMP_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1"})
    subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                   check=True)
    rows = [row for row in read_jsonl(raw)
            if row.get("observer_mode") == mode and int(row.get("repeat", -1)) == repeat]
    if not rows or rows[-1].get("status") != "valid":
        if attempt == 1:
            run_condition(root, freeze, stage, mode, repeat, prompt_seed, query_seed, log)
        else:
            raise RuntimeError(f"observer block invalid twice: {stage}/{mode}/r{repeat}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("freeze", type=Path)
    parser.add_argument("--stage", choices=("minimal_event", "legacy_event", "all"),
                        default="all")
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = json.loads(freeze_path.read_text())
    if git_head() != freeze["repository_commit"]:
        raise SystemExit("repository commit differs from OBSERVER_VALIDATION_FREEZE.json")
    root = freeze_path.parent
    stages = []
    if args.stage in ("minimal_event", "all"):
        stages.append(("observer_minimal_event", freeze["minimal_event_pairs"]))
    if args.stage in ("legacy_event", "all"):
        stages.append(("observer_legacy_event", freeze["legacy_event_pairs"]))
    with (root / "observer_validation_orchestration.log").open("a") as log:
        for stage, pairs in stages:
            for pair in pairs:
                for mode in pair["order"]:
                    run_condition(root, freeze, stage, mode, int(pair["repeat"]),
                                  int(pair["prompt_seed"]), int(pair["query_seed"]), log)


if __name__ == "__main__":
    main()
