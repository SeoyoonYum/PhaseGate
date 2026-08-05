#!/usr/bin/env python3
"""Freeze, run, and select base-M4 r5 calibration policies."""
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
from statistics import median
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BLOCK = REPO / "scripts/run_static_phaseaware_pilot.py"
SLO_GRID = [1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.strip()


def policy_name(spec: dict[str, Any]) -> str:
    if spec["family"] == "fixed": return f"fixed{spec['decode_cap']}"
    return f"phasegate{spec['prefill_cap']}to{spec['decode_cap']}"


def create_freeze(campaign: Path) -> dict[str, Any]:
    path = campaign / "CALIBRATION_POLICY_FREEZE.json"
    if path.exists(): return json.loads(path.read_text())
    mechanism = read_jsonl(campaign / "m4_mechanism_clean/raw/runs.jsonl")
    cap4 = [row for row in mechanism if row.get("policy") == "fixed4"]
    cap4_safe = len(cap4) == 3 and all(row.get("status") == "valid"
        and int(row.get("swap_used_delta_bytes", 0)) == 0
        and row.get("memory_pressure_clean") for row in cap4)
    fixed_caps = [0, 1, 2] + ([4] if cap4_safe else [])
    candidates = [{"family": "fixed", "prefill_cap": cap, "decode_cap": cap}
                  for cap in fixed_caps]
    candidates += [{"family": "phasegate", "prefill_cap": high, "decode_cap": low}
                   for high, low in ((2, 0), (2, 1), (4, 0), (4, 1), (4, 2))
                   if high != 4 or cap4_safe]
    repeats = []
    for repeat in range(3):
        order = list(candidates); random.Random(2026087100 + repeat).shuffle(order)
        repeats.append({"repeat": repeat, "prompt_seed": 3710000011 + repeat * 10007,
                        "query_seed": 3710500014 + repeat * 10007, "order": order})
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "repository_commit": git_head(), "observer_mode": "event",
               "cap4_valid_and_memory_safe": cap4_safe, "candidates": candidates,
               "slo_grid": SLO_GRID,
               "workload": {"context_tokens": 2048, "output_tokens": 128,
                            "measured_requests": 100, "valid_repeats": 3,
                            "max_workers": 4, "fresh_process_per_block": True},
               "repeats": repeats,
               "selection_rule": "all 3 repeats jointly feasible; maximize median QPS, then lower normalized TPOT, TTFT, max cap, lexicographic name"}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def run_candidate(campaign: Path, spec: dict[str, Any], repeat_spec: dict[str, Any],
                  index: Path, model: Path, log: Any) -> None:
    name = policy_name(spec); repeat = int(repeat_spec["repeat"])
    raw = campaign / "m4_calibration/raw/runs.jsonl"
    existing = [row for row in read_jsonl(raw)
                if row.get("policy") == name and int(row.get("repeat", -1)) == repeat]
    if any(row.get("status") == "valid" for row in existing): return
    attempt = max([int(row.get("attempt", 0)) for row in existing], default=0) + 1
    if attempt > 2: raise RuntimeError(f"calibration retry exhausted: {name}/r{repeat}")
    family = str(spec["family"]); high = int(spec["prefill_cap"]); low = int(spec["decode_cap"])
    command = [sys.executable, str(BLOCK), "--run-one", "--stage", "m4_calibration",
        "--policy", family, "--observer-mode", "event", "--repeat", str(repeat),
        "--attempt", str(attempt), "--fixed-workers", str(low), "--prefill-cap", str(high),
        "--decode-cap", str(low), "--prompt-seed", str(repeat_spec["prompt_seed"]),
        "--query-seed", str(repeat_spec["query_seed"]), "--model", str(model),
        "--index", str(index), "--context", "2048", "--output-tokens", "128",
        "--llm-requests", "100", "--max-workers", "4", "--feeders", "8",
        "--queries-per-task", "4096", "--chunk", "16", "--ef-search", "128",
        "--top-k", "10", "--memory-sample-interval-s", "1", "--warmup-s", "2",
        "--mem-limit-gb", "5.5", "--min-headroom-gb", "3.0", "--memory-idle-seconds", "2",
        "--sentinel-tolerance", "0.03", "--sentinel-cooldown", "2",
        "--sentinel-attempts", "2", "--sentinel-reps", "2",
        "--sentinel-reference-warmup-s", "120", "--within-block-drift-tolerance", "0.10",
        "--within-block-qps-drift-tolerance", "0.20", "--min-duration-s", "5",
        "--min-completed-queries", "1000", "--baseline-file",
        str(campaign / "r5_normalization_baseline.json")]
    env = {**os.environ, "PHASEGATE_CAMPAIGN_ROOT": str(campaign),
           "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = [row for row in read_jsonl(raw)
            if row.get("policy") == name and int(row.get("repeat", -1)) == repeat]
    if not rows or rows[-1].get("status") != "valid":
        if attempt == 1:
            run_candidate(campaign, spec, repeat_spec, index, model, log); return
        raise RuntimeError(f"calibration invalid twice: {name}/r{repeat}")


def select(campaign: Path, protocol: dict[str, Any]) -> None:
    rows = [row for row in read_jsonl(campaign / "m4_calibration/raw/runs.jsonl")
            if row.get("status") == "valid"]
    expected = len(protocol["candidates"]) * 3
    if len(rows) != expected: raise RuntimeError(f"calibration incomplete: {len(rows)}/{expected}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows: groups.setdefault(str(row["policy"]), []).append(row)
    summary = []
    for name, group in sorted(groups.items()):
        summary.append({"policy": name, "family": group[0]["policy_arg"],
            "prefill_cap": group[0]["prefill_cap"], "decode_cap": group[0]["decode_cap"],
            "valid_repeats": len(group),
            "median_retrieval_qps": median(float(row["total_retrieval_goodput_qps"]) for row in group),
            "median_normalized_p95_tpot": median(float(row["normalized_p95_tpot"]) for row in group),
            "median_normalized_p95_ttft": median(float(row["normalized_p95_ttft"]) for row in group),
            "max_normalized_p95_tpot": max(float(row["normalized_p95_tpot"]) for row in group),
            "max_normalized_p95_ttft": max(float(row["normalized_p95_ttft"]) for row in group)})
    with (campaign / "m4_calibration_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summary)
    with (campaign / "m4_calibration_runs.csv").open("w", newline="") as handle:
        fields = list(rows[0]); writer = csv.DictWriter(handle, fieldnames=fields,
            extrasaction="ignore", lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    budget_selections = {}
    for budget in SLO_GRID:
        feasible = [item for item in summary if item["valid_repeats"] == 3
                    and item["max_normalized_p95_tpot"] <= budget
                    and item["max_normalized_p95_ttft"] <= budget]
        chosen = {}
        for family in ("fixed", "phasegate"):
            options = [item for item in feasible if item["family"] == family
                       and int(item["decode_cap"]) >= 1]
            options.sort(key=lambda item: (-item["median_retrieval_qps"],
                item["median_normalized_p95_tpot"], item["median_normalized_p95_ttft"],
                max(int(item["prefill_cap"]), int(item["decode_cap"])), item["policy"]))
            chosen[family] = options[0]["policy"] if options else None
        budget_selections[str(budget)] = chosen
    primary = next((budget for budget in SLO_GRID
                    if all(budget_selections[str(budget)].values())), None)
    if primary is None: raise RuntimeError("no primary budget with both continuous families")
    baseline = json.loads((campaign / "r5_normalization_baseline.json").read_text())
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_head(), "primary_B": primary,
        "selected_fixed": budget_selections[str(primary)]["fixed"],
        "selected_phasegate": budget_selections[str(primary)]["phasegate"],
        "budget_specific_selections": budget_selections, "baseline": baseline,
        "calibration_freeze": "CALIBRATION_POLICY_FREEZE.json"}
    (campaign / "frozen_m4_selection.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True); parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args(); campaign = args.campaign.resolve(); protocol = create_freeze(campaign)
    if git_head() != protocol["repository_commit"]: raise SystemExit("commit differs from calibration freeze")
    with (campaign / "m4_calibration_orchestration.log").open("a") as log:
        for repeat_spec in protocol["repeats"]:
            for spec in repeat_spec["order"]:
                run_candidate(campaign, spec, repeat_spec, args.index.resolve(), args.model.resolve(), log)
    select(campaign, protocol)


if __name__ == "__main__": main()
