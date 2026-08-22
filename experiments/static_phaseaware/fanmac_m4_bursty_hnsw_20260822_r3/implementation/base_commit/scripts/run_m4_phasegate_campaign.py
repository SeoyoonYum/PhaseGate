#!/usr/bin/env python3
"""Resume-safe orchestration for the preregistered Apple M4 campaign."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BLOCK = REPO / "scripts/run_static_phaseaware_pilot.py"
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
MODEL_REVISION = "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"
SLO_GRID = [1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stage_runs(campaign: Path, stage: str) -> list[dict]:
    return read_jsonl(campaign / stage / "raw/runs.jsonl")


def run_one(args: argparse.Namespace, stage: str, policy: str, repeat: int,
            requests: int, prompt_seed: int, query_seed: int, prefill: int = 0,
            decode: int = 0, baseline: Path | None = None, output_tokens: int = 128,
            schedule: Path | None = None, offset: float = 0.0) -> dict:
    label = (f"fixed{decode}" if policy == "fixed" else
             f"phasegate{prefill}to{decode}" if policy == "phasegate" else
             f"timegate{prefill}to{decode}" if policy == "timegate" else policy)
    existing = [row for row in stage_runs(args.campaign, stage)
                if row.get("policy") == label and int(row.get("repeat", -1)) == repeat]
    valid = [row for row in existing if row.get("status") == "valid"]
    if valid:
        return valid[-1]
    start_attempt = max([int(row.get("attempt", 0)) for row in existing], default=0) + 1
    for attempt in range(start_attempt, min(start_attempt + 2, 3)):
        command = [sys.executable, str(BLOCK), "--run-one", "--stage", stage,
                   "--policy", policy, "--repeat", str(repeat), "--attempt", str(attempt),
                   "--fixed-workers", str(decode), "--prefill-cap", str(prefill),
                   "--decode-cap", str(decode), "--prompt-seed", str(prompt_seed),
                   "--query-seed", str(query_seed), "--model", str(args.model),
                   "--context", "2048", "--output-tokens", str(output_tokens),
                   "--llm-requests", str(requests), "--max-workers", "2", "--feeders", "8",
                   "--queries-per-task", "4096", "--chunk", "16", "--ef-search", "128",
                   "--top-k", "10", "--index", str(args.index), "--sample-ms", "5",
                   "--rss-sample-stride", "20", "--warmup-s", "2", "--mem-limit-gb", "5.5",
                   "--min-headroom-gb", "3.0", "--memory-idle-seconds", "2",
                   "--sentinel-tolerance", "0.03", "--sentinel-cooldown", "2",
                   "--sentinel-attempts", "2", "--sentinel-reps", "2",
                   "--within-block-drift-tolerance", "0.10",
                   "--within-block-qps-drift-tolerance", "0.20", "--min-duration-s", "5",
                   "--min-completed-queries", "1000"]
        if baseline is not None:
            command += ["--baseline-file", str(baseline)]
        if schedule is not None:
            command += ["--timegate-schedule", str(schedule),
                        "--timegate-offset-s", str(offset)]
        env = {**os.environ, "PHASEGATE_CAMPAIGN_ROOT": str(args.campaign),
               "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
        log = args.campaign / "orchestration.log"
        with log.open("a") as handle:
            subprocess.run(command, cwd=REPO, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        rows = [row for row in stage_runs(args.campaign, stage)
                if row.get("policy") == label and int(row.get("repeat", -1)) == repeat
                and int(row.get("attempt", -1)) == attempt]
        if rows and rows[-1].get("status") == "valid":
            return rows[-1]
    raise RuntimeError(f"{stage}/{label}/repeat-{repeat} failed twice")


def baseline(args: argparse.Namespace, stage: str, repeats: int, requests: int,
             seed: int, output_tokens: int = 128) -> Path:
    rows = [run_one(args, stage, "llm-only", repeat, requests,
                    seed + repeat * 10_000, seed + 5000 + repeat * 10_000,
                    output_tokens=output_tokens) for repeat in range(repeats)]
    tpot = [float(row["p95_tpot_ms"]) for row in rows]
    ttft = [float(row["p95_ttft_ms"]) for row in rows]
    payload = {"stage": stage, "valid_repeats": repeats,
               "p95_tpot_ms": float(np.median(tpot)),
               "p95_ttft_ms": float(np.median(ttft)),
               "run_keys": [row["run_key"] for row in rows]}
    path = args.campaign / stage / "baseline.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def calibration_selection(args: argparse.Namespace, baseline_path: Path) -> dict:
    candidates = [("fixed", 0, 0), ("fixed", 1, 1), ("fixed", 2, 2),
                  ("phasegate", 2, 0), ("phasegate", 2, 1)]
    freeze = {"candidate_rule": "M4 allowed caps only; K_hi=2 from CPU scaling",
              "candidates": [{"policy": p, "prefill_cap": hi, "decode_cap": lo}
                             for p, hi, lo in candidates],
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "seeds": {"base": 2026085200}}
    (args.campaign / "CALIBRATION_POLICY_FREEZE.json").write_text(
        json.dumps(freeze, indent=2) + "\n")
    orders = []
    for repeat in range(3):
        order = list(candidates); random.Random(2026085200 + repeat).shuffle(order); orders.append(order)
        for policy, high, low in order:
            run_one(args, "calibration", policy, repeat, 100,
                    2026085200 + repeat * 10000, 2026085700 + repeat * 10000,
                    high, low, baseline_path)
    rows = [row for row in stage_runs(args.campaign, "calibration")
            if row.get("policy_arg") != "llm-only" and row.get("status") == "valid"]
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["policy"], []).append(row)
    summaries = {}
    for name, group in groups.items():
        summaries[name] = {"policy": name, "policy_arg": group[0]["policy_arg"],
                           "prefill_cap": group[0]["prefill_cap"], "decode_cap": group[0]["decode_cap"],
                           "median_qps": float(np.median([x["total_retrieval_goodput_qps"] for x in group])),
                           "median_tpot_norm": float(np.median([x["normalized_p95_tpot"] for x in group])),
                           "median_ttft_norm": float(np.median([x["normalized_p95_ttft"] for x in group])),
                           "runs": group}
    selections = {}
    for budget in SLO_GRID:
        feasible = [item for item in summaries.values()
                    if len(item["runs"]) == 3 and all(float(x["normalized_p95_tpot"]) <= budget
                    and float(x["normalized_p95_ttft"]) <= budget for x in item["runs"])]
        for family in ("fixed", "phasegate"):
            options = [item for item in feasible if item["policy_arg"] == family
                       and int(item["decode_cap"]) >= 1]
            options.sort(key=lambda x: (-x["median_qps"], x["median_tpot_norm"],
                                        x["median_ttft_norm"], max(x["prefill_cap"], x["decode_cap"]),
                                        x["policy"]))
            selections.setdefault(str(budget), {})[family] = options[0]["policy"] if options else None
    primary = next((b for b in SLO_GRID if selections[str(b)]["fixed"]
                    and selections[str(b)]["phasegate"]), None)
    if primary is None:
        raise RuntimeError("no primary_B with continuous Fixed and PhaseGate feasibility")
    payload = {"primary_B": primary, "selected_fixed": selections[str(primary)]["fixed"],
               "selected_phasegate": selections[str(primary)]["phasegate"],
               "budget_specific_selections": selections,
               "baseline": json.loads(baseline_path.read_text()), "orders": orders,
               "model_revision": MODEL_REVISION, "repository_commit": subprocess.run(
                   ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                   text=True, check=True).stdout.strip()}
    (args.campaign / "frozen_m4_selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def freeze_timegate(args: argparse.Namespace, selected: str) -> Path:
    timelines = sorted((args.campaign / "calibration/raw/timelines").glob(f"*{selected}*.json"))
    intervals = []
    for path in timelines:
        events = json.loads(path.read_text())["events"]
        starts = {(e.get("request_id"), e["phase"]): e["timestamp"] for e in events if e["event"] == "start"}
        for event in events:
            if event["event"] != "end" or event["phase"] not in ("PREFILL", "DECODE"):
                continue
            start = starts.get((event.get("request_id"), event["phase"]))
            if start is not None and event["timestamp"] > start:
                intervals.append({"cap": 2 if event["phase"] == "PREFILL" else 1,
                                  "duration_s": event["timestamp"] - start,
                                  "source_phase": event["phase"]})
    if not intervals:
        raise RuntimeError("cannot construct TimeGate schedule")
    rng = random.Random(2026086300)
    period = sum(item["duration_s"] for item in intervals)
    offsets = [rng.random() * period for _ in range(7)]
    payload = {"construction": "calibration interval replay with frozen circular offsets",
               "phase_access": False, "high_cap": 2, "low_cap": 1,
               "intervals": intervals, "period_s": period,
               "high_duty_fraction": sum(x["duration_s"] for x in intervals if x["cap"] == 2) / period,
               "transition_rate_hz": len(intervals) / period,
               "heldout_offsets_s": offsets, "offset_seed": 2026086300}
    path = args.campaign / "timegate_schedule_freeze.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--from-stage", choices=("mechanism", "baseline", "calibration", "heldout", "shape"), default="mechanism")
    args = parser.parse_args()
    args.campaign = args.campaign.resolve(); args.index = args.index.resolve(); args.model = args.model.resolve()
    stages = ["mechanism", "baseline", "calibration", "heldout", "shape"]
    begin = stages.index(args.from_stage)
    campaign_baseline = args.campaign / "isolated_baseline/baseline.json"
    if begin <= 0:
        mechanism_base = baseline(args, "mechanism", 3, 100, 2026084100)
        for repeat in range(3):
            for policy, high, low in (("fixed", 1, 1), ("fixed", 2, 2)):
                run_one(args, "mechanism", policy, repeat, 100, 2026084100 + repeat * 10000,
                        2026084600 + repeat * 10000, high, low, mechanism_base)
    if begin <= 1:
        campaign_baseline = baseline(args, "isolated_baseline", 5, 150, 2026085100)
        data = json.loads(campaign_baseline.read_text())
        rows = [row for row in stage_runs(args.campaign, "isolated_baseline") if row["status"] == "valid"]
        for key, baseline_key in (("p95_tpot_ms", "p95_tpot_ms"), ("p95_ttft_ms", "p95_ttft_ms")):
            median = float(data[baseline_key])
            if any(abs(float(row[key]) / median - 1) > .03 for row in rows):
                raise RuntimeError(f"unstable isolated baseline: {key}")
    selection_path = args.campaign / "frozen_m4_selection.json"
    if begin <= 2:
        selection = calibration_selection(args, campaign_baseline)
    else:
        selection = json.loads(selection_path.read_text())
    schedule = freeze_timegate(args, selection["selected_phasegate"])
    schedule_data = json.loads(schedule.read_text())
    if begin <= 3:
        # Semantic smoke does not contribute to inference.
        for index, (policy, hi, lo) in enumerate((("fixed", 1, 1), ("phasegate", 2, 1),
                                                  ("timegate", 2, 1))):
            run_one(args, "timegate_semantic", policy, 0, 3, 2026086200, 2026086250,
                    hi, lo, campaign_baseline, 128,
                    schedule if policy == "timegate" else None,
                    schedule_data["heldout_offsets_s"][0])
        pre = baseline(args, "pre_eval_baseline", 3, 150, 2026086400)
        old, fresh = json.loads(campaign_baseline.read_text()), json.loads(pre.read_text())
        for key in ("p95_tpot_ms", "p95_ttft_ms"):
            if abs(float(fresh[key]) / float(old[key]) - 1) > .03:
                raise RuntimeError(f"pre-evaluation baseline drift >3%: {key}")
        fixed_cap = int(selection["selected_fixed"].replace("fixed", ""))
        held = [("llm-only", 0, 0), ("fixed", fixed_cap, fixed_cap),
                ("phasegate", 2, 1), ("timegate", 2, 1)]
        if fixed_cap != 1:
            held.append(("fixed", 1, 1))
        for repeat in range(7):
            order = list(held); random.Random(2026086500 + repeat).shuffle(order)
            for policy, hi, lo in order:
                run_one(args, "heldout", policy, repeat, 250,
                        2026086500 + repeat * 10000, 2026087000 + repeat * 10000,
                        hi, lo, None if policy == "llm-only" else campaign_baseline, 128,
                        schedule if policy == "timegate" else None,
                        schedule_data["heldout_offsets_s"][repeat])
    if begin <= 4:
        fixed_cap = int(selection["selected_fixed"].replace("fixed", ""))
        for length, seed in ((64, 2026088100), (128, 2026088200), (512, 2026088300)):
            stage = f"output_length_{length}"
            length_base = baseline(args, stage, 3, 100, seed, length)
            for repeat in range(5):
                order = [("fixed", fixed_cap, fixed_cap), ("phasegate", 2, 1)]
                random.Random(seed + repeat).shuffle(order)
                for policy, hi, lo in order:
                    run_one(args, stage, policy, repeat, 100, seed + repeat * 10000,
                            seed + 5000 + repeat * 10000, hi, lo, length_base, length)
    (args.campaign / "COLLECTION_COMPLETE.json").write_text(json.dumps(
        {"completed_utc": datetime.now(timezone.utc).isoformat(),
         "selection": selection, "model_revision": MODEL_REVISION}, indent=2) + "\n")


if __name__ == "__main__":
    main()
