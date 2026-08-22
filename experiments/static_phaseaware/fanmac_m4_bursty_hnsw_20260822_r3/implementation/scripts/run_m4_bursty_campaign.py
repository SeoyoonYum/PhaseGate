#!/usr/bin/env python3
"""Freeze and resume the base-M4 bursty baseline and primary collection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BLOCK = REPO / "scripts/run_static_phaseaware_pilot.py"
R5 = REPO / "experiments/static_phaseaware/fanmac_m4_causal_shape_20260805_r5"
MODEL = Path("/Users/m1/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-4bit/snapshots/8b403126fc14f14cfc99bb4cfa72ecbc129ea677")
INDEX = REPO / "experiments/phaseguard/index/hnsw_100k_d384.faiss"
MODEL_REVISION = "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"
INDEX_SHA256 = "4c65bde676235523dbba2f1dc78a44de3f447470d38105488586d7ca486a51f0"
STARTING_COMMIT = "d5868e14e7e7dccc04dd5215dc76619891398da3"
sys.path.insert(0, str(REPO / "src"))

from phaseguard.demand_gate import FrozenDemandTrace, generate_demand_trace  # noqa: E402

POLICIES = (
    {"name": "fixed1", "family": "fixed", "high": 1, "low": 1},
    {"name": "phasegate4to1", "family": "phasegate", "high": 4, "low": 1},
    {"name": "timegate4to1", "family": "timegate", "high": 4, "low": 1},
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.strip()


def git_clean() -> bool:
    return not subprocess.run(["git", "status", "--short", "--untracked-files=no"],
                              cwd=REPO, check=True,
                              capture_output=True, text=True).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")


def coverage_offset(trace: FrozenDemandTrace, duration_s: float) -> float:
    candidates = np.arange(0.0, trace.duration_s, 0.25)
    return float(min(candidates, key=lambda offset: (
        abs(trace.scheduled_on_fraction(float(offset), duration_s) - trace.demand_level),
        float(offset))))


def current_machine_manifest() -> dict[str, Any]:
    r5_manifest = json.loads((R5 / "machine_manifest.json").read_text())
    sw_vers = subprocess.run(["sw_vers"], check=True, capture_output=True,
                             text=True).stdout
    hardware = subprocess.run(["system_profiler", "SPHardwareDataType",
                               "SPDisplaysDataType"], check=True, capture_output=True,
                              text=True).stdout
    power = subprocess.run(["pmset", "-g", "custom"], check=True,
                           capture_output=True, text=True).stdout
    power_source = subprocess.run(["pmset", "-g", "batt"], check=True,
                                  capture_output=True, text=True).stdout
    packages = {name: importlib.metadata.version(name) for name in
                ("mlx", "mlx-lm", "numpy", "faiss-cpu", "matplotlib")}
    model_files = {}
    for name, expected in r5_manifest["model"]["files"].items():
        path = MODEL / name
        model_files[name] = {"size": path.stat().st_size, "sha256": sha256(path),
                             "matches_r5": path.stat().st_size == int(expected["size"])
                             and sha256(path) == expected["sha256"]}
    manifest = {
        "created_utc": utc_now(), "hardware_reference": r5_manifest["hardware"],
        "hardware_raw": hardware, "macos_raw": sw_vers, "power_raw": power,
        "power_source_raw": power_source,
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": packages, "repository_commit": git_head(),
        "repository_branch": subprocess.run(
            ["git", "branch", "--show-current"], cwd=REPO, check=True,
            capture_output=True, text=True).stdout.strip(),
        "starting_r5_commit": STARTING_COMMIT,
        "model": {"identifier": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                  "revision": MODEL_REVISION, "path": str(MODEL), "files": model_files},
        "index": {"path": str(INDEX), "file_size": INDEX.stat().st_size,
                  "sha256": sha256(INDEX), "vectors": 100000, "dimensions": 384,
                  "graph_degree": 32, "ef_construction": 80, "ef_search": 128,
                  "top_k": 10},
        "observer": {"mode": "event", "memory_sampling_max_hz": 1,
                     "production_subprocess_polling": False},
    }
    return manifest


def assert_artifacts_and_environment(manifest: dict[str, Any]) -> None:
    expected_packages = {"mlx": "0.31.2", "mlx-lm": "0.31.3", "numpy": "2.4.6",
                         "faiss-cpu": "1.14.3", "matplotlib": "3.11.0"}
    if manifest["python"] != "3.13.14" or manifest["packages"] != expected_packages:
        raise RuntimeError("Python/package identity differs from frozen r5 environment")
    if not all(item["matches_r5"] for item in manifest["model"]["files"].values()):
        raise RuntimeError("model file identity differs from r5 manifest")
    if manifest["index"]["file_size"] != 180820834 or manifest["index"]["sha256"] != INDEX_SHA256:
        raise RuntimeError("index identity differs from r5")
    hardware = manifest["hardware_raw"]
    required = ("Model Name: Mac mini", "Model Identifier: Mac16,10",
                "Chip: Apple M4", "Total Number of Cores: 10 (4 Performance and 6 Efficiency)",
                "Memory: 16 GB")
    if not all(value in hardware for value in required):
        raise RuntimeError("machine is not the frozen base-M4 Mac mini")
    if hardware.count("Total Number of Cores: 10") < 2:
        raise RuntimeError("expected both 10-core CPU and 10-core GPU")
    if "lowpowermode         0" not in manifest["power_raw"]:
        raise RuntimeError("Low Power Mode is not disabled")
    if "AC Power" not in manifest["power_source_raw"]:
        raise RuntimeError("machine is not drawing AC power")


def validate_baseline_stability_tolerance(value: float) -> float:
    tolerance = float(value)
    if not 0.0 < tolerance < 1.0:
        raise ValueError("baseline stability tolerance must be between 0 and 1")
    return tolerance


def evaluate_baseline_stability(
    rows: list[dict[str, Any]], tolerance: float
) -> tuple[float, float, list[dict[str, Any]], bool]:
    tolerance = validate_baseline_stability_tolerance(tolerance)
    if len(rows) != 5:
        raise ValueError(f"baseline stability requires exactly five rows, found {len(rows)}")
    med_tpot = median(float(row["p95_tpot_ms"]) for row in rows)
    med_ttft = median(float(row["p95_ttft_ms"]) for row in rows)
    evaluated = []
    def within_tolerance(deviation: float) -> bool:
        return deviation < tolerance or math.isclose(
            deviation, tolerance, rel_tol=0.0, abs_tol=1e-12
        )

    for row in rows:
        tpot_dev = abs(float(row["p95_tpot_ms"]) / med_tpot - 1)
        ttft_dev = abs(float(row["p95_ttft_ms"]) / med_ttft - 1)
        evaluated.append({
            **row,
            "tpot_abs_deviation": tpot_dev,
            "ttft_abs_deviation": ttft_dev,
            "baseline_stability_tolerance": tolerance,
            "within_frozen_tolerance_both": (
                within_tolerance(tpot_dev) and within_tolerance(ttft_dev)
            ),
        })
    return med_tpot, med_ttft, evaluated, all(
        row["within_frozen_tolerance_both"] for row in evaluated
    )


def prepare(
    campaign: Path,
    smoke_root: Path,
    compatibility_root: Path,
    baseline_stability_tolerance: float,
    protocol_seed_offset: int,
    baseline_stability_rationale: str | None,
) -> dict[str, Any]:
    baseline_stability_tolerance = validate_baseline_stability_tolerance(
        baseline_stability_tolerance
    )
    if protocol_seed_offset < 0:
        raise ValueError("protocol seed offset must be nonnegative")
    rationale = (baseline_stability_rationale or "").strip()
    if not math.isclose(
        baseline_stability_tolerance, .03, rel_tol=0.0, abs_tol=1e-12
    ) and not rationale:
        raise ValueError(
            "a non-default baseline stability tolerance requires a frozen rationale"
        )
    if not rationale:
        rationale = "Original handoff default of +/-3%."
    freeze_path = campaign / "BURSTY_LOAD_PROTOCOL_FREEZE.json"
    if freeze_path.exists():
        existing = json.loads(freeze_path.read_text())
        existing_tolerance = float(
            existing["validity"]["baseline_stability_absolute_relative_tolerance"]
        )
        if not math.isclose(
            existing_tolerance,
            baseline_stability_tolerance,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or int(existing.get("protocol_seed_offset", 0)) != protocol_seed_offset:
            raise RuntimeError("requested prepare settings differ from immutable freeze")
        return existing
    if not git_clean():
        raise RuntimeError("final protocol freeze requires a clean committed worktree")
    manifest = current_machine_manifest(); assert_artifacts_and_environment(manifest)
    mechanics = json.loads((smoke_root / "MECHANICS_SMOKE_RESULT.json").read_text())
    compatibility = json.loads((compatibility_root / "COMPATIBILITY_RESULT.json").read_text())
    if not mechanics["passed"] or not compatibility["passed"]:
        raise RuntimeError("smoke/compatibility gate has not passed")
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign / "machine_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for source in (
        R5 / "timegate_schedule_freeze.json",
        smoke_root / "SMOKE_PROTOCOL_FREEZE.json",
        smoke_root / "MECHANICS_SMOKE_RESULT.json",
        smoke_root / "mechanics_smoke_audit.csv",
        compatibility_root / "COMPATIBILITY_PROTOCOL_FREEZE.json",
        compatibility_root / "COMPATIBILITY_RESULT.json",
        compatibility_root / "compatibility_pairwise.csv",
    ):
        shutil.copy2(source, campaign / source.name)
    measured_20_s = float(mechanics["median_measured_block_duration_s"])
    expected_300_s = measured_20_s * 15.0
    trace_dir = campaign / "frozen_traces"; trace_dir.mkdir()
    trace_specs: dict[str, Any] = {}
    for duty_index, (key, level) in enumerate((("5", .05), ("25", .25), ("100", 1.0))):
        trace_specs[key] = []
        for repeat in range(5):
            primary_seed = 4300000000 + protocol_seed_offset + duty_index * 10000 + repeat
            backup_seed = 4301000000 + protocol_seed_offset + duty_index * 10000 + repeat
            pair = {}
            for role, seed in (("primary", primary_seed), ("backup", backup_seed)):
                trace = generate_demand_trace(demand_level=level, seed=seed)
                path = trace_dir / f"duty{key}_r{repeat}_{role}_seed{seed}.json"
                trace.write(path)
                pair[role] = {"seed": seed, "path": str(path.resolve()),
                              "offset_s": (0.0 if level == 1.0 else
                                           coverage_offset(trace, expected_300_s)),
                              "duration_s": trace.duration_s,
                              "sha256": sha256(path)}
            trace_specs[key].append({"repeat": repeat, **pair})
    timegate = json.loads((campaign / "timegate_schedule_freeze.json").read_text())
    period = float(timegate["period_s"])
    repeats = []
    for repeat in range(5):
        duties = ["5", "25", "100"]
        random.Random(2026082300 + protocol_seed_offset + repeat).shuffle(duties)
        cells = []
        for duty_order, duty in enumerate(duties):
            policies = [dict(item) for item in POLICIES]
            random.Random(
                2026082400 + protocol_seed_offset + repeat * 10 + duty_order
            ).shuffle(policies)
            cells.append({"duty": duty, "policy_order": policies,
                "prompt_seed": 4310000011 + protocol_seed_offset
                + repeat * 100_000 + int(duty) * 100,
                "query_seed": 4310500014 + protocol_seed_offset
                + repeat * 100_000 + int(duty) * 100,
                "timegate_offset_s": random.Random(
                    2026082500 + protocol_seed_offset + repeat * 10 + duty_order
                ).random() * period})
        repeats.append({"repeat": repeat, "duty_order": duties, "cells": cells})
    baseline_sets = {}
    for set_index, set_name in enumerate(("A", "B")):
        baseline_sets[set_name] = [
            {"repeat": repeat,
             "prompt_seed": 4320000011 + protocol_seed_offset
             + set_index * 1_000_000 + repeat * 10007,
             "query_seed": 4320500014 + protocol_seed_offset
             + set_index * 1_000_000 + repeat * 10007}
            for repeat in range(5)]
    midpoint = [
        {"repeat": repeat,
         "prompt_seed": 4330000011 + protocol_seed_offset + repeat * 10007,
         "query_seed": 4330500014 + protocol_seed_offset + repeat * 10007}
        for repeat in range(3)
    ]
    freeze = {
        "schema_version": 2, "created_utc": utc_now(), "repository_commit": git_head(),
        "starting_r5_commit": STARTING_COMMIT,
        "protocol_seed_offset": protocol_seed_offset,
        "baseline_stability_gate": {
            "absolute_relative_tolerance": baseline_stability_tolerance,
            "selection_rationale": rationale,
            "applies_to": "every run-level p95 TPOT and p95 TTFT versus its five-run median",
        },
        "machine_manifest": "machine_manifest.json",
        "model": {"path": str(MODEL), "revision": MODEL_REVISION},
        "index": {"path": str(INDEX), "size": 180820834, "sha256": INDEX_SHA256},
        "observer_mode": "event", "production_subprocess_polling": False,
        "memory_sampling_interval_s": 1.0,
        "latency_definitions": {"token_timestamp_position": "after model() and mx.eval(), unchanged from r5",
            "p95_tpot": "p95 across request-level p95 token gaps",
            "p95_ttft": "p95 across request TTFT"},
        "B": 1.25, "context_tokens": 2048, "output_tokens": 128,
        "llm_requests_per_block": 300, "faiss_internal_threads": 1,
        "maximum_application_workers": 4, "mlx_memory_limit_gb": 5.5,
        "policies": [
            {**POLICIES[0], "exact_semantics": "cap=1 in PREFILL and DECODE"},
            {**POLICIES[1], "exact_semantics": "cap=4 in PREFILL; cap=1 in DECODE"},
            {**POLICIES[2], "exact_semantics": "phase-blind r5 wall-clock schedule using caps 4 and 1"},
        ],
        "demand_gate_exact_semantics": (
            "a chunk may start iff demand is ON AND worker index is below the current "
            "policy cap AND active chunks are below that cap; OFF cannot raise the cap"
        ),
        "timegate_schedule": "timegate_schedule_freeze.json",
        "demand_levels": [0.05, 0.25, 1.0],
        "demand_generator": {"minimum_interval_s": .25,
            "on": {"shift_s": .25, "gamma_shape": 2, "gamma_scale": .875},
            "off_5pct": {"shift_s": .25, "gamma_shape": 2, "gamma_scale": 18.875},
            "off_25pct": {"shift_s": .25, "gamma_shape": 2, "gamma_scale": 2.875},
            "continuous_100pct": True, "minimum_trace_duration_s": 7200},
        "runtime_estimate_basis": {"measured_20_request_median_s": measured_20_s,
            "linear_300_request_estimate_s": expected_300_s,
            "policy_blocks_estimate_hours": expected_300_s * 45 / 3600},
        "traces": trace_specs, "baseline_sets": baseline_sets,
        "midpoint_recheck": midpoint, "repeats": repeats,
        "paired_repeat_seeds": [
            2026082300 + protocol_seed_offset + repeat for repeat in range(5)
        ],
        "execution": {"fresh_process_per_block": True,
            "baseline_valid_repeats": 5, "primary_paired_repeats_per_duty": 5,
            "total_primary_policy_blocks": 45,
            "midpoint_after_completed_repeat_index": 1, "block_timeout_s": 7200},
        "validity": {"scheduled_duty_tolerance_absolute": .03,
            "matched_triplet_duty_max_difference_absolute": .02,
            "individual_hard_invalid_retry_limit": 1,
            "triplet_trace_retry_limit": 1,
            "baseline_stability_absolute_relative_tolerance": baseline_stability_tolerance,
            "midpoint_drift_absolute_relative_tolerance": .03,
            "pageout_with_zero_swap_normal_pressure": "soft flag",
            "slo_failure": "valid performance result",
            "hard_invalid_conditions": [
                "crash, hang, or process failure",
                "wrong machine or changed software/model/index identity",
                "swap growth", "warning or critical memory pressure",
                "thermal or power warning",
                "corrupt, missing, duplicated, or truncated request/token/event data",
                "wrong output-token count", "non-monotonic timestamps or event sequence",
                "event reconstruction mismatch",
                "incorrect Fixed, PhaseGate, TimeGate, or DemandGate semantics",
                "TimeGate consulting phase",
                "DemandGate consulting phase or policy identity",
                "new HNSW chunk beginning during OFF",
                "production observer subprocess launch",
                "realized duty outside the frozen tolerance",
                "matched-triplet duty imbalance outside the frozen tolerance",
                "severe within-block drift under the unchanged r5 rule",
            ]},
        "stop_rules": {"baseline_A_failure": "diagnose and permit exactly one documented environmental correction before frozen B",
            "baseline_B_failure": "stop campaign", "midpoint_drift_failure": "stop before remaining repeats",
            "implementation_change_after_freeze": "new campaign directory"},
        "analysis": {"unit": "run-level matched comparison", "paired_bootstrap_resamples": 10000,
            "request_bootstrap_for_primary_ci": False,
            "metrics": ["median retrieval QPS over full block wall time",
                "median retrieval QPS over scheduled-ON time",
                "completed retrieval queries per LLM request", "normalized p95 TPOT",
                "normalized p95 TTFT", "joint SLO pass count",
                "scheduled and actual retrieval-active fractions",
                "prefill and decode active-worker time", "cap-binding fraction during ON",
                "ON-to-OFF drain time", "pageout, swap, pressure, power, thermal proxy and soft flags",
                "PhaseGate/Fixed QPS ratio", "PhaseGate/TimeGate QPS ratio",
                "absolute PhaseGate-minus-Fixed QPS", "SLO pass-count contrast"]},
    }
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    return freeze


def block_command(*, stage: str, policy: dict[str, Any], repeat: int, attempt: int,
                  requests: int, prompt_seed: int, query_seed: int,
                  baseline: Path | None, trace: Path | None = None,
                  demand_offset_s: float = 0.0, timegate_offset_s: float = 0.0,
                  campaign: Path) -> list[str]:
    result = [sys.executable, str(BLOCK), "--run-one", "--stage", stage,
        "--policy", str(policy["family"]), "--observer-mode", "event",
        "--repeat", str(repeat), "--attempt", str(attempt),
        "--fixed-workers", str(policy["low"]), "--prefill-cap", str(policy["high"]),
        "--decode-cap", str(policy["low"]), "--prompt-seed", str(prompt_seed),
        "--query-seed", str(query_seed), "--model", str(MODEL), "--index", str(INDEX),
        "--context", "2048", "--output-tokens", "128", "--llm-requests", str(requests),
        "--max-workers", "4", "--feeders", "8", "--queries-per-task", "4096",
        "--chunk", "16", "--ef-search", "128", "--top-k", "10",
        "--memory-sample-interval-s", "1", "--warmup-s", "2", "--mem-limit-gb", "5.5",
        "--min-headroom-gb", "3", "--memory-idle-seconds", "2",
        "--sentinel-tolerance", ".03", "--sentinel-cooldown", "2",
        "--sentinel-attempts", "2", "--sentinel-reps", "2",
        "--sentinel-reference-warmup-s", "120", "--within-block-drift-tolerance", ".10",
        "--within-block-qps-drift-tolerance", ".20", "--min-duration-s", "5",
        "--min-completed-queries", "1000", "--slo-multiplier", "1.25"]
    if baseline is not None:
        result += ["--baseline-file", str(baseline)]
    if policy["family"] == "timegate":
        result += ["--timegate-schedule", str(campaign / "timegate_schedule_freeze.json"),
                   "--timegate-offset-s", str(timegate_offset_s)]
    if trace is not None:
        result += ["--demand-trace", str(trace), "--demand-offset-s", str(demand_offset_s)]
    return result


def run_subprocess(command: list[str], campaign: Path, log: Any) -> None:
    environment = {**os.environ, "PHASEGATE_CAMPAIGN_ROOT": str(campaign),
                   "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    subprocess.run(command, cwd=REPO, env=environment, stdout=log,
                   stderr=subprocess.STDOUT, check=True, timeout=7200)


def run_block_with_retry(*, campaign: Path, stage: str, policy: dict[str, Any],
                         repeat: int, requests: int, prompt_seed: int, query_seed: int,
                         baseline: Path | None, log: Any, trace: Path | None = None,
                         demand_offset_s: float = 0.0,
                         timegate_offset_s: float = 0.0) -> dict[str, Any]:
    raw = campaign / stage / "raw/runs.jsonl"
    trace_seed = FrozenDemandTrace.load(trace).seed if trace is not None else None
    def matches_trace(row: dict[str, Any]) -> bool:
        if trace_seed is None:
            return True
        audit = row.get("bursty_demand_audit")
        return ((isinstance(audit, dict) and int(audit.get("trace_seed", -1)) == trace_seed)
                or str(row.get("run_key", "")).endswith(f"_s{trace_seed}"))
    existing = [row for row in read_jsonl(raw) if row.get("policy") == policy["name"]
                and int(row.get("repeat", -1)) == repeat
                and matches_trace(row)]
    failures_path = campaign / stage / "raw/orchestration_failures.jsonl"
    failures = [row for row in read_jsonl(failures_path)
                if row.get("policy") == policy["name"]
                and int(row.get("repeat", -1)) == repeat
                and (trace_seed is None or int(row.get("trace_seed", -1)) == trace_seed)]
    valid = [row for row in existing if row.get("status") == "valid"]
    if valid:
        return valid[-1]
    if existing and "demand_duty_clean" in str(existing[-1].get("invalid_reason", "")):
        return existing[-1]
    attempt = max([int(row.get("attempt", 0)) for row in [*existing, *failures]], default=0) + 1
    if attempt > 2:
        raise RuntimeError(f"individual hard-invalid retry exhausted: {stage}/{policy['name']}/r{repeat}")
    command = block_command(stage=stage, policy=policy, repeat=repeat,
        attempt=attempt, requests=requests, prompt_seed=prompt_seed, query_seed=query_seed,
        baseline=baseline, trace=trace, demand_offset_s=demand_offset_s,
        timegate_offset_s=timegate_offset_s, campaign=campaign)
    try:
        run_subprocess(command, campaign, log)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        append_jsonl(failures_path, {"created_utc": utc_now(), "stage": stage,
            "policy": policy["name"], "repeat": repeat, "attempt": attempt,
            "trace_seed": trace_seed, "failure": type(exc).__name__,
            "returncode": getattr(exc, "returncode", None), "command": command})
        if attempt >= 2:
            raise RuntimeError(
                f"process-failure retry exhausted: {stage}/{policy['name']}/r{repeat}"
            ) from exc
        time.sleep(2.0)
        return run_block_with_retry(campaign=campaign, stage=stage, policy=policy,
            repeat=repeat, requests=requests, prompt_seed=prompt_seed,
            query_seed=query_seed, baseline=baseline, log=log, trace=trace,
            demand_offset_s=demand_offset_s, timegate_offset_s=timegate_offset_s)
    rows = [row for row in read_jsonl(raw) if row.get("policy") == policy["name"]
            and int(row.get("repeat", -1)) == repeat
            and matches_trace(row)]
    if not rows or max(int(row.get("attempt", 0)) for row in rows) < attempt:
        append_jsonl(failures_path, {"created_utc": utc_now(), "stage": stage,
            "policy": policy["name"], "repeat": repeat, "attempt": attempt,
            "trace_seed": trace_seed, "failure": "missing_result_row",
            "returncode": 0, "command": command})
        if attempt >= 2:
            raise RuntimeError(
                f"missing-result retry exhausted: {stage}/{policy['name']}/r{repeat}"
            )
        return run_block_with_retry(campaign=campaign, stage=stage, policy=policy,
            repeat=repeat, requests=requests, prompt_seed=prompt_seed,
            query_seed=query_seed, baseline=baseline, log=log, trace=trace,
            demand_offset_s=demand_offset_s, timegate_offset_s=timegate_offset_s)
    if rows and rows[-1].get("status") == "valid":
        return rows[-1]
    if rows and "demand_duty_clean" in str(rows[-1].get("invalid_reason", "")):
        # A duration-dependent duty mismatch is retried only as a complete
        # three-policy triplet with the already-frozen backup trace.
        return rows[-1]
    return run_block_with_retry(campaign=campaign, stage=stage, policy=policy,
        repeat=repeat, requests=requests, prompt_seed=prompt_seed, query_seed=query_seed,
        baseline=baseline, log=log, trace=trace, demand_offset_s=demand_offset_s,
        timegate_offset_s=timegate_offset_s)


def export_baseline(campaign: Path, freeze: dict[str, Any], set_name: str) -> bool:
    stage = f"baseline_{set_name}"
    rows = [row for row in read_jsonl(campaign / stage / "raw/runs.jsonl")
            if row.get("status") == "valid"]
    by_repeat = {int(row["repeat"]): row for row in rows}
    selected = [by_repeat[index] for index in range(5)]
    tolerance = validate_baseline_stability_tolerance(
        float(freeze["validity"]["baseline_stability_absolute_relative_tolerance"])
    )
    med_tpot, med_ttft, evaluated, passed = evaluate_baseline_stability(
        selected, tolerance
    )
    output = []
    for row in evaluated:
        output.append({"baseline_set": set_name, "repeat": row["repeat"],
            "run_key": row["run_key"], "p95_tpot_ms": row["p95_tpot_ms"],
            "p95_ttft_ms": row["p95_ttft_ms"],
            "tpot_abs_deviation": row["tpot_abs_deviation"],
            "ttft_abs_deviation": row["ttft_abs_deviation"],
            "baseline_stability_tolerance": tolerance,
            "within_frozen_tolerance_both": row["within_frozen_tolerance_both"],
            "duration_s": row["duration_s"], "pageouts_delta": row["pageouts_delta"],
            "swap_used_delta_bytes": row["swap_used_delta_bytes"],
            "memory_pressure_clean": row["memory_pressure_clean"],
            "observer_subprocess_count": row["observer_subprocess_count_during_block"]})
    write_csv(campaign / "bursty_load_baseline_runs.csv", output)
    request_objects = {item["run_key"]: item["requests"] for item in
                       read_jsonl(campaign / stage / "raw/requests.jsonl")}
    request_rows = []
    for row in selected:
        for index, item in enumerate(request_objects[row["run_key"]]):
            request_rows.append({"baseline_set": set_name, "repeat": row["repeat"],
                "run_key": row["run_key"], "request_index": index,
                "request_id": item["request_id"], "ttft_ms": item["ttft_ms"],
                "mean_tpot_ms": item["mean_tpot_ms"], "p95_tpot_ms": item["p95_tpot_ms"],
                "token_timestamp_count": len(item["token_timestamps"])})
    write_csv(campaign / "bursty_load_baseline_request_metrics.csv", request_rows)
    result = {"set": set_name, "passed": passed, "median_p95_tpot_ms": med_tpot,
              "median_p95_ttft_ms": med_ttft,
              "baseline_stability_tolerance": tolerance,
              "run_keys": [row["run_key"] for row in selected]}
    (campaign / f"BASELINE_{set_name}_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    if passed:
        (campaign / "bursty_load_normalization_baseline.json").write_text(json.dumps({
            "stage": f"baseline_{set_name}", "valid_repeats": 5,
            "p95_tpot_ms": med_tpot, "p95_ttft_ms": med_ttft,
            "B": freeze["B"],
            "baseline_stability_tolerance": tolerance,
            "run_keys": result["run_keys"]}, indent=2) + "\n")
    return passed


def baseline(campaign: Path, freeze: dict[str, Any], set_name: str) -> None:
    if git_head() != freeze["repository_commit"] or not git_clean():
        raise RuntimeError("repository differs from protocol freeze")
    assert_artifacts_and_environment(current_machine_manifest())
    if set_name == "B" and not (campaign / "BASELINE_B_ENVIRONMENTAL_CORRECTION.json").exists():
        raise RuntimeError("baseline B requires a documented environmental correction")
    policy = {"name": "llm-only", "family": "llm-only", "high": 0, "low": 0}
    with (campaign / f"baseline_{set_name}_orchestration.log").open("a") as log:
        for spec in freeze["baseline_sets"][set_name]:
            run_block_with_retry(campaign=campaign, stage=f"baseline_{set_name}",
                policy=policy, repeat=int(spec["repeat"]), requests=300,
                prompt_seed=int(spec["prompt_seed"]), query_seed=int(spec["query_seed"]),
                baseline=None, log=log)
    if not export_baseline(campaign, freeze, set_name):
        tolerance = float(
            freeze["validity"]["baseline_stability_absolute_relative_tolerance"]
        )
        raise RuntimeError(
            f"baseline set {set_name} failed frozen +/-{tolerance:.1%} stability gate"
        )


def triplet_selection_path(campaign: Path) -> Path:
    return campaign / "PRIMARY_TRIPLET_SELECTION.json"


def read_triplet_selection(campaign: Path) -> dict[str, Any]:
    path = triplet_selection_path(campaign)
    return json.loads(path.read_text()) if path.exists() else {"selected": []}


def write_triplet_selection(campaign: Path, payload: dict[str, Any]) -> None:
    path = triplet_selection_path(campaign)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def run_triplet(campaign: Path, freeze: dict[str, Any], repeat_spec: dict[str, Any],
                cell: dict[str, Any], log: Any) -> list[dict[str, Any]]:
    repeat = int(repeat_spec["repeat"]); duty = str(cell["duty"])
    selected = read_triplet_selection(campaign)
    existing_selection = next((item for item in selected["selected"]
                               if int(item["repeat"]) == repeat and item["duty"] == duty), None)
    if existing_selection:
        keys = set(existing_selection["run_keys"])
        return [row for row in read_jsonl(campaign / "primary/raw/runs.jsonl")
                if row.get("run_key") in keys]
    trace_pair = freeze["traces"][duty][repeat]
    baseline_path = campaign / "bursty_load_normalization_baseline.json"
    for trace_role in ("primary", "backup"):
        trace_spec = trace_pair[trace_role]
        rows = []
        for policy in cell["policy_order"]:
            rows.append(run_block_with_retry(campaign=campaign, stage="primary",
                policy=policy, repeat=repeat, requests=300,
                prompt_seed=int(cell["prompt_seed"]), query_seed=int(cell["query_seed"]),
                baseline=baseline_path, log=log, trace=Path(trace_spec["path"]),
                demand_offset_s=float(trace_spec["offset_s"]),
                timegate_offset_s=float(cell["timegate_offset_s"])))
        realized = [float(row["bursty_demand_audit"]["scheduled_on_fraction"]) for row in rows]
        target = float(rows[0]["bursty_demand_audit"]["demand_level"])
        duty_clean = (all(abs(value - target) <= (.03 if target in (.05, .25) else 1e-9)
                          for value in realized)
                      and max(realized) - min(realized) <= .02)
        if duty_clean:
            selected["selected"].append({"repeat": repeat, "duty": duty,
                "trace_role": trace_role, "trace_seed": trace_spec["seed"],
                "run_keys": [row["run_key"] for row in rows],
                "scheduled_on_fractions": realized,
                "max_difference": max(realized) - min(realized)})
            write_triplet_selection(campaign, selected)
            return rows
        if trace_role == "backup":
            raise RuntimeError(f"matched triplet duty retry exhausted: repeat={repeat}, duty={duty}")
    raise AssertionError("unreachable")


def midpoint(campaign: Path, freeze: dict[str, Any], log: Any) -> None:
    result_path = campaign / "MIDPOINT_RECHECK_RESULT.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if not result["passed"]:
            raise RuntimeError("midpoint drift previously failed")
        return
    policy = {"name": "llm-only", "family": "llm-only", "high": 0, "low": 0}
    rows = [run_block_with_retry(campaign=campaign, stage="midpoint", policy=policy,
        repeat=int(spec["repeat"]), requests=300, prompt_seed=int(spec["prompt_seed"]),
        query_seed=int(spec["query_seed"]), baseline=None, log=log)
        for spec in freeze["midpoint_recheck"]]
    baseline_data = json.loads((campaign / "bursty_load_normalization_baseline.json").read_text())
    tpot = median(float(row["p95_tpot_ms"]) for row in rows)
    ttft = median(float(row["p95_ttft_ms"]) for row in rows)
    result = {"median_p95_tpot_ms": tpot, "median_p95_ttft_ms": ttft,
              "tpot_relative_drift": tpot / baseline_data["p95_tpot_ms"] - 1,
              "ttft_relative_drift": ttft / baseline_data["p95_ttft_ms"] - 1}
    result["passed"] = abs(result["tpot_relative_drift"]) <= .03 and abs(
        result["ttft_relative_drift"]) <= .03
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise RuntimeError("midpoint drift exceeds 3%; stop before remaining repeats")


def primary(campaign: Path, freeze: dict[str, Any]) -> None:
    if git_head() != freeze["repository_commit"] or not git_clean():
        raise RuntimeError("repository differs from protocol freeze")
    assert_artifacts_and_environment(current_machine_manifest())
    if not (campaign / "bursty_load_normalization_baseline.json").exists():
        raise RuntimeError("passing fresh baseline is required before primary")
    with (campaign / "primary_orchestration.log").open("a") as log:
        for repeat_spec in freeze["repeats"]:
            repeat = int(repeat_spec["repeat"])
            for cell in repeat_spec["cells"]:
                run_triplet(campaign, freeze, repeat_spec, cell, log)
            if repeat == 1:
                midpoint(campaign, freeze, log)
    selection = read_triplet_selection(campaign)
    if len(selection["selected"]) != 15:
        raise RuntimeError(f"expected 15 selected triplets, found {len(selection['selected'])}")
    (campaign / "PRIMARY_COLLECTION_COMPLETE.json").write_text(json.dumps({
        "completed_utc": utc_now(), "repository_commit": git_head(),
        "selected_triplets": len(selection["selected"]), "selected_blocks": 45}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--compatibility-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "baseline-A", "baseline-B", "primary"),
                        required=True)
    parser.add_argument("--baseline-stability-tolerance", type=float, default=.03)
    parser.add_argument("--protocol-seed-offset", type=int, default=0)
    parser.add_argument("--baseline-stability-rationale")
    args = parser.parse_args(); campaign = args.campaign.resolve()
    if args.stage == "prepare":
        prepare(
            campaign,
            args.smoke_root.resolve(),
            args.compatibility_root.resolve(),
            args.baseline_stability_tolerance,
            args.protocol_seed_offset,
            args.baseline_stability_rationale,
        )
        return
    freeze = json.loads((campaign / "BURSTY_LOAD_PROTOCOL_FREEZE.json").read_text())
    if args.stage == "baseline-A": baseline(campaign, freeze, "A")
    elif args.stage == "baseline-B": baseline(campaign, freeze, "B")
    elif args.stage == "primary": primary(campaign, freeze)


if __name__ == "__main__":
    main()
