#!/usr/bin/env python3
"""Freeze and run the diagnostic Base-M4 Mini contention campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, TextIO


REPO = Path(__file__).resolve().parents[1]
BLOCK = REPO / "scripts/run_static_phaseaware_pilot.py"
EXPECTED_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
EXPECTED_REVISION = "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"
EXPECTED_INDEX_SHA = "4c65bde676235523dbba2f1dc78a44de3f447470d38105488586d7ca486a51f0"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def command_output(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def thermal_snapshot() -> dict[str, Any]:
    return {"timestamp_utc": now(), "thermal": command_output(["pmset", "-g", "therm"]),
            "power": command_output(["pmset", "-g", "batt"])}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def verify_machine(machine: dict[str, Any]) -> None:
    hardware = machine["hardware"]
    expected = {"model_name": "Mac mini", "model_identifier": "Mac16,10",
                "chip": "Apple M4", "performance_cores": 4,
                "efficiency_cores": 6, "total_cpu_cores": 10,
                "gpu_cores": 10, "unified_memory": "16 GB"}
    mismatches = {key: (hardware.get(key), value) for key, value in expected.items()
                  if hardware.get(key) != value}
    if mismatches:
        raise SystemExit(f"not the expected base-M4 Mac mini: {mismatches}")
    if machine["model"]["identifier"] != EXPECTED_MODEL:
        raise SystemExit("model identifier mismatch")
    if machine["model"]["revision"] != EXPECTED_REVISION:
        raise SystemExit("model revision mismatch")
    if machine["index"]["sha256"] != EXPECTED_INDEX_SHA:
        raise SystemExit("index checksum mismatch")


def prepare(campaign: Path, index: Path, model: Path, r4: Path, machine_source: Path) -> None:
    if campaign.exists() and any(campaign.iterdir()):
        required = [campaign / "CAMPAIGN_MANIFEST.json",
                    campaign / "BASELINE_PROTOCOL_FREEZE.json",
                    campaign / "FROZEN_EXECUTION_MATRIX.json"]
        if not all(path.exists() for path in required):
            raise SystemExit("nonempty campaign directory lacks complete freezes")
        return
    campaign.mkdir(parents=True, exist_ok=True)
    machine = json.loads(machine_source.read_text())
    verify_machine(machine)
    if not index.is_file() or sha256(index) != EXPECTED_INDEX_SHA:
        raise SystemExit("exact HNSW index is missing or has the wrong checksum")
    if not model.is_dir():
        raise SystemExit("exact local model snapshot is missing")
    required_model_files = ("config.json", "tokenizer.json", "model.safetensors")
    if any(not (model / name).is_file() for name in required_model_files):
        raise SystemExit("model snapshot is incomplete")
    if not r4.is_dir():
        raise SystemExit("immutable r4 campaign is missing")
    stop_report = r4.parents[3] / "M4_CAMPAIGN_STOP_REPORT.md"
    if not stop_report.exists():
        stop_report = REPO / "M4_CAMPAIGN_STOP_REPORT.md"
    source_mechanism = r4 / "m4_mechanism_runs.csv"
    head = git_head()
    baseline_sets = {
        "A": [{"repeat": repeat, "prompt_seed": 2026085100 + repeat * 10_000,
               "query_seed": 2026090100 + repeat * 10_000} for repeat in range(5)],
        "B": [{"repeat": repeat, "prompt_seed": 2026089100 + repeat * 10_000,
               "query_seed": 2026094100 + repeat * 10_000} for repeat in range(5)],
    }
    policies = [{"policy": "llm-only", "cap": 0}, {"policy": "fixed", "cap": 1},
                {"policy": "fixed", "cap": 2}, {"policy": "fixed", "cap": 4}]
    repeats = []
    for repeat in range(3):
        order = [dict(item) for item in policies]
        random.Random(2026080601 + repeat).shuffle(order)
        repeats.append({"repeat": repeat, "prompt_seed": 2026084100 + repeat * 10_000,
                        "query_seed": 2026084600 + repeat * 10_000, "order": order})
    common = {"context_tokens": 2048, "output_tokens": 128,
              "observer_mode": "event", "memory_sample_interval_s": 1,
              "warmup_s": 2, "fresh_process_per_block": True,
              "faiss_omp_threads": 1, "max_workers": 4,
              "queries_per_task": 4096, "chunk": 16, "ef_search": 128, "top_k": 10,
              "mlx_memory_limit_gb": 5.5}
    baseline_freeze = {
        "created_utc": now(), "repository_commit": head,
        "source": "r4 isolated-baseline and revalidation request counts/seeds",
        "trace_sets": baseline_sets, "workload": {**common, "measured_requests": 150,
                                                   "valid_repeats": 5},
        "gate": {"definition": "every run's p95 TPOT and TTFT within +/-3% of its set median; zero pageout and swap growth; normal memory pressure",
                 "max_revalidation_sets": 1, "set_B_requires_documented_environmental_correction": True},
    }
    matrix = {
        "created_utc": now(), "repository_commit": head,
        "campaign_scope": "diagnostic contention only; no calibration, PhaseGate, TimeGate, held-out, or output-length sweep",
        "workload": {**common, "measured_requests": 100, "valid_repeats": 3},
        "policies": policies, "repeats": repeats,
        "cap_interpretation": {"cap_1": "low-cap condition", "cap_2": "campaign-frozen K_hi",
                               "cap_4": "diagnostic-only; cannot change K_hi or future policy caps"},
        "frozen_K_hi": 2,
        "normalization": "within-repeat LLM-only for TTFT/TPOT; never compare raw QPS across devices",
    }
    manifest = {
        "created_utc": now(), "campaign_id": campaign.name, "repository_commit": head,
        "device_scope": "fan-cooled base-M4 Mac mini, not M4 Pro",
        "machine": machine["hardware"], "macos": machine["macos"],
        "python": machine["python"], "packages": machine["packages"],
        "model": {"identifier": EXPECTED_MODEL, "revision": EXPECTED_REVISION,
                  "quantization": "4-bit", "local_path": str(model)},
        "index": {"path": str(index), "sha256": EXPECTED_INDEX_SHA,
                  "vectors": 100_000, "dimensions": 384, "M": 32,
                  "efConstruction": 80, "file_size": index.stat().st_size},
        "observer": {"mode": "event", "legacy_5ms_sampler": False,
                     "repeated_ps_subprocesses": False, "memory_sampling_max_hz": 1,
                     "phase_timing_source": "PhaseMonitor/event timestamps",
                     "rss_sampling": "block boundary and <=1 Hz native resource API"},
        "thermal_monitoring": "pmset -g therm and pmset -g batt at block boundaries",
        "source_files": {"r4_campaign": str(r4), "r4_mechanism_csv_sha256": sha256(source_mechanism),
                         "stop_report": str(stop_report), "stop_report_sha256": sha256(stop_report)},
        "initial_thermal_power": thermal_snapshot(),
        "scientific_constraints": {"frozen_K_hi": 2, "cap4_diagnostic_only": True,
                                   "valid_blocks_never_rerun_for_outcome": True,
                                   "hard_invalid_retry_limit": 1},
    }
    (campaign / "BASELINE_PROTOCOL_FREEZE.json").write_text(json.dumps(baseline_freeze, indent=2) + "\n")
    (campaign / "FROZEN_EXECUTION_MATRIX.json").write_text(json.dumps(matrix, indent=2) + "\n")
    (campaign / "CAMPAIGN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def common_command(stage: str, repeat: int, attempt: int, policy: str, cap: int,
                   prompt_seed: int, query_seed: int, requests: int, model: Path,
                   index: Path, baseline: Path | None) -> list[str]:
    command = [sys.executable, str(BLOCK), "--run-one", "--stage", stage,
               "--policy", policy, "--observer-mode", "event", "--repeat", str(repeat),
               "--attempt", str(attempt), "--fixed-workers", str(cap),
               "--prefill-cap", str(cap), "--decode-cap", str(cap),
               "--prompt-seed", str(prompt_seed), "--query-seed", str(query_seed),
               "--model", str(model), "--index", str(index), "--context", "2048",
               "--output-tokens", "128", "--llm-requests", str(requests),
               "--max-workers", "4", "--feeders", "8", "--queries-per-task", "4096",
               "--chunk", "16", "--ef-search", "128", "--top-k", "10",
               "--memory-sample-interval-s", "1", "--warmup-s", "2",
               "--mem-limit-gb", "5.5", "--min-headroom-gb", "3.0",
               "--memory-idle-seconds", "2", "--sentinel-tolerance", "0.03",
               "--sentinel-cooldown", "2", "--sentinel-attempts", "2",
               "--sentinel-reps", "2", "--sentinel-reference-warmup-s", "120",
               "--within-block-drift-tolerance", "0.10",
               "--within-block-qps-drift-tolerance", "0.20", "--min-duration-s", "5",
               "--min-completed-queries", "1000"]
    if baseline is not None and policy != "llm-only":
        command += ["--baseline-file", str(baseline)]
    return command


def run_block(campaign: Path, stage: str, policy: str, cap: int, repeat: int,
              prompt_seed: int, query_seed: int, requests: int, model: Path,
              index: Path, baseline: Path | None, log: TextIO) -> dict[str, Any]:
    label = "llm-only" if policy == "llm-only" else f"fixed{cap}"
    raw = campaign / stage / "raw/runs.jsonl"
    existing = [row for row in read_jsonl(raw) if row.get("policy") == label
                and int(row.get("repeat", -1)) == repeat]
    valid = [row for row in existing if row.get("status") == "valid"]
    if valid:
        return valid[-1]
    attempt = max([int(row.get("attempt", 0)) for row in existing], default=0) + 1
    if attempt > 2:
        raise RuntimeError(f"hard-invalid retry exhausted: {stage}/{label}/r{repeat}")
    command = common_command(stage, repeat, attempt, policy, cap, prompt_seed,
                             query_seed, requests, model, index, baseline)
    audit = {"stage": stage, "policy": label, "repeat": repeat, "attempt": attempt,
             "before": thermal_snapshot(), "started_utc": now()}
    env = {**os.environ, "PHASEGATE_CAMPAIGN_ROOT": str(campaign),
           "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    result = subprocess.run(command, cwd=REPO, env=env, stdout=log,
                            stderr=subprocess.STDOUT)
    audit.update({"ended_utc": now(), "returncode": result.returncode,
                  "after": thermal_snapshot()})
    append_jsonl(campaign / "thermal_power_audit.jsonl", audit)
    if result.returncode:
        if attempt == 1:
            return run_block(campaign, stage, policy, cap, repeat, prompt_seed,
                             query_seed, requests, model, index, baseline, log)
        raise RuntimeError(f"block process failed twice: {stage}/{label}/r{repeat}")
    rows = [row for row in read_jsonl(raw) if row.get("policy") == label
            and int(row.get("repeat", -1)) == repeat]
    if not rows or rows[-1].get("status") != "valid":
        if attempt == 1:
            return run_block(campaign, stage, policy, cap, repeat, prompt_seed,
                             query_seed, requests, model, index, baseline, log)
        raise RuntimeError(f"hard-invalid result twice: {stage}/{label}/r{repeat}")
    return rows[-1]


def export_baseline(campaign: Path, set_name: str) -> bool:
    stage = f"contention_baseline_{set_name}"
    valid = [row for row in read_jsonl(campaign / stage / "raw/runs.jsonl")
             if row.get("status") == "valid"]
    by_repeat = {int(row["repeat"]): row for row in valid}
    if len(by_repeat) != 5:
        raise RuntimeError(f"expected five valid baseline blocks, found {len(by_repeat)}")
    rows = [by_repeat[i] for i in range(5)]
    med_tpot = median(float(row["p95_tpot_ms"]) for row in rows)
    med_ttft = median(float(row["p95_ttft_ms"]) for row in rows)
    output = []
    for row in rows:
        tpot_dev = abs(float(row["p95_tpot_ms"]) / med_tpot - 1)
        ttft_dev = abs(float(row["p95_ttft_ms"]) / med_ttft - 1)
        output.append({"baseline_set": set_name, "repeat": row["repeat"],
            "run_key": row["run_key"], "p95_tpot_ms": row["p95_tpot_ms"],
            "tpot_abs_deviation": tpot_dev, "p95_ttft_ms": row["p95_ttft_ms"],
            "ttft_abs_deviation": ttft_dev, "within_3pct_both": tpot_dev <= .03 and ttft_dev <= .03,
            "pageouts_delta": row["pageouts_delta"], "swap_used_delta_bytes": row["swap_used_delta_bytes"],
            "memory_pressure_clean": row["memory_pressure_clean"], "duration_s": row["duration_s"],
            "observer_mode": row["observer_mode"],
            "observer_subprocess_count": row["observer_subprocess_count_during_block"],
            "token_count_exact": row["observer_reconstruction_audit"]["token_events"] == 150 * 128,
            "event_reconstruction_exact": row["observer_reconstruction_audit"]["query_accounting_exact"]
                                          and row["observer_reconstruction_audit"]["timestamps_monotonic"]})
    write_csv(campaign / f"contention_baseline_{set_name}_runs.csv", output)
    passed = all(row["within_3pct_both"] and int(row["pageouts_delta"]) == 0
                 and int(row["swap_used_delta_bytes"]) == 0 and row["memory_pressure_clean"]
                 and int(row["observer_subprocess_count"]) == 0
                 and row["token_count_exact"] and row["event_reconstruction_exact"] for row in output)
    result = {"created_utc": now(), "baseline_set": set_name, "passed": passed,
              "median_p95_tpot_ms": med_tpot, "median_p95_ttft_ms": med_ttft,
              "run_keys": [row["run_key"] for row in output],
              "gate": "all five runs within +/-3% around set medians; zero pageout/swap growth; normal pressure"}
    (campaign / f"CONTENTION_BASELINE_{set_name}_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    if passed:
        normalization = {"stage": stage, "valid_repeats": 5,
                         "p95_tpot_ms": med_tpot, "p95_ttft_ms": med_ttft,
                         "run_keys": result["run_keys"]}
        (campaign / "contention_normalization_baseline.json").write_text(json.dumps(normalization, indent=2) + "\n")
    return passed


def run_baseline(campaign: Path, set_name: str, model: Path, index: Path) -> None:
    freeze = json.loads((campaign / "BASELINE_PROTOCOL_FREEZE.json").read_text())
    if git_head() != freeze["repository_commit"]:
        raise SystemExit("repository commit differs from baseline freeze")
    if set_name == "B" and not (campaign / "BASELINE_B_ENVIRONMENT_CORRECTION.json").exists():
        raise SystemExit("baseline B requires a documented environmental correction")
    log_path = campaign / "contention_orchestration.log"
    with log_path.open("a") as log:
        for trace in freeze["trace_sets"][set_name]:
            run_block(campaign, f"contention_baseline_{set_name}", "llm-only", 0,
                      trace["repeat"], trace["prompt_seed"], trace["query_seed"],
                      150, model, index, None, log)
    if not export_baseline(campaign, set_name):
        raise SystemExit(f"contention baseline set {set_name} failed the frozen gate")


def passed_baseline(campaign: Path) -> dict[str, Any]:
    for name in ("A", "B"):
        path = campaign / f"CONTENTION_BASELINE_{name}_RESULT.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result.get("passed"):
                return result
    raise SystemExit("no frozen baseline set passed; contention blocks are forbidden")


def run_contention(campaign: Path, model: Path, index: Path) -> None:
    baseline = passed_baseline(campaign)
    matrix = json.loads((campaign / "FROZEN_EXECUTION_MATRIX.json").read_text())
    if git_head() != matrix["repository_commit"]:
        raise SystemExit("repository commit differs from execution-matrix freeze")
    if int(matrix["frozen_K_hi"]) != 2:
        raise SystemExit("K_hi interpretation changed")
    normalizer = campaign / "contention_normalization_baseline.json"
    if not normalizer.exists():
        raise SystemExit("normalization baseline missing")
    with (campaign / "contention_orchestration.log").open("a") as log:
        for repeat in matrix["repeats"]:
            for spec in repeat["order"]:
                run_block(campaign, "m4_mini_contention", spec["policy"], spec["cap"],
                          repeat["repeat"], repeat["prompt_seed"], repeat["query_seed"],
                          100, model, index, normalizer, log)
    raw = read_jsonl(campaign / "m4_mini_contention/raw/runs.jsonl")
    valid = [row for row in raw if row.get("status") == "valid"]
    keys = {(row["policy"], int(row["repeat"])) for row in valid}
    expected = {(policy, repeat) for policy in ("llm-only", "fixed1", "fixed2", "fixed4")
                for repeat in range(3)}
    if keys != expected or len(valid) != 12:
        raise RuntimeError(f"contention matrix incomplete: {len(valid)} valid, keys={keys}")
    fields = ["policy", "repeat", "attempt", "status", "prompt_seed", "query_seed",
              "p95_tpot_ms", "p95_ttft_ms", "total_retrieval_goodput_qps",
              "prefill_retrieval_qps", "decode_retrieval_qps",
              "retrieval_latency_p50_ms", "retrieval_latency_p95_ms",
              "prefill_active_retrieval_worker_mean", "decode_active_retrieval_worker_mean",
              "pageouts_delta", "swap_used_delta_bytes", "memory_pressure_clean",
              "observer_mode", "observer_subprocess_count_during_block", "duration_s",
              "resident_memory_bytes", "peak_resident_memory_bytes", "run_key"]
    with (campaign / "m4_mini_contention_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(sorted(valid, key=lambda row: (int(row["repeat"]), row["policy"])))
    complete = {"created_utc": now(), "repository_commit": git_head(),
                "passed_baseline_set": baseline["baseline_set"], "valid_blocks": 12,
                "invalid_attempts_preserved": len(raw) - len(valid), "frozen_K_hi": 2,
                "cap4_interpretation": "diagnostic-only"}
    (campaign / "M4_MINI_CONTENTION_COLLECTION_COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--r4", required=True, type=Path)
    parser.add_argument("--machine-source", required=True, type=Path)
    parser.add_argument("--stage", choices=("prepare", "baseline-A", "baseline-B", "contention"),
                        required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(); index = args.index.resolve(); model = args.model.resolve()
    r4 = args.r4.resolve(); machine_source = args.machine_source.resolve()
    if args.stage == "prepare":
        prepare(campaign, index, model, r4, machine_source)
    else:
        if not campaign.exists():
            raise SystemExit("campaign must be prepared first")
        if args.stage.startswith("baseline-"):
            run_baseline(campaign, args.stage[-1], model, index)
        else:
            run_contention(campaign, model, index)


if __name__ == "__main__":
    main()
