#!/usr/bin/env python3
"""Shared orchestration helpers for the fan-cooled static PhaseGate campaign."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "experiments/phaseguard/index/hnsw_100k_d384.faiss"

CALIBRATION_POLICIES = (
    ("fixed0", "fixed0", 0, 0),
    ("fixed1", "fixed", 1, 1),
    ("fixed2", "fixed", 2, 2),
    ("fixed3", "fixed", 3, 3),
    ("fixed4", "fixed", 4, 4),
    ("phasegate1to0", "phasegate", 1, 0),
    ("phasegate2to0", "phasegate", 2, 0),
    ("phasegate3to0", "phasegate", 3, 0),
    ("phasegate4to0", "phasegate", 4, 0),
    ("phasegate2to1", "phasegate", 2, 1),
    ("phasegate3to1", "phasegate", 3, 1),
    ("phasegate4to1", "phasegate", 4, 1),
    ("phasegate3to2", "phasegate", 3, 2),
    ("phasegate4to2", "phasegate", 4, 2),
    ("phasegate4to3", "phasegate", 4, 3),
)

SMOKE_POLICIES = tuple(
    item for item in CALIBRATION_POLICIES
    if item[0] in {"fixed0", "fixed1", "fixed2", "fixed4",
                   "phasegate4to0", "phasegate4to1", "phasegate4to2"}
)


def campaign_default() -> Path:
    raw = subprocess.run(
        ["system_profiler", "SPHardwareDataType"], capture_output=True, text=True,
        timeout=30, check=True).stdout
    match = re.search(r"Chip:\s*(.+)", raw)
    chip = re.sub(r"[^a-z0-9]+", "_", (match.group(1) if match else "unknown").lower()).strip("_")
    return REPO / "experiments/static_phaseaware" / f"fanmac_main_{chip}_{datetime.now():%Y%m%d}"


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--campaign-dir", type=Path)
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--llm-requests", type=int, default=4)
    ap.add_argument("--mem-limit-gb", type=float, default=6.0)
    ap.add_argument("--min-headroom-gb", type=float, default=6.5)
    ap.add_argument("--memory-idle-seconds", type=float, default=30.0)
    ap.add_argument("--sentinel-cooldown", type=float, default=30.0)
    ap.add_argument("--max-attempts", type=int, default=12)


def resolve_campaign(value: Path | None) -> Path:
    path = campaign_default() if value is None else value
    if not path.is_absolute():
        path = REPO / path
    for name in ("hardware", "smoke", "isolated_baseline", "characterization",
                 "calibration", "evaluation", "raw", "processed", "figures", "logs"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path.resolve()


def configure_root(campaign: Path) -> Any:
    os.environ["STATIC_PHASEAWARE_ROOT"] = str(campaign)
    sys.path.insert(0, str(REPO / "scripts"))
    import run_static_phaseaware_pilot as pilot
    if pilot.ROOT != campaign:
        raise RuntimeError(f"campaign root mismatch: {pilot.ROOT} != {campaign}")
    return pilot


def assert_environment() -> None:
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True,
                           timeout=10, check=True).stdout
    if "AC Power" not in power:
        raise RuntimeError("campaign requires AC power")
    custom = subprocess.run(["pmset", "-g", "custom"], capture_output=True, text=True,
                            timeout=10, check=True).stdout
    if re.search(r"lowpowermode\s+1", custom):
        raise RuntimeError("Low Power Mode must be disabled")
    browser = subprocess.run(
        ["pgrep", "-ifl", r"Google Chrome.app|Chromium.app|Arc.app|Firefox.app|Safari.app/Contents/MacOS/Safari"],
        capture_output=True, text=True, timeout=10)
    if browser.stdout.strip():
        raise RuntimeError("browser processes must be closed:\n" + browser.stdout.strip())
    if not INDEX.exists():
        raise RuntimeError(f"missing HNSW index: {INDEX}")


def command_output(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)
    return (result.stdout + result.stderr).strip()


def capture_hardware(campaign: Path, model: str, context: int, output_tokens: int) -> None:
    hardware_raw = command_output(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"])
    software_raw = command_output(["system_profiler", "SPSoftwareDataType"])
    meta = json.loads(INDEX.with_suffix(INDEX.suffix + ".json").read_text())
    def field(pattern: str, default: str = "unknown") -> str:
        found = re.search(pattern, hardware_raw)
        return found.group(1).strip() if found else default
    cpu_match = re.search(r"Total Number of Cores:\s*(\d+).*?\((\d+) Performance and (\d+) Efficiency\)",
                          hardware_raw, re.S)
    gpu_match = re.search(r"Type:\s*GPU.*?Total Number of Cores:\s*(\d+)", hardware_raw, re.S)
    payload = {
        "captured": datetime.now().isoformat(),
        "model_name": field(r"Model Name:\s*(.+)"),
        "model_identifier": field(r"Model Identifier:\s*(.+)"),
        "chip": field(r"Chip:\s*(.+)"),
        "cpu_cores_total": int(cpu_match.group(1)) if cpu_match else None,
        "performance_cores": int(cpu_match.group(2)) if cpu_match else None,
        "efficiency_cores": int(cpu_match.group(3)) if cpu_match else None,
        "gpu_cores": int(gpu_match.group(1)) if gpu_match else None,
        "unified_memory_bytes": int(command_output(["sysctl", "-n", "hw.memsize"])),
        "macos": command_output(["sw_vers"]),
        "python": sys.version.split()[0],
        "git_commit": command_output(["git", "rev-parse", "HEAD"], REPO),
        "model": model, "quantization": "4bit", "context_length": context,
        "output_length": output_tokens, "hnsw": meta,
        "available_disk": command_output(["df", "-h", str(campaign)]).splitlines()[-1],
        "initial_vm_stat": command_output(["vm_stat"]),
        "initial_swap": command_output(["sysctl", "vm.swapusage"]),
        "initial_memory_pressure": command_output(["memory_pressure", "-Q"]),
        "initial_process_rss": command_output(["ps", "-o", "rss=", "-p", str(os.getpid())]),
    }
    versions = command_output([sys.executable, "-c",
        "import mlx,mlx.core as mx,faiss,numpy,matplotlib;"
        "print(mlx.__version__);print(mx.default_device());print(faiss.__version__);"
        "print(numpy.__version__);print(matplotlib.__version__)"])
    payload["runtime_versions"] = versions.splitlines()
    (campaign / "hardware" / "hardware.json").write_text(json.dumps(payload, indent=2) + "\n")
    (campaign / "hardware" / "environment.txt").write_text(
        hardware_raw + "\n\n" + software_raw + "\n\n" + versions + "\n")
    # Required top-level aliases.
    (campaign / "hardware.json").write_text(json.dumps(payload, indent=2) + "\n")
    (campaign / "environment.txt").write_text(
        hardware_raw + "\n\n" + software_raw + "\n\n" + versions + "\n")


def base_namespace(pilot: Any, user: argparse.Namespace, stage: str,
                   seed: int, baseline: Path | None = None) -> argparse.Namespace:
    args = pilot.parser().parse_args([])
    args.stage = stage
    args.seed = seed
    args.model = user.model
    args.context = user.context
    args.output_tokens = user.output_tokens
    args.llm_requests = user.llm_requests
    args.mem_limit_gb = user.mem_limit_gb
    args.min_headroom_gb = user.min_headroom_gb
    args.memory_idle_seconds = user.memory_idle_seconds
    args.sentinel_cooldown = user.sentinel_cooldown
    args.max_attempts = user.max_attempts
    args.index = INDEX
    args.baseline_file = baseline
    return args


def rows_for(pilot: Any, stage: str, smoke: bool = False) -> list[dict[str, Any]]:
    return pilot.read_jsonl(pilot.raw_path(stage, smoke))


HARD_VALIDITY_KEYS = {"request_count_exact", "token_count_exact", "token_timestamps_valid",
                      "active_worker_log_present", "policy_cap_applied", "decode_admission_zero"}


def hard_failure_flags(row: dict[str, Any]) -> list[str]:
    """Interpret legacy strict rows under the resumed hard-failure-only protocol."""
    explicit = row.get("hard_failure_flags")
    if explicit:
        return [str(flag) for flag in explicit]
    if "p95_tpot_ms" not in row:
        return ["incomplete_or_corrupt_block"]
    validity = row.get("validity", {})
    return [key for key in HARD_VALIDITY_KEYS if validity.get(key) is False]


def analysis_eligible(row: dict[str, Any]) -> bool:
    return "p95_tpot_ms" in row and not hard_failure_flags(row)


def clean_run(row: dict[str, Any]) -> bool:
    if not analysis_eligible(row):
        return False
    if "clean_run" in row:
        return bool(row["clean_run"])
    validity = row.get("validity", {})
    return row.get("status") == "valid" and all(bool(value) for value in validity.values())


def ensure_block(pilot: Any, args: argparse.Namespace, item: tuple[str, str, int, int],
                 repeat: int, smoke: bool = False) -> dict[str, Any]:
    label, policy, prefill, decode = item
    args.smoke = smoke
    while True:
        stage_rows = rows_for(pilot, args.stage, smoke)
        rows = [row for row in stage_rows
                if row.get("policy") == label and int(row.get("repeat", -1)) == repeat]
        latest = max(rows, key=lambda row: int(row.get("attempt", -1)), default=None)
        if latest is not None and hard_failure_flags(latest):
            raise RuntimeError(f"hard failure in {label} repeat {repeat}: "
                               f"{','.join(hard_failure_flags(latest))}")
        eligible = [row for row in rows if analysis_eligible(row)]
        clean = [row for row in eligible if clean_run(row)]
        if clean:
            return clean[-1]
        # One soft retry is useful.  Earlier strict-mode attempts may already
        # have consumed it; retain and accept their latest complete result.
        if len(eligible) >= 2:
            return eligible[-1]
        if eligible:
            recover_after_pageout(pilot, args.stage, label, eligible[-1])
        attempt = pilot.max_attempt(args.stage, smoke, label, repeat) + 1
        if attempt > args.max_attempts:
            raise RuntimeError(f"could not obtain valid {stage_label(args.stage)} {label} repeat {repeat}")
        command = pilot.block_command(args, policy, prefill, decode, repeat, attempt)
        subprocess.run(command, cwd=REPO, check=True, env=os.environ.copy())


def _pageouts() -> int:
    raw = command_output(["vm_stat"])
    match = re.search(r"(?m)^Pageouts:\s*(\d+)", raw)
    return int(match.group(1)) if match else -1


def _swap_used_bytes() -> int | None:
    raw = command_output(["sysctl", "vm.swapusage"])
    match = re.search(r"used\s*=\s*([0-9.]+)([BKMG])", raw, re.I)
    if not match:
        return None
    scale = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(match.group(1)) * scale[match.group(2).upper()])


def _stale_experiment_pids() -> list[tuple[int, str]]:
    raw = command_output(["ps", "-axo", "pid=,command="])
    stale: list[tuple[int, str]] = []
    for line in raw.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        pid, command = int(fields[0]), fields[1]
        if pid == os.getpid():
            continue
        if "run_static_phaseaware_pilot.py" in command or "phaseguard-faiss-index-owner" in command:
            stale.append((pid, command))
    return stale


def recover_after_pageout(pilot: Any, stage: str, label: str, failed: dict[str, Any]) -> None:
    """Clean up before the one permitted soft-flag retry; always preserve diagnostics."""
    log_dir = pilot.ROOT / stage / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    before_pageouts, before_swap = _pageouts(), _swap_used_bytes()
    stale_before = _stale_experiment_pids()
    terminated: list[int] = []
    for pid, _ in stale_before:
        try:
            os.kill(pid, signal.SIGTERM)
            terminated.append(pid)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while _stale_experiment_pids() and time.monotonic() < deadline:
        time.sleep(.2)
    time.sleep(30)
    after_pageouts, after_swap = _pageouts(), _swap_used_bytes()
    stale_after = _stale_experiment_pids()
    pressure = command_output(["memory_pressure", "-Q"])
    recovery = {
        "failed_run_key": failed.get("run_key"), "policy": label,
        "failed_pageouts": pageout_delta(failed), "terminated_pids": terminated,
        "stale_before": stale_before, "stale_after": stale_after,
        "idle_seconds": 30, "pageouts_before": before_pageouts,
        "pageouts_after": after_pageouts,
        "pageout_delta_during_recovery": after_pageouts - before_pageouts,
        "swap_before_bytes": before_swap, "swap_after_bytes": after_swap,
        "memory_pressure": pressure,
        "process_snapshot": command_output(
            ["ps", "-axo", "pid,ppid,rss,vsz,%mem,etime,command"]),
    }
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    (log_dir / f"pageout_recovery_{label}_{stamp}.json").write_text(
        json.dumps(recovery, indent=2) + "\n")
    swap_grew = (before_swap is not None and after_swap is not None and after_swap > before_swap)
    recovery["soft_retry_permitted"] = not swap_grew
    # A small system pageout during recovery is itself a soft flag under the
    # resumed protocol; swap growth is caught as a hard failure by the next
    # block's preflight and persisted in this diagnostic either way.
    return recovery


def pageout_delta(row: dict[str, Any]) -> int:
    values = [row.get("pageouts_delta")]
    for key in ("preload_memory_preflight", "resident_memory_preflight"):
        nested = row.get(key)
        if isinstance(nested, dict):
            values.append(nested.get("pageouts_delta"))
    return max((int(value) for value in values if value is not None), default=0)


def capture_pageout_halt(pilot: Any, stage: str, label: str,
                         recent: list[dict[str, Any]]) -> None:
    log_dir = pilot.ROOT / stage / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    commands = (["date"], ["sysctl", "vm.swapusage"], ["memory_pressure"], ["vm_stat"],
                ["ps", "-axo", "pid,ppid,rss,vsz,%mem,etime,command"])
    sections = ["Pageout retry halt", f"policy={label}",
                "recent=" + json.dumps([{key: row.get(key) for key in
                    ("run_key", "repeat", "attempt", "invalid_reason", "pageouts_delta")}
                    for row in recent], default=str)]
    for command in commands:
        sections += ["", "$ " + " ".join(command), command_output(command)]
    sections += ["", "$ process contamination scan",
                 command_output(["pgrep", "-fl",
                    "run_.*calibration|run_.*evaluation|run_static_phaseaware|phaseguard-cpu|python"])]
    (log_dir / f"pageout_halt_{label}_{stamp}.txt").write_text("\n".join(sections) + "\n")


def stage_label(stage: str) -> str:
    return stage.replace("_", " ")


def write_baseline_from_runs(pilot: Any, campaign: Path, repeats: list[int]) -> dict[str, Any]:
    rows = [row for row in rows_for(pilot, "isolated_baseline")
            if row.get("policy_arg") == "llm-only" and row.get("status") == "valid"
            and int(row["repeat"]) in repeats]
    by_repeat = {int(row["repeat"]): row for row in rows}
    selected = [by_repeat[index] for index in repeats if index in by_repeat]
    if len(selected) != 5:
        raise RuntimeError("isolated baseline requires five valid repeats")
    tpot = [float(row["p95_tpot_ms"]) for row in selected]
    ttft = [float(row["p95_ttft_ms"]) for row in selected]
    tpot_med, ttft_med = float(np.median(tpot)), float(np.median(ttft))
    variation = max(max(abs(value / tpot_med - 1) for value in tpot),
                    max(abs(value / ttft_med - 1) for value in ttft))
    payload = {
        "stage": "isolated_baseline", "valid_repeats": 5,
        "repeat_ids": repeats, "p95_tpot_ms": tpot_med, "p95_ttft_ms": ttft_med,
        "tpot_threshold_ms": 1.10 * tpot_med, "ttft_threshold_ms": 1.10 * ttft_med,
        "tpot_slo_multiplier": 1.10, "ttft_slo_multiplier": 1.10,
        "max_run_level_p95_deviation": variation,
        "variation_within_5pct": variation <= .05,
        "run_keys": [row["run_key"] for row in selected],
    }
    path = campaign / "isolated_baseline" / "baseline.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def run_isolated_baseline(pilot: Any, user: argparse.Namespace, campaign: Path,
                          seed: int) -> Path:
    args = base_namespace(pilot, user, "isolated_baseline", seed)
    for cohort in range(3):
        repeats = list(range(cohort * 5, cohort * 5 + 5))
        for repeat in repeats:
            ensure_block(pilot, args, ("llm-only", "llm-only", 0, 0), repeat)
        payload = write_baseline_from_runs(pilot, campaign, repeats)
        if payload["variation_within_5pct"]:
            return campaign / "isolated_baseline" / "baseline.json"
        time.sleep(user.sentinel_cooldown)
    raise RuntimeError("isolated baseline p95 variation exceeded 5% in three cohorts")


def median(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def choose(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    feasible = [row for row in candidates if row["strict_feasible"]]
    if not feasible:
        return None
    best_qps = max(float(row["median_retrieval_qps"]) for row in feasible)
    near = [row for row in feasible if float(row["median_retrieval_qps"]) >= best_qps / 1.03]
    return min(near, key=lambda row: (int(row["prefill_cap"]) + int(row["decode_cap"]),
                                     int(row["decode_cap"]), str(row["policy"])))


def summarize_and_freeze(pilot: Any, campaign: Path, seed: int,
                         baseline: Path) -> dict[str, Any]:
    valid = [row for row in rows_for(pilot, "calibration") if row.get("status") == "valid"]
    summary: list[dict[str, Any]] = []
    for label, policy_arg, prefill, decode in CALIBRATION_POLICIES:
        rows = sorted([row for row in valid if row.get("policy") == label],
                      key=lambda row: int(row["repeat"]))[:3]
        if len(rows) != 3:
            raise RuntimeError(f"calibration incomplete for {label}: {len(rows)}/3")
        summary.append({
            "policy": label, "policy_arg": policy_arg, "prefill_cap": prefill,
            "decode_cap": decode, "valid_repeats": 3,
            "median_normalized_p95_tpot": median(rows, "normalized_p95_tpot"),
            "median_normalized_p95_ttft": median(rows, "normalized_p95_ttft"),
            "joint_slo_pass_count": sum(bool(row["joint_slo_pass"]) for row in rows),
            "strict_feasible": all(bool(row["joint_slo_pass"]) for row in rows),
            "median_retrieval_qps": median(rows, "total_retrieval_goodput_qps"),
            "run_keys": [row["run_key"] for row in rows],
        })
    best_fixed = choose([row for row in summary if row["policy_arg"] in {"fixed", "fixed0"}])
    best_phase = choose([row for row in summary if row["policy_arg"] == "phasegate"])
    matched: list[dict[str, Any]] = []
    by_name = {row["policy"]: row for row in summary}
    for cap in (1, 2):
        fixed, gate = by_name[f"fixed{cap}"], by_name[f"phasegate4to{cap}"]
        tpot_ratio = gate["median_normalized_p95_tpot"] / fixed["median_normalized_p95_tpot"]
        ttft_ratio = gate["median_normalized_p95_ttft"] / fixed["median_normalized_p95_ttft"]
        matched.append({
            "fixed": fixed["policy"], "phasegate": gate["policy"],
            "tpot_ratio": tpot_ratio, "ttft_ratio": ttft_ratio,
            "qps_gain": gate["median_retrieval_qps"] / fixed["median_retrieval_qps"] - 1,
            "passes_calibration_condition": tpot_ratio <= 1.03 and ttft_ratio <= 1.03
            and gate["median_retrieval_qps"] > fixed["median_retrieval_qps"],
            "absolute_slo_pass": gate["strict_feasible"],
        })
    payload = {
        "frozen_at": datetime.now().isoformat(), "git_commit": command_output(
            ["git", "rev-parse", "HEAD"], REPO),
        "selection_data": "calibration only", "calibration_seed": seed,
        "evaluation_seed": seed + 10_000_000, "baseline_file": str(baseline),
        "slo_multiplier": 1.10,
        "tie_rule": "within 3% QPS choose lower total cap, then lower decode cap, then simpler name",
        "best_fixed": best_fixed, "best_phasegate": best_phase,
        "latency_matched_pairs": matched, "calibration_summary": summary,
    }
    path = campaign / "frozen_selection.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    (campaign / "processed" / "frozen_selection.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    return payload
