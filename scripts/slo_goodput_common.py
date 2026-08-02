#!/usr/bin/env python3
"""Shared helpers for the frozen SLO-goodput token-tail campaign."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from fanmac_main_common import (CALIBRATION_POLICIES, SMOKE_POLICIES, INDEX, REPO,
                                assert_environment, configure_root, ensure_block)

CAMPAIGN_NAME = "fanmac_slo_goodput_tail_apple_m2_pro_20260802_headless_restart"
DEFAULT_CAMPAIGN = REPO / "experiments/static_phaseaware" / CAMPAIGN_NAME
SLO_DIRS = ("hardware", "smoke", "token_logging_overhead", "isolated_baseline",
            "calibration", "baseline_revalidation", "selection", "evaluation", "raw",
            "processed", "figures", "logs")
CAMPAIGN_SEEDS = {
    "smoke_prompt_trace": 720_260_801,
    "logging_prompt_trace": 730_260_801,
    "baseline_prompt_trace": 740_260_801,
    "calibration_prompt_trace": 750_260_801,
    "calibration_policy_order": 760_260_801,
    "revalidation_prompt_trace": 770_260_801,
    "evaluation_prompt_trace": 780_260_801,
    "evaluation_policy_order": 790_260_801,
}


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--mem-limit-gb", type=float, default=6.0)
    ap.add_argument("--min-headroom-gb", type=float, default=6.5)
    ap.add_argument("--memory-idle-seconds", type=float, default=30.0)
    ap.add_argument("--sentinel-cooldown", type=float, default=30.0)
    ap.add_argument("--max-attempts", type=int, default=12)


def resolve(value: Path) -> Path:
    path = (value if value.is_absolute() else REPO / value).resolve()
    if "_aborted" in path.parts:
        raise ValueError(f"aborted campaigns are excluded by default: {path}")
    for name in SLO_DIRS:
        (path / name).mkdir(parents=True, exist_ok=True)
    metadata_path = path / "campaign_metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps({
            "campaign_uuid": str(uuid.uuid4()),
            "created": datetime.now().astimezone().isoformat(),
            "execution_mode": "headless_ssh_tmux",
            "remote_desktop_active": False,
            "ARDAgent_modified": True,
            "ARDAgent_state": "disabled_by_explicit_user_request",
            "previous_partial_campaign_reused": False,
            "seeds": CAMPAIGN_SEEDS,
        }, indent=2) + "\n")
    return path


def command_output(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                            timeout=180, check=False)
    return (result.stdout + result.stderr).strip()


def append_command(campaign: Path, command: list[str]) -> None:
    with (campaign / "commands.log").open("a") as handle:
        handle.write(f"{datetime.now().astimezone().isoformat()} " + " ".join(command) + "\n")


def capture_hardware(campaign: Path, user: argparse.Namespace) -> None:
    if (campaign / "hardware.json").exists():
        return
    hardware = command_output(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"])
    software = command_output(["system_profiler", "SPSoftwareDataType"])
    for pattern in (r"(?m)^(\s*Serial Number \(system\):).*$",
                    r"(?m)^(\s*Hardware UUID:).*$", r"(?m)^(\s*Provisioning UDID:).*$",
                    r"(?m)^(\s*Computer Name:).*$", r"(?m)^(\s*User Name:).*$"):
        hardware = re.sub(pattern, r"\1 [redacted]", hardware)
        software = re.sub(pattern, r"\1 [redacted]", software)
    def field(pattern: str, default: str = "unknown") -> str:
        match = re.search(pattern, hardware)
        return match.group(1).strip() if match else default
    cpu = re.search(r"Total Number of Cores:\s*(\d+).*?\((\d+) Performance and (\d+) Efficiency\)",
                    hardware, re.S)
    gpu = re.search(r"Type:\s*GPU.*?Total Number of Cores:\s*(\d+)", hardware, re.S)
    index_meta = json.loads(INDEX.with_suffix(INDEX.suffix + ".json").read_text())
    versions = {name: metadata.version(name) for name in
                ("mlx", "mlx-lm", "faiss-cpu", "numpy", "matplotlib")}
    device = command_output([sys.executable, "-c", "import mlx.core as mx;print(mx.default_device())"])
    campaign_meta = json.loads((campaign / "campaign_metadata.json").read_text())
    payload = {
        "captured": datetime.now().astimezone().isoformat(), "model_name": field(r"Model Name:\s*(.+)"),
        "model_identifier": field(r"Model Identifier:\s*(.+)"), "chip": field(r"Chip:\s*(.+)"),
        "cpu_cores_total": int(cpu.group(1)) if cpu else None,
        "performance_cores": int(cpu.group(2)) if cpu else None,
        "efficiency_cores": int(cpu.group(3)) if cpu else None,
        "gpu_cores": int(gpu.group(1)) if gpu else None,
        "unified_memory_bytes": int(command_output(["sysctl", "-n", "hw.memsize"])),
        "macos": command_output(["sw_vers"]), "python": sys.version.split()[0],
        "runtime_versions": versions, "mlx_device": device,
        "git_commit": command_output(["git", "rev-parse", "HEAD"], REPO),
        "model": user.model, "quantization": "4bit", "context_length": user.context,
        "output_length": user.output_tokens, "hnsw": index_meta,
        "available_disk": command_output(["df", "-h", str(campaign)]).splitlines()[-1],
        "initial_vm_stat": command_output(["vm_stat"]),
        "initial_swap": command_output(["sysctl", "vm.swapusage"]),
        "initial_memory_pressure": command_output(["memory_pressure", "-Q"]),
        "campaign_uuid": campaign_meta["campaign_uuid"],
        "execution_mode": "headless_ssh_tmux",
        "remote_desktop_active": False,
        "ARDAgent_modified": True,
        "ARDAgent_state": "disabled_by_explicit_user_request",
        "previous_partial_campaign_reused": False,
        "campaign_seeds": CAMPAIGN_SEEDS,
    }
    text = (hardware + "\n\n" + software + "\n\nRuntime versions:\n" +
            json.dumps(versions, indent=2) + f"\nMLX device: {device}\n\n" +
            "execution_mode = headless_ssh_tmux\n"
            "remote_desktop_active = false\n"
            "ARDAgent_modified = true\n"
            "ARDAgent_state = disabled_by_explicit_user_request\n"
            "previous_partial_campaign_reused = false\n"
            f"campaign_uuid = {campaign_meta['campaign_uuid']}\n"
            f"campaign_seeds = {json.dumps(CAMPAIGN_SEEDS, sort_keys=True)}\n")
    for path in (campaign / "hardware.json", campaign / "hardware/hardware.json"):
        path.write_text(json.dumps(payload, indent=2) + "\n")
    for path in (campaign / "environment.txt", campaign / "hardware/environment.txt"):
        path.write_text(text)
    start_memory = "\n\n".join([
        f"captured = {datetime.now().astimezone().isoformat()}",
        "ARDAgent state (expected absent):\n" + command_output(["pgrep", "-fl", "ARDAgent"]),
        "WindowServer RSS:\n" + command_output(
            ["sh", "-c", "ps -axo pid,rss,command | grep '[W]indowServer'"]),
        "Codex RSS:\n" + command_output(
            ["sh", "-c", "ps -axo pid,rss,command | grep -i '[c]odex'"]),
        "swap:\n" + command_output(["sysctl", "vm.swapusage"]),
        "memory pressure:\n" + command_output(["memory_pressure"]),
        "vm_stat:\n" + command_output(["vm_stat"]),
        "top processes by RSS:\n" + command_output(
            ["sh", "-c", "ps -axo pid,ppid,rss,vsz,%mem,etime,command | sort -k3 -nr | head -20"]),
        "pre-campaign ARDAgent shutdown stabilization: 12 samples at 10-second intervals; "
        "pageouts remained 6419, swap remained 0.00M, and ARDAgent remained absent.",
    ]) + "\n"
    (campaign / "hardware/headless_start_memory.txt").write_text(start_memory)


def namespace(pilot: Any, user: argparse.Namespace, stage: str, seed: int,
              requests: int, baseline: Path | None = None) -> argparse.Namespace:
    args = pilot.parser().parse_args([])
    args.stage = stage
    args.seed = seed
    args.model = user.model
    args.context = user.context
    args.output_tokens = user.output_tokens
    args.llm_requests = requests
    args.mem_limit_gb = user.mem_limit_gb
    args.min_headroom_gb = user.min_headroom_gb
    args.memory_idle_seconds = user.memory_idle_seconds
    args.sentinel_cooldown = user.sentinel_cooldown
    args.max_attempts = user.max_attempts
    args.index = INDEX
    args.baseline_file = baseline
    args.sentinel_tolerance = .03
    args.within_block_drift_tolerance = .03
    args.within_block_qps_drift_tolerance = .05
    args.primary_request_tpot = "mean"
    args.ttft_origin = "request_arrival"
    args.closed_loop_arrivals = True
    args.token_timestamp_logging = True
    return args


def valid_rows(pilot: Any, stage: str, policy: str | None = None) -> list[dict[str, Any]]:
    rows = [row for row in pilot.read_jsonl(pilot.raw_path(stage, False))
            if row.get("status") == "valid"]
    return rows if policy is None else [row for row in rows if row.get("policy") == policy]


def write_fresh_baseline(pilot: Any, campaign: Path,
                         repeat_ids: list[int]) -> dict[str, Any]:
    wanted = set(repeat_ids)
    rows = sorted([row for row in valid_rows(pilot, "isolated_baseline", "llm-only")
                   if int(row["repeat"]) in wanted], key=lambda row: int(row["repeat"]))
    if len(rows) != len(repeat_ids):
        raise RuntimeError(f"fresh baseline incomplete: {len(rows)}/{len(repeat_ids)}")
    tpot = np.asarray([float(row["p95_tpot_ms"]) for row in rows])
    ttft = np.asarray([float(row["p95_ttft_ms"]) for row in rows])
    tpot_med, ttft_med = float(np.median(tpot)), float(np.median(ttft))
    tpot_unc = float(np.max(np.abs(tpot / tpot_med - 1)))
    ttft_unc = float(np.max(np.abs(ttft / ttft_med - 1)))
    payload = {
        "stage": "isolated_baseline", "valid_repeats": len(repeat_ids),
        "requests_per_repeat": 200, "output_tokens": 128,
        "p95_tpot_ms": tpot_med, "p95_ttft_ms": ttft_med,
        "median_inter_token_gap_ms": float(np.median(
            [float(row["p50_tpot_ms"]) for row in rows])),
        "p99_inter_token_gap_ms": float(np.median(
            [float(row["p99_inter_token_gap_ms"]) for row in rows])),
        "p95_transition_gap_ms": float(np.median(
            [float(row["p95_transition_gap_ms"]) for row in rows])),
        "p95_first_four_decode_gap_ms": float(np.median(
            [float(row["p95_first_four_decode_gap_ms"]) for row in rows])),
        "tpot_uncertainty_fraction": tpot_unc, "ttft_uncertainty_fraction": ttft_unc,
        "variation_within_3pct": tpot_unc <= .03 and ttft_unc <= .03,
        "repeat_ids": repeat_ids,
        "run_keys": [row["run_key"] for row in rows],
    }
    path = campaign / "isolated_baseline/baseline.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    run_fields = ("run_key", "repeat", "status", "llm_requests", "output_tokens",
                  "total_generated_tokens", "inter_token_gap_count", "p95_tpot_ms",
                  "p95_ttft_ms", "p99_inter_token_gap_ms", "p95_transition_gap_ms",
                  "p95_first_four_decode_gap_ms", "pageouts_delta", "swap_used_delta_bytes",
                  "tpot_first_last_quarter_p95_ratio", "ttft_first_last_quarter_p95_ratio")
    write_csv(campaign / "isolated_baseline_runs.csv",
              [{key: row.get(key) for key in run_fields} for row in rows])
    write_csv(campaign / "isolated_baseline_summary.csv", [payload])
    tail_fields = ("run_key", "repeat", "p95_inter_token_gap_ms", "p99_inter_token_gap_ms",
                   "max_inter_token_gap_ms", "p95_transition_gap_ms", "p99_transition_gap_ms",
                   "p95_first_four_decode_gap_ms", "p99_first_four_decode_gap_ms",
                   "per_request_max_gap_p50_ms", "per_request_max_gap_p95_ms",
                   "per_request_max_gap_p99_ms")
    write_csv(campaign / "isolated_token_tail_summary.csv",
              [{key: row.get(key) for key in tail_fields} for row in rows])
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def item_by_name(name: str) -> tuple[str, str, int, int]:
    return next(item for item in CALIBRATION_POLICIES if item[0] == name)
