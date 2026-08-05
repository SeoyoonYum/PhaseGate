#!/usr/bin/env python3
"""Resume-safe cross-device PhaseGate replication on a fan-cooled M2 Mac mini."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATE = "20260805"
CAMPAIGN = REPO / "experiments/static_phaseaware" / f"fanmac_m2mini_replication_{DATE}"
INDEX_100K = REPO / "experiments/phaseguard/index/hnsw_100k_d384.faiss"
INDEX_50K = REPO / "experiments/phaseguard/index/hnsw_50k_d384_m2_fallback.faiss"
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
SEEDS = {
    "preflight_prompt_query": 926_805_001,
    "semantic_prompt_query": 926_805_101,
    "cpu_scaling_query": 926_805_201,
    "mechanism_prompt_query": 926_805_301,
    "mechanism_policy_order": 926_805_401,
    "baseline_prompt": 926_805_501,
    "calibration_prompt_query": 926_805_601,
    "calibration_policy_order": 926_805_701,
    "timegate_offsets": 926_805_801,
    "sentinel_prompt": 926_805_901,
    "heldout_prompt_query": 926_806_001,
    "heldout_policy_order": 926_806_101,
    "output512_baseline_prompt": 926_806_201,
    "output512_prompt_query": 926_806_301,
    "output512_policy_order": 926_806_401,
}


def command(command: list[str], timeout: int = 300) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return (result.stdout + result.stderr).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def campaign_dirs() -> None:
    for name in ("hardware", "logs", "m2_preflight", "m2_semantic_smoke",
                 "m2_cpu_scaling", "m2_mechanism", "m2_baseline", "m2_calibration",
                 "m2_timegate_smoke", "m2_sentinel", "m2_heldout",
                 "m2_output512_baseline", "m2_output512", "figures", "bundle"):
        (CAMPAIGN / name).mkdir(parents=True, exist_ok=True)


def field(text: str, pattern: str, default: str = "unknown") -> str:
    match = re.search(pattern, text, re.M)
    return match.group(1).strip() if match else default


def memory_pressure_state(raw: str) -> str:
    lowered = raw.lower()
    if "critical" in lowered:
        return "critical"
    if "warn" in lowered:
        return "warning"
    return "normal"


def model_revision() -> str:
    root = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-4bit/snapshots"
    snapshots = sorted(path.name for path in root.glob("*") if path.is_dir())
    return snapshots[-1] if snapshots else "unknown"


def write_machine_audit() -> dict[str, Any]:
    campaign_dirs()
    hardware = command(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"])
    software = command(["system_profiler", "SPSoftwareDataType"])
    model_identifier = field(hardware, r"Model Identifier:\s*(.+)")
    chip = field(hardware, r"Chip:\s*(.+)")
    cpu = re.search(r"Total Number of Cores:\s*(\d+).*?\((\d+) Performance and (\d+) Efficiency\)", hardware, re.S)
    gpu = re.search(r"Chipset Model:.*?Total Number of Cores:\s*(\d+)", hardware, re.S)
    packages = {name: importlib.metadata.version(name) for name in
                ("mlx", "mlx-lm", "numpy", "faiss-cpu")}
    pressure = command(["memory_pressure"])
    remote = command(["sh", "-c", "who; netstat -anv -p tcp 2>/dev/null | grep ESTABLISHED | grep -E '5900|3283' || true"])
    payload: dict[str, Any] = {
        "captured": datetime.now().astimezone().isoformat(),
        "repository_path": str(REPO),
        "branch": command(["git", "branch", "--show-current"]),
        "repository_commit": command(["git", "rev-parse", "HEAD"]),
        "initial_worktree_was_clean": True,
        "m4_manifest_available": False,
        "model_name": field(hardware, r"Model Name:\s*(.+)"),
        "model_identifier": model_identifier,
        "chip": chip,
        "cpu_cores_total": int(cpu.group(1)) if cpu else int(command(["sysctl", "-n", "hw.physicalcpu"])),
        "performance_cores": int(cpu.group(2)) if cpu else int(command(["sysctl", "-n", "hw.perflevel0.physicalcpu"])),
        "efficiency_cores": int(cpu.group(3)) if cpu else int(command(["sysctl", "-n", "hw.perflevel1.physicalcpu"])),
        "gpu_cores": int(gpu.group(1)) if gpu else None,
        "unified_memory_bytes": int(command(["sysctl", "-n", "hw.memsize"])),
        "macos": command(["sw_vers"]),
        "python": sys.version.split()[0],
        "packages": packages,
        "model": {"identifier": MODEL_ID, "revision": model_revision(), "quantization": "4bit"},
        "index": {"path": str(INDEX_100K), "sha256": sha256(INDEX_100K),
                  "bytes": INDEX_100K.stat().st_size,
                  "configuration": json.loads(INDEX_100K.with_suffix(".faiss.json").read_text())},
        "disk": command(["df", "-h", str(REPO)]),
        "power": command(["pmset", "-g", "batt"]),
        "power_settings": command(["pmset", "-g", "custom"]),
        "initial_vm_stat": command(["vm_stat"]),
        "initial_swap": command(["sysctl", "vm.swapusage"]),
        "initial_memory_pressure": pressure,
        "initial_memory_pressure_state": memory_pressure_state(pressure),
        "caffeinate_processes": command(["pgrep", "-fl", "caffeinate"]),
        "remote_desktop_audit": remote,
        "remote_desktop_active_connection": bool(re.search(r"ESTABLISHED.*(?:5900|3283)|(?:5900|3283).*ESTABLISHED", remote)),
        "execution_mode": "headless_ssh_tmux",
        "seeds": SEEDS,
    }
    if payload["model_identifier"] != "Mac14,3" or payload["chip"] != "Apple M2":
        raise RuntimeError(f"not the required Apple M2 Mac mini: {model_identifier} {chip}")
    if "AC Power" not in payload["power"] or re.search(r"lowpowermode\s+1", payload["power_settings"]):
        raise RuntimeError("AC power and disabled Low Power Mode are required")
    if payload["remote_desktop_active_connection"]:
        raise RuntimeError("active Remote Desktop connection detected")
    (CAMPAIGN / "m2_machine_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# M2 Machine Audit", "",
             f"- Device: {payload['model_name']} `{model_identifier}` ({chip})",
             f"- CPU: {payload['performance_cores']} performance + {payload['efficiency_cores']} efficiency cores",
             f"- GPU cores: {payload['gpu_cores']}",
             f"- Unified memory: {payload['unified_memory_bytes'] / 1024**3:.0f} GiB",
             f"- Python: {payload['python']}",
             f"- Packages: `{json.dumps(packages, sort_keys=True)}`",
             f"- Repository commit: `{payload['repository_commit']}`",
             f"- Model: `{MODEL_ID}` revision `{payload['model']['revision']}`, 4-bit",
             f"- 100k index SHA-256: `{payload['index']['sha256']}`",
             f"- Initial memory pressure: {payload['initial_memory_pressure_state']}",
             f"- Active Remote Desktop connection: {payload['remote_desktop_active_connection']}",
             "- Power: AC; Low Power Mode disabled; caffeinate active.", "",
             "The Remote Management service may be listening, but the audit found no active GUI connection."
             ]
    (CAMPAIGN / "M2_MACHINE_AUDIT.md").write_text("\n".join(lines) + "\n")
    return payload


def configure_pilot() -> Any:
    os.environ["STATIC_PHASEAWARE_ROOT"] = str(CAMPAIGN)
    sys.path.insert(0, str(REPO / "scripts"))
    import run_static_phaseaware_pilot as pilot
    if pilot.ROOT != CAMPAIGN:
        raise RuntimeError(f"pilot root mismatch: {pilot.ROOT}")
    return pilot


def block_namespace(pilot: Any, stage: str, seed: int, requests: int,
                    baseline: Path | None, index: Path, output_tokens: int = 128) -> argparse.Namespace:
    args = pilot.parser().parse_args([])
    args.stage = stage
    args.seed = seed
    args.model = "1.5B"
    args.context = 2048
    args.output_tokens = output_tokens
    args.llm_requests = requests
    args.mem_limit_gb = 5.5
    args.min_headroom_gb = 6.5
    args.memory_idle_seconds = 10.0
    args.sentinel_cooldown = 15.0
    args.sentinel_attempts = 4
    args.sentinel_reps = 2
    args.max_attempts = 2
    args.index = index
    args.baseline_file = baseline
    args.sentinel_tolerance = .03
    args.within_block_drift_tolerance = .10
    args.within_block_qps_drift_tolerance = .15
    args.primary_request_tpot = "mean"
    args.ttft_origin = "request_arrival"
    args.closed_loop_arrivals = True
    args.token_timestamp_logging = True
    args.soft_validity = True
    return args


def read_rows(pilot: Any, stage: str) -> list[dict[str, Any]]:
    return pilot.read_jsonl(pilot.raw_path(stage, False))


def label_for(item: tuple[str, str, int, int]) -> str:
    return item[0]


def ensure_block(pilot: Any, args: argparse.Namespace, item: tuple[str, str, int, int],
                 repeat: int) -> dict[str, Any]:
    label, policy, prefill, decode = item
    for attempt in (1, 2):
        existing = [row for row in read_rows(pilot, args.stage)
                    if row.get("policy") == label and int(row.get("repeat", -1)) == repeat]
        good = [row for row in existing if "p95_tpot_ms" in row and
                not row.get("hard_failure_flags") and row.get("status") == "valid"]
        if good:
            return good[-1]
        if any(int(row.get("attempt", 0)) == attempt for row in existing):
            continue
        command_line = pilot.block_command(args, policy, prefill, decode, repeat, attempt)
        result = subprocess.run(command_line, cwd=REPO, env=os.environ.copy())
        if result.returncode and attempt == 2:
            raise RuntimeError(f"block crashed twice: {label} repeat {repeat}")
    existing = [row for row in read_rows(pilot, args.stage)
                if row.get("policy") == label and int(row.get("repeat", -1)) == repeat]
    if not existing:
        raise RuntimeError(f"missing result after two attempts: {label} repeat {repeat}")
    return existing[-1]


def write_preflight_baseline(row: dict[str, Any]) -> Path:
    payload = {"stage": "m2_preflight", "p95_tpot_ms": row["p95_tpot_ms"],
               "p95_ttft_ms": row["p95_ttft_ms"],
               "median_inter_token_gap_ms": row["p50_tpot_ms"],
               "p99_inter_token_gap_ms": row["p99_inter_token_gap_ms"],
               "p95_transition_gap_ms": row["p95_transition_gap_ms"],
               "source_run_key": row["run_key"]}
    path = CAMPAIGN / "m2_preflight/baseline.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def preflight_hard_failure(row: dict[str, Any]) -> bool:
    pressure_values = [row.get("memory_free_percent_before"), row.get("memory_free_percent_after")]
    return (row.get("status") != "valid" or bool(row.get("hard_failure_flags"))
            or int(row.get("swap_used_delta_bytes") or 0) > 0
            or any(value is not None and int(value) < 10 for value in pressure_values))


def full_preflight(pilot: Any) -> str:
    args = block_namespace(pilot, "m2_preflight", SEEDS["preflight_prompt_query"],
                           50, None, INDEX_100K)
    llm = ensure_block(pilot, args, ("llm-only", "llm-only", 0, 0), 0)
    baseline = write_preflight_baseline(llm)
    args.baseline_file = baseline
    rows = [llm]
    for item in (("fixed1", "fixed", 1, 1), ("fixed4", "fixed", 4, 4),
                 ("phasegate4to1", "phasegate", 4, 1)):
        rows.append(ensure_block(pilot, args, item, 0))
    hard = [row["policy"] for row in rows if preflight_hard_failure(row)]
    headroom = [int(row.get("headroom_after_bytes") or 0) for row in rows]
    persistent_upward = (len(headroom) >= 3 and all(b < a for a, b in zip(headroom, headroom[1:]))
                         and headroom[0] - headroom[-1] > 256 * 1024**2)
    if persistent_upward and not hard:
        raise RuntimeError("persistent upward memory trend requires environment stabilization")
    mode = "REDUCED_50K_MECHANISM_ONLY" if hard else "FULL_100K"
    payload = {"captured": datetime.now().astimezone().isoformat(), "rows": rows,
               "hard_failure_policies": hard, "persistent_upward_memory_trend": persistent_upward,
               "frozen_memory_mode": mode, "index": str(INDEX_100K),
               "decision_rule": "50k only after 100k hard memory failure",
               "positive_pageout_alone_is_soft": True}
    (CAMPAIGN / "m2_full_workload_preflight.json").write_text(json.dumps(payload, indent=2) + "\n")
    if mode == "REDUCED_50K_MECHANISM_ONLY" and not INDEX_50K.exists():
        subprocess.run([sys.executable, str(REPO / "scripts/build_hnsw_index.py"),
                        "--output", str(INDEX_50K), "--vectors", "50000",
                        "--dimensions", "384", "--ef-construction", "80",
                        "--graph-degree", "32", "--seed", "20260729"], cwd=REPO, check=True)
    decision = ["# M2 Memory Mode Decision", "", f"Frozen mode: **{mode}**", "",
                f"- 100k hard-failure policies after one retry: {hard or 'none'}",
                f"- Persistent upward memory trend: {persistent_upward}",
                "- Positive pageout with zero swap and normal pressure was treated as a soft flag.",
                "- This decision was frozen before mechanism or held-out performance outcomes."]
    if mode != "FULL_100K":
        decision += ["", "All subsequent results are labeled **50k reduced-index mechanism-only** and are not QPS-comparable to 100k campaigns."]
    (CAMPAIGN / "M2_MEMORY_MODE_DECISION.md").write_text("\n".join(decision) + "\n")
    (CAMPAIGN / "memory_mode.json").write_text(json.dumps({"mode": mode,
        "index": str(INDEX_100K if mode == "FULL_100K" else INDEX_50K)}, indent=2) + "\n")
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=("audit", "preflight"), default="preflight")
    args = parser.parse_args()
    manifest = write_machine_audit()
    print(json.dumps({"machine": manifest["model_identifier"], "chip": manifest["chip"],
                      "campaign": str(CAMPAIGN)}, indent=2), flush=True)
    if args.through == "audit":
        return
    pilot = configure_pilot()
    mode = full_preflight(pilot)
    print(json.dumps({"preflight_complete": True, "frozen_memory_mode": mode,
                      "campaign": str(CAMPAIGN)}, indent=2))


if __name__ == "__main__":
    main()
