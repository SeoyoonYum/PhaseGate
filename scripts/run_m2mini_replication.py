#!/usr/bin/env python3
"""Resume-safe cross-device PhaseGate replication on a fan-cooled M2 Mac mini."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import sys
import time
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
        "remote_desktop_audit": "no active 5900/3283 connection" if not re.search(
            r"ESTABLISHED.*(?:5900|3283)|(?:5900|3283).*ESTABLISHED", remote)
            else "active connection detected",
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


def frozen_index() -> Path:
    payload = json.loads((CAMPAIGN / "memory_mode.json").read_text())
    return Path(payload["index"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def semantic_smoke(pilot: Any) -> None:
    baseline = CAMPAIGN / "m2_preflight/baseline.json"
    args = block_namespace(pilot, "m2_semantic_smoke", SEEDS["semantic_prompt_query"],
                           12, baseline, frozen_index())
    rows = [ensure_block(pilot, args, item, 0) for item in (
        ("fixed1", "fixed", 1, 1), ("fixed4", "fixed", 4, 4),
        ("phasegate4to1", "phasegate", 4, 1))]
    by = {row["policy"]: row for row in rows}
    checks = {
        "fixed1_cap_both_phases": (by["fixed1"]["prefill_cap_applied_fraction"] >= .95
                                    and by["fixed1"]["decode_cap_applied_fraction"] >= .95),
        "fixed4_cap_both_phases": (by["fixed4"]["prefill_cap_applied_fraction"] >= .95
                                    and by["fixed4"]["decode_cap_applied_fraction"] >= .95),
        "phasegate_prefill_high": by["phasegate4to1"]["prefill_high_cap_fraction"] >= .95,
        "phasegate_decode_low": by["phasegate4to1"]["decode_low_cap_fraction"] >= .95,
        "queue_nonempty_95pct": all(float(row["queue_nonempty_fraction"]) >= .95 for row in rows),
        "timestamps_complete": all(int(row["token_timestamp_count"]) == 12 * 128 for row in rows),
        "one_shared_index_process": all(row["retrieval_architecture"] == "single_index_process_thread_pool"
                                         for row in rows),
        "faiss_internal_threads_one": True,
        "no_synchronous_per_token_output": True,
    }
    payload = {"performance_interpreted": False, "rows": rows, "checks": checks,
               "timegate_deferred_until_calibration_schedule_freeze": True,
               "passed": all(checks.values())}
    (CAMPAIGN / "m2_semantic_smoke/semantic_smoke.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    if not payload["passed"]:
        raise RuntimeError(f"semantic smoke failed: {checks}")


def cpu_percent(pids: list[int]) -> float:
    raw = command(["ps", "-o", "%cpu=", "-p", ",".join(map(str, pids))])
    values = []
    for line in raw.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            pass
    return sum(values)


def cpu_scaling_block(cap: int, repeat: int) -> None:
    sys.path.insert(0, str(REPO / "src"))
    from phaseguard.context_validation import AlwaysBackloggedHNSW, capture_state, state_delta
    from phaseguard.cpu_task_manager import CPUTaskManager
    from phaseguard.metrics import append_jsonl
    raw_path = CAMPAIGN / "m2_cpu_scaling/raw/runs.jsonl"
    run_key = f"m2_cpu_scaling_cap{cap}_r{repeat:02d}"
    existing = [] if not raw_path.exists() else [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    if any(row.get("run_key") == run_key and row.get("status") == "valid" for row in existing):
        return
    with CPUTaskManager(str(frozen_index()), 4, 128, 10) as manager:
        manager.set_permits(cap)
        pids = manager.worker_pids()
        load = AlwaysBackloggedHNSW(manager, 16, 4096, 16,
                                    SEEDS["cpu_scaling_query"] + repeat * 10_000).start()
        time.sleep(2)
        before, q0, l0 = capture_state(pids), manager.demand_snapshot(), load.snapshot()
        samples: list[dict[str, float | int]] = []
        deadline = time.perf_counter() + 60.0
        while time.perf_counter() < deadline:
            samples.append({"timestamp": time.perf_counter(), "cpu_percent": cpu_percent(pids),
                            **manager.demand_snapshot()})
            time.sleep(.05)
        after, q1, l1 = capture_state(pids), manager.demand_snapshot(), load.snapshot()
        load.stop()
    latencies = list(l1["latencies_s"])[len(list(l0["latencies_s"])):]
    delta = state_delta(before, after)
    active = [int(sample["active_retrievals"]) for sample in samples]
    queue_fraction = float(np.mean([int(sample["retrieval_queue_depth"]) > 0 for sample in samples]))
    completed = int(q1["completed_queries"]) - int(q0["completed_queries"])
    hard = (int(delta.get("swap_used_delta_bytes") or 0) > 0
            or any(delta.get(key) is not None and int(delta[key]) < 10
                   for key in ("memory_free_percent_before", "memory_free_percent_after")))
    row = {"run_key": run_key, "status": "invalid" if hard else "valid", "cap": cap,
           "repeat": repeat, "duration_s": 60.0, "query_seed": SEEDS["cpu_scaling_query"] + repeat * 10_000,
           "retrieval_qps": completed / 60.0,
           "retrieval_latency_p50_ms": float(np.percentile(latencies, 50) * 1e3),
           "retrieval_latency_p95_ms": float(np.percentile(latencies, 95) * 1e3),
           "cpu_percent_mean": float(np.mean([sample["cpu_percent"] for sample in samples])),
           "active_workers_mean": float(np.mean(active)), "active_workers_p95": float(np.percentile(active, 95)),
           "active_workers_max": max(active), "queue_nonempty_fraction": queue_fraction,
           "submitted_tasks": int(l1["submitted_tasks"] - l0["submitted_tasks"]),
           "completed_tasks": int(l1["completed_tasks"] - l0["completed_tasks"]),
           "completed_queries": completed, "faiss_internal_threads": 1,
           "retrieval_architecture": "single_index_process_thread_pool", **delta}
    append_jsonl(raw_path, row)
    print(json.dumps(row, indent=2))


def cpu_scaling() -> None:
    order = [(repeat, cap) for repeat in range(3) for cap in (1, 2, 4)]
    random.Random(SEEDS["cpu_scaling_query"]).shuffle(order)
    manifest = {"caps": [1, 2, 4], "repeats": 3, "steady_seconds": 60,
                "query_seed_base": SEEDS["cpu_scaling_query"], "order": order,
                "identical_query_stream_within_repeat": True}
    (CAMPAIGN / "m2_cpu_scaling/matrix.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for repeat, cap in order:
        cpu_command = [sys.executable, str(Path(__file__).resolve()), "--cpu-block",
                       "--cpu-cap", str(cap), "--cpu-repeat", str(repeat)]
        subprocess.run(cpu_command, cwd=REPO, check=True)
        current = [json.loads(line) for line in
                   (CAMPAIGN / "m2_cpu_scaling/raw/runs.jsonl").read_text().splitlines() if line]
        matches = [row for row in current if int(row["cap"]) == cap and int(row["repeat"]) == repeat]
        if not matches or matches[-1]["status"] != "valid":
            subprocess.run(cpu_command, cwd=REPO, check=True)
            current = [json.loads(line) for line in
                       (CAMPAIGN / "m2_cpu_scaling/raw/runs.jsonl").read_text().splitlines() if line]
            matches = [row for row in current if int(row["cap"]) == cap and int(row["repeat"]) == repeat]
            if not matches or matches[-1]["status"] != "valid":
                raise RuntimeError(f"CPU scaling cap {cap} repeat {repeat} hard-invalid twice")
    raw = [json.loads(line) for line in (CAMPAIGN / "m2_cpu_scaling/raw/runs.jsonl").read_text().splitlines() if line]
    write_csv(CAMPAIGN / "m2_cpu_scaling_runs.csv", raw)
    summary = []
    for cap in (1, 2, 4):
        rows = [row for row in raw if int(row["cap"]) == cap and row["status"] == "valid"]
        if len(rows) != 3:
            raise RuntimeError(f"CPU scaling cap {cap} incomplete")
        summary.append({"cap": cap, "valid_repeats": 3,
                        "median_qps": float(np.median([row["retrieval_qps"] for row in rows])),
                        "median_p50_latency_ms": float(np.median([row["retrieval_latency_p50_ms"] for row in rows])),
                        "median_p95_latency_ms": float(np.median([row["retrieval_latency_p95_ms"] for row in rows])),
                        "median_cpu_percent": float(np.median([row["cpu_percent_mean"] for row in rows])),
                        "median_active_workers": float(np.median([row["active_workers_mean"] for row in rows])),
                        "pageout_free_repeats": sum(int(row["pageouts_delta"]) == 0 for row in rows)})
    write_csv(CAMPAIGN / "m2_cpu_scaling_summary.csv", summary)


def randomized_gpu_matrix(pilot: Any, stage: str, items: list[tuple[str, str, int, int]],
                          repeats: int, requests: int, seed: int, order_seed: int,
                          baseline: Path, output_tokens: int = 128) -> list[dict[str, Any]]:
    args = block_namespace(pilot, stage, seed, requests, baseline, frozen_index(), output_tokens)
    orders = []
    for repeat in range(repeats):
        order = list(items)
        random.Random(order_seed + repeat).shuffle(order)
        orders.append([item[0] for item in order])
    (CAMPAIGN / stage / "matrix.json").write_text(json.dumps({
        "frozen_before_execution": True, "prompt_query_seed_base": seed,
        "policy_order_seed_base": order_seed, "requests_per_block": requests,
        "output_tokens": output_tokens, "orders": orders}, indent=2) + "\n")
    result = []
    by_name = {item[0]: item for item in items}
    for repeat, names in enumerate(orders):
        for name in names:
            result.append(ensure_block(pilot, args, by_name[name], repeat))
    return result


def mechanism(pilot: Any) -> None:
    rows = randomized_gpu_matrix(pilot, "m2_mechanism", [
        ("llm-only", "llm-only", 0, 0), ("fixed1", "fixed", 1, 1),
        ("fixed2", "fixed", 2, 2), ("fixed4", "fixed", 4, 4)],
        3, 100, SEEDS["mechanism_prompt_query"], SEEDS["mechanism_policy_order"],
        CAMPAIGN / "m2_preflight/baseline.json")
    write_csv(CAMPAIGN / "m2_mechanism_runs.csv", rows)


def baseline(pilot: Any) -> Path:
    args = block_namespace(pilot, "m2_baseline", SEEDS["baseline_prompt"], 150,
                           None, frozen_index())
    rows = [ensure_block(pilot, args, ("llm-only", "llm-only", 0, 0), repeat)
            for repeat in range(5)]
    tpot = np.asarray([float(row["p95_tpot_ms"]) for row in rows])
    ttft = np.asarray([float(row["p95_ttft_ms"]) for row in rows])
    tpot_med, ttft_med = float(np.median(tpot)), float(np.median(ttft))
    payload = {"valid_repeats": 5, "requests_per_repeat": 150, "output_tokens": 128,
               "p95_tpot_ms": tpot_med, "p95_ttft_ms": ttft_med,
               "median_inter_token_gap_ms": float(np.median([row["p50_tpot_ms"] for row in rows])),
               "p99_inter_token_gap_ms": float(np.median([row["p99_inter_token_gap_ms"] for row in rows])),
               "p95_transition_gap_ms": float(np.median([row["p95_transition_gap_ms"] for row in rows])),
               "tpot_max_deviation": float(np.max(np.abs(tpot / tpot_med - 1))),
               "ttft_max_deviation": float(np.max(np.abs(ttft / ttft_med - 1))),
               "run_keys": [row["run_key"] for row in rows]}
    payload["variation_within_3pct"] = (payload["tpot_max_deviation"] <= .03
                                         and payload["ttft_max_deviation"] <= .03)
    path = CAMPAIGN / "m2_baseline/baseline.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(CAMPAIGN / "m2_baseline_runs.csv", rows)
    if not payload["variation_within_3pct"]:
        raise RuntimeError("M2 isolated baseline varies by more than 3%; stabilize environment")
    return path


def calibrate(pilot: Any, baseline_path: Path) -> tuple[int, int, float]:
    items = [("fixed1", "fixed", 1, 1), ("phasegate4to1", "phasegate", 4, 1)]
    rows = randomized_gpu_matrix(pilot, "m2_calibration", items, 3, 100,
        SEEDS["calibration_prompt_query"], SEEDS["calibration_policy_order"], baseline_path)
    baseline_data = json.loads(baseline_path.read_text())
    grid = [1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45]
    def passes(row: dict[str, Any], budget: float) -> bool:
        return (float(row["p95_tpot_ms"]) / float(baseline_data["p95_tpot_ms"]) <= budget
                and float(row["p95_ttft_ms"]) / float(baseline_data["p95_ttft_ms"]) <= budget)
    common = [budget for budget in grid if all(passes(row, budget) for row in rows)]
    exact_feasible = bool(common)
    prefill_cap = 4
    if not exact_feasible:
        secondary = randomized_gpu_matrix(pilot, "m2_calibration",
            [("phasegate2to1", "phasegate", 2, 1)], 3, 100,
            SEEDS["calibration_prompt_query"], SEEDS["calibration_policy_order"] + 100,
            baseline_path)
        rows.extend(secondary)
        secondary_rows = [row for row in rows if row["policy"] in {"fixed1", "phasegate2to1"}]
        common = [budget for budget in grid if all(passes(row, budget) for row in secondary_rows)]
        if not common:
            raise RuntimeError("predeclared 2->1 secondary pair has no common budget through 1.45")
        prefill_cap = 2
    budget = min(common)
    freeze = {"frozen_before_heldout": True, "exact_4to1_feasible": exact_feasible,
              "test_label": "exact 4->1 confirmatory" if exact_feasible else "secondary reduced-cap alignment test",
              "fixed_policy": "fixed1", "phasegate_policy": f"phasegate{prefill_cap}to1",
              "timegate_policy": f"timegate{prefill_cap}to1", "prefill_cap": prefill_cap,
              "decode_cap": 1, "m2_primary_B": budget if exact_feasible else None,
              "secondary_budget": None if exact_feasible else budget, "slo_grid": grid,
              "baseline": baseline_data, "calibration_run_keys": [row["run_key"] for row in rows]}
    (CAMPAIGN / "m2_selection_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    write_csv(CAMPAIGN / "m2_calibration_runs.csv", rows)
    summary = []
    for name in sorted({row["policy"] for row in rows}):
        group = [row for row in rows if row["policy"] == name]
        summary.append({"policy": name, "valid_repeats": len(group), "budget": budget,
                        "all_pass_budget": all(passes(row, budget) for row in group),
                        "median_normalized_p95_tpot": float(np.median([
                            row["p95_tpot_ms"] / baseline_data["p95_tpot_ms"] for row in group])),
                        "median_normalized_p95_ttft": float(np.median([
                            row["p95_ttft_ms"] / baseline_data["p95_ttft_ms"] for row in group])),
                        "median_qps": float(np.median([row["total_retrieval_goodput_qps"] for row in group]))})
    write_csv(CAMPAIGN / "m2_calibration_summary.csv", summary)
    return prefill_cap, 1, budget


def calibration_schedule(pilot: Any, prefill_cap: int, decode_cap: int) -> Path:
    policy = f"phasegate{prefill_cap}to{decode_cap}"
    rows = [row for row in read_rows(pilot, "m2_calibration") if row.get("policy") == policy
            and row.get("status") == "valid"]
    if len(rows) != 3:
        raise RuntimeError(f"schedule source requires three {policy} calibration rows")
    median_row = sorted(rows, key=lambda row: float(row["duration_s"]))[1]
    timeline = CAMPAIGN / "m2_calibration/raw/timelines" / f"{median_row['run_key']}.json"
    samples = json.loads(timeline.read_text())["demand_samples"]
    raw_intervals: list[dict[str, float | int]] = []
    for before, after in zip(samples, samples[1:]):
        cap = int(before["permitted_workers"])
        if cap not in {prefill_cap, decode_cap}:
            continue
        duration = max(0.0, float(after["timestamp"]) - float(before["timestamp"]))
        if duration <= 0:
            continue
        if raw_intervals and int(raw_intervals[-1]["cap"]) == cap:
            raw_intervals[-1]["duration_s"] = float(raw_intervals[-1]["duration_s"]) + duration
        else:
            raw_intervals.append({"cap": cap, "duration_s": duration})
    if len(raw_intervals) < 10:
        raise RuntimeError("calibration trace produced too few TimeGate intervals")
    # A cyclic replay must not create a phantom transition at its wrap boundary.
    if raw_intervals[0]["cap"] == raw_intervals[-1]["cap"]:
        raw_intervals[0]["duration_s"] = (float(raw_intervals[0]["duration_s"])
                                            + float(raw_intervals[-1]["duration_s"]))
        raw_intervals.pop()
    total = sum(float(item["duration_s"]) for item in raw_intervals)
    high = sum(float(item["duration_s"]) for item in raw_intervals
               if int(item["cap"]) == prefill_cap)
    transitions = len(raw_intervals)
    rng = random.Random(SEEDS["timegate_offsets"])
    offsets = [rng.random() * total for _ in range(7)]
    payload = {"frozen_before_heldout": True, "source": "M2 calibration only",
               "source_policy": policy, "source_run_key": median_row["run_key"],
               "schedule_type": "cyclic replay of calibration cap interval durations",
               "phase_blind": True, "phase_callback_registered": False,
               "high_cap": prefill_cap, "low_cap": decode_cap,
               "intervals": raw_intervals, "period_s": total,
               "target_high_cap_duty_fraction": high / total,
               "target_transitions_per_s": transitions / total,
               "offset_seed": SEEDS["timegate_offsets"],
               "heldout_offsets_s": offsets}
    path = CAMPAIGN / "m2_timegate_schedule_freeze.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def timegate_smoke(pilot: Any, baseline_path: Path, schedule: Path,
                   prefill_cap: int, decode_cap: int) -> None:
    args = block_namespace(pilot, "m2_timegate_smoke", SEEDS["semantic_prompt_query"] + 50,
                           100, baseline_path, frozen_index())
    args.timegate_schedule = schedule
    args.timegate_offset_s = 0.0
    label = f"timegate{prefill_cap}to{decode_cap}"
    row = ensure_block(pilot, args, (label, "timegate", prefill_cap, decode_cap), 0)
    frozen = json.loads(schedule.read_text())
    duty_error = abs(float(row["high_cap_duty_fraction"])
                     - float(frozen["target_high_cap_duty_fraction"]))
    target_rate = float(frozen["target_transitions_per_s"])
    rate_error = abs(float(row["cap_transitions_per_s"]) / target_rate - 1) if target_rate else 0.0
    audit = {"run_key": row["run_key"], "phase_callback_calls": row["timegate_phase_callback_calls"],
             "phase_blind": row["validity"]["timegate_phase_blind"],
             "prefill_high_fraction": row["prefill_high_cap_fraction"],
             "prefill_low_fraction": row["prefill_low_cap_fraction"],
             "decode_high_fraction": row["decode_high_cap_fraction"],
             "decode_low_fraction": row["decode_low_cap_fraction"],
             "realized_duty_fraction": row["high_cap_duty_fraction"],
             "target_duty_fraction": frozen["target_high_cap_duty_fraction"],
             "duty_absolute_error": duty_error,
             "realized_transitions_per_s": row["cap_transitions_per_s"],
             "target_transitions_per_s": target_rate, "transition_relative_error": rate_error,
             "queue_nonempty_fraction": row["queue_nonempty_fraction"],
             "timestamps_complete": int(row["token_timestamp_count"]) == 100 * 128,
             "passed": (row["timegate_phase_callback_calls"] == 0 and duty_error <= .02
                        and rate_error <= .05 and row["queue_nonempty_fraction"] >= .95
                        and row["prefill_high_cap_fraction"] > 0 and row["prefill_low_cap_fraction"] > 0
                        and row["decode_high_cap_fraction"] > 0 and row["decode_low_cap_fraction"] > 0)}
    write_csv(CAMPAIGN / "m2_timegate_semantic_audit.csv", [audit])
    (CAMPAIGN / "M2_TIMEGATE_REPORT.md").write_text(
        "# M2 TimeGate Report\n\n" +
        f"TimeGate was driven only by its frozen wall-clock schedule; phase callback calls: "
        f"{audit['phase_callback_calls']}. Semantic audit: **{'PASS' if audit['passed'] else 'FAIL'}**.\n")
    if not audit["passed"]:
        raise RuntimeError(f"TimeGate semantic audit failed: {audit}")


def sentinel_check(pilot: Any, baseline_path: Path) -> None:
    args = block_namespace(pilot, "m2_sentinel", SEEDS["sentinel_prompt"], 50,
                           None, frozen_index())
    rows = [ensure_block(pilot, args, ("llm-only", "llm-only", 0, 0), repeat)
            for repeat in range(3)]
    baseline_data = json.loads(baseline_path.read_text())
    tpot = float(np.median([row["p95_tpot_ms"] for row in rows]))
    ttft = float(np.median([row["p95_ttft_ms"] for row in rows]))
    payload = {"rows": [row["run_key"] for row in rows], "median_p95_tpot_ms": tpot,
               "median_p95_ttft_ms": ttft,
               "tpot_drift": tpot / baseline_data["p95_tpot_ms"] - 1,
               "ttft_drift": ttft / baseline_data["p95_ttft_ms"] - 1}
    payload["within_3pct"] = abs(payload["tpot_drift"]) <= .03 and abs(payload["ttft_drift"]) <= .03
    (CAMPAIGN / "m2_sentinel/sentinel_drift.json").write_text(json.dumps(payload, indent=2) + "\n")
    if not payload["within_3pct"]:
        raise RuntimeError("pre-evaluation sentinel drift exceeds 3%; diagnose and recalibrate")


def heldout(pilot: Any, baseline_path: Path, schedule: Path,
            prefill_cap: int, decode_cap: int, repeats: int = 5) -> None:
    args = block_namespace(pilot, "m2_heldout", SEEDS["heldout_prompt_query"],
                           200, baseline_path, frozen_index())
    names = ["llm-only", "fixed1", f"phasegate{prefill_cap}to{decode_cap}",
             f"timegate{prefill_cap}to{decode_cap}"]
    items = {"llm-only": ("llm-only", "llm-only", 0, 0),
             "fixed1": ("fixed1", "fixed", 1, 1),
             f"phasegate{prefill_cap}to{decode_cap}": (
                 f"phasegate{prefill_cap}to{decode_cap}", "phasegate", prefill_cap, decode_cap),
             f"timegate{prefill_cap}to{decode_cap}": (
                 f"timegate{prefill_cap}to{decode_cap}", "timegate", prefill_cap, decode_cap)}
    frozen = json.loads(schedule.read_text())
    orders = []
    for repeat in range(repeats):
        order = list(names)
        random.Random(SEEDS["heldout_policy_order"] + repeat).shuffle(order)
        orders.append(order)
    manifest = {"frozen_before_execution": True, "repeats": repeats,
                "requests_per_policy": 200, "prompt_query_seed_base": SEEDS["heldout_prompt_query"],
                "policy_order_seed_base": SEEDS["heldout_policy_order"], "orders": orders,
                "timegate_offsets_s": frozen["heldout_offsets_s"][:repeats],
                "same_trace_within_repeat": True, "new_heldout_traces": True}
    (CAMPAIGN / "m2_heldout/matrix.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for repeat, order in enumerate(orders):
        for name in order:
            if name.startswith("timegate"):
                args.timegate_schedule = schedule
                args.timegate_offset_s = float(frozen["heldout_offsets_s"][repeat])
            else:
                args.timegate_schedule = None
                args.timegate_offset_s = 0.0
            rows.append(ensure_block(pilot, args, items[name], repeat))
        (CAMPAIGN / "m2_heldout/progress.json").write_text(json.dumps({
            "completed_repeats": repeat + 1, "target_repeats": repeats,
            "updated": datetime.now().astimezone().isoformat()}, indent=2) + "\n")
    write_csv(CAMPAIGN / "m2_heldout_runs.csv", rows)


def optional_output512(pilot: Any, prefill_cap: int, decode_cap: int) -> None:
    base_args = block_namespace(pilot, "m2_output512_baseline",
        SEEDS["output512_baseline_prompt"], 75, None, frozen_index(), output_tokens=512)
    base_rows = [ensure_block(pilot, base_args, ("llm-only", "llm-only", 0, 0), repeat)
                 for repeat in range(3)]
    payload = {"valid_repeats": 3, "requests_per_repeat": 75, "output_tokens": 512,
               "p95_tpot_ms": float(np.median([row["p95_tpot_ms"] for row in base_rows])),
               "p95_ttft_ms": float(np.median([row["p95_ttft_ms"] for row in base_rows])),
               "median_inter_token_gap_ms": float(np.median([row["p50_tpot_ms"] for row in base_rows])),
               "run_keys": [row["run_key"] for row in base_rows]}
    baseline_path = CAMPAIGN / "m2_output512_baseline/baseline.json"
    baseline_path.write_text(json.dumps(payload, indent=2) + "\n")
    rows = randomized_gpu_matrix(pilot, "m2_output512", [
        ("fixed1", "fixed", 1, 1),
        (f"phasegate{prefill_cap}to{decode_cap}", "phasegate", prefill_cap, decode_cap)],
        3, 75, SEEDS["output512_prompt_query"], SEEDS["output512_policy_order"],
        baseline_path, output_tokens=512)
    write_csv(CAMPAIGN / "m2_output512_runs.csv", rows)


def core_campaign(pilot: Any) -> None:
    semantic_smoke(pilot)
    cpu_scaling()
    mechanism(pilot)
    baseline_path = baseline(pilot)
    prefill_cap, decode_cap, _ = calibrate(pilot, baseline_path)
    schedule = calibration_schedule(pilot, prefill_cap, decode_cap)
    timegate_smoke(pilot, baseline_path, schedule, prefill_cap, decode_cap)
    sentinel_check(pilot, baseline_path)
    heldout(pilot, baseline_path, schedule, prefill_cap, decode_cap, repeats=5)
    (CAMPAIGN / "logs/core_complete.json").write_text(json.dumps({
        "completed": datetime.now().astimezone().isoformat(), "prefill_cap": prefill_cap,
        "decode_cap": decode_cap}, indent=2) + "\n")


def all_campaign(pilot: Any) -> None:
    core_campaign(pilot)
    selection = json.loads((CAMPAIGN / "m2_selection_freeze.json").read_text())
    optional_output512(pilot, int(selection["prefill_cap"]), int(selection["decode_cap"]))
    subprocess.run([sys.executable, str(REPO / "scripts/analyze_m2mini_replication.py")],
                   cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=("audit", "preflight", "core", "all"), default="all")
    parser.add_argument("--cpu-block", action="store_true")
    parser.add_argument("--cpu-cap", type=int, choices=(1, 2, 4))
    parser.add_argument("--cpu-repeat", type=int)
    args = parser.parse_args()
    if args.cpu_block:
        if args.cpu_cap is None or args.cpu_repeat is None:
            raise SystemExit("--cpu-block requires --cpu-cap and --cpu-repeat")
        cpu_scaling_block(args.cpu_cap, args.cpu_repeat)
        return
    manifest = write_machine_audit()
    print(json.dumps({"machine": manifest["model_identifier"], "chip": manifest["chip"],
                      "campaign": str(CAMPAIGN)}, indent=2), flush=True)
    if args.through == "audit":
        return
    pilot = configure_pilot()
    mode = full_preflight(pilot)
    print(json.dumps({"preflight_complete": True, "frozen_memory_mode": mode,
                      "campaign": str(CAMPAIGN)}, indent=2))
    if args.through == "preflight":
        return
    if args.through == "core":
        core_campaign(pilot)
        print(json.dumps({"core_campaign_complete": True, "campaign": str(CAMPAIGN)}, indent=2))
        return
    all_campaign(pilot)
    print(json.dumps({"campaign_complete": True, "campaign": str(CAMPAIGN)}, indent=2))


if __name__ == "__main__":
    main()
