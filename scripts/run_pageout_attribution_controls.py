#!/usr/bin/env python3
"""Randomized, non-calibration attribution study for global macOS pageouts."""
from __future__ import annotations

import csv
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from phaseguard.metrics import vm_snapshot

INDEX = REPO / "experiments/phaseguard/index/hnsw_100k_d384.faiss"
OUT = REPO / "experiments/static_phaseaware/pageout_attribution_controls_apple_m2_pro_20260803"
SEED = 860_263_001
IDLE_DURATION_S = 260.0
LOW_INTENSITY = {"idle", "llm-only", "fixed0"}
HIGH_INTENSITY = {"fixed4", "phasegate4to3"}


def command_output(command: list[str], timeout: float = 120.0) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                            check=False)
    return result.stdout + result.stderr


def rss_for(pattern: str) -> int | None:
    raw = command_output(["ps", "-axo", "rss=,command="], 10)
    values = []
    for line in raw.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and fields[0].isdigit() and re.search(pattern, fields[1], re.I):
            values.append(int(fields[0]) * 1024)
    return max(values) if values else None


def host_snapshot() -> dict[str, Any]:
    vm = vm_snapshot()
    pressure = command_output(["memory_pressure", "-Q"], 20).strip()
    free = re.search(r"free percentage:\s*(\d+)%", pressure)
    swap = command_output(["sysctl", "vm.swapusage"], 10)
    used = re.search(r"used\s*=\s*([0-9.]+)([BKMG])", swap, re.I)
    scale = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3}
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "pageouts": int(vm.get("pageouts", 0)),
        "compressor_pages": int(vm.get("pages_occupied_by_compressor", 0)),
        "stored_compressor_pages": int(vm.get("pages_stored_in_compressor", 0)),
        "swap_used_bytes": (None if not used else int(float(used.group(1)) * scale[used.group(2).upper()])),
        "memory_pressure": pressure,
        "memory_free_percent": None if not free else int(free.group(1)),
        "ardagent_rss_bytes": rss_for(r"ARDAgent"),
        "windowserver_rss_bytes": rss_for(r"WindowServer"),
    }


def process_tree(root_pid: int) -> tuple[int, int, list[int]]:
    raw = command_output(["ps", "-axo", "pid=,ppid=,rss=,command="], 20)
    data: dict[int, tuple[int, int, str]] = {}
    for line in raw.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) == 4 and all(part.isdigit() for part in fields[:3]):
            data[int(fields[0])] = (int(fields[1]), int(fields[2]) * 1024, fields[3])
    wanted, changed = {root_pid}, True
    while changed:
        changed = False
        for pid, (ppid, _, _) in data.items():
            if ppid in wanted and pid not in wanted:
                wanted.add(pid); changed = True
    model = sum(rss for pid, (_, rss, _) in data.items() if pid in wanted and pid == root_pid)
    retrieval = sum(rss for pid, (_, rss, cmd) in data.items()
                    if pid in wanted and (pid != root_pid or "phaseguard" in cmd.lower()))
    return model, retrieval, sorted(wanted)


def process_tree_physical_footprint(pids: list[int]) -> int | None:
    """Summed phys_footprint without accidentally including this controller."""
    total = 0
    for pid in pids:
        raw = command_output(["footprint", "-p", str(pid), "-f", "bytes", "--noCategories"], 20)
        match = re.search(r"phys_footprint:\s*([0-9]+)\s+B", raw)
        if not match:
            return None
        total += int(match.group(1))
    return total


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def workload_args(policy: str, window: int, root: Path, baseline: Path | None) -> list[str]:
    base = [sys.executable, str(REPO / "scripts/run_static_phaseaware_pilot.py"), "--run-one",
            "--stage", "paired_pilot", "--repeat", str(window), "--attempt", "1",
            "--prompt-seed", str(SEED + window * 10_000),
            "--query-seed", str(SEED + window * 10_000 + 5_000),
            "--model", "1.5B", "--context", "2048", "--output-tokens", "128",
            "--llm-requests", "100", "--sample-ms", "5", "--rss-sample-stride", "20",
            "--max-workers", "4", "--feeders", "16", "--queries-per-task", "4096",
            "--chunk", "16", "--ef-search", "128", "--top-k", "10", "--index", str(INDEX),
            "--warmup-s", "1", "--mem-limit-gb", "5.5", "--min-headroom-gb", "6.5",
            "--memory-idle-seconds", "30", "--sentinel-tolerance", "0.03",
            "--sentinel-cooldown", "30", "--sentinel-attempts", "4", "--sentinel-reps", "2",
            "--within-block-drift-tolerance", "0.03", "--within-block-qps-drift-tolerance", "0.05",
            "--min-duration-s", "5", "--min-completed-queries", "1000",
            "--primary-request-tpot", "mean", "--ttft-origin", "request_arrival", "--closed-loop-arrivals"]
    if policy == "llm-only":
        base += ["--policy", "llm-only", "--fixed-workers", "0", "--prefill-cap", "0", "--decode-cap", "0"]
    elif policy == "fixed0":
        base += ["--policy", "fixed0", "--fixed-workers", "0", "--prefill-cap", "0", "--decode-cap", "0"]
    elif policy == "fixed4":
        base += ["--policy", "fixed", "--fixed-workers", "4", "--prefill-cap", "4", "--decode-cap", "4"]
    elif policy == "phasegate4to3":
        base += ["--policy", "phasegate", "--fixed-workers", "3", "--prefill-cap", "4", "--decode-cap", "3"]
    else:
        raise ValueError(policy)
    if baseline is not None:
        base += ["--baseline-file", str(baseline)]
    return base


def last_run(root: Path, policy: str, window: int) -> dict[str, Any]:
    path = root / "paired_pilot/raw/runs.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    label = {"llm-only": "llm-only", "fixed0": "fixed0", "fixed4": "fixed4",
             "phasegate4to3": "phasegate4to3"}[policy]
    candidates = [row for row in rows if row.get("policy") == label and int(row.get("repeat", -1)) == window]
    if not candidates:
        raise RuntimeError(f"missing row {policy} window={window}")
    return max(candidates, key=lambda row: int(row.get("attempt", -1)))


def run_window(policy: str, window: int, root: Path, baseline: Path, out: Path) -> dict[str, Any]:
    started = time.monotonic(); before = host_snapshot()
    samples: dict[str, list[int]] = {"model": [], "retrieval": [], "physical": [], "ard": [], "window": []}
    stop = threading.Event(); process: subprocess.Popen[str] | None = None

    def sampler() -> None:
        next_footprint = 0.0
        while not stop.is_set():
            if process is not None and process.poll() is None:
                model, retrieval, pids = process_tree(process.pid)
                samples["model"].append(model); samples["retrieval"].append(retrieval)
                if time.monotonic() >= next_footprint:
                    footprint = process_tree_physical_footprint(pids)
                    if footprint is not None: samples["physical"].append(footprint)
                    next_footprint = time.monotonic() + 5.0
            ard, window_server = rss_for(r"ARDAgent"), rss_for(r"WindowServer")
            if ard is not None: samples["ard"].append(ard)
            if window_server is not None: samples["window"].append(window_server)
            stop.wait(.5)

    if policy == "idle":
        thread = threading.Thread(target=sampler, daemon=True); thread.start()
        time.sleep(IDLE_DURATION_S)
        returncode, result = 0, None
    else:
        env = os.environ.copy(); env["STATIC_PHASEAWARE_ROOT"] = str(root)
        process = subprocess.Popen(workload_args(policy, window, root, baseline), cwd=REPO, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        thread = threading.Thread(target=sampler, daemon=True); thread.start()
        output, _ = process.communicate(); returncode, result = process.returncode, output
        (out / "raw" / f"window_{window:02d}_{policy}.log").write_text(result or "")
    stop.set(); thread.join(timeout=5)
    after = host_snapshot(); elapsed = time.monotonic() - started
    row: dict[str, Any] = {
        "window": window, "policy": policy, "timestamp": before["timestamp"],
        "duration_s": elapsed, "global_pageout_delta": after["pageouts"] - before["pageouts"],
        "pageout_rate_per_s": (after["pageouts"] - before["pageouts"]) / elapsed,
        "swap_delta_bytes": (None if before["swap_used_bytes"] is None or after["swap_used_bytes"] is None
                             else after["swap_used_bytes"] - before["swap_used_bytes"]),
        "compressor_pages_delta": after["compressor_pages"] - before["compressor_pages"],
        "stored_compressor_pages_delta": after["stored_compressor_pages"] - before["stored_compressor_pages"],
        "memory_pressure_before": before["memory_pressure"], "memory_pressure_after": after["memory_pressure"],
        "memory_free_percent_before": before["memory_free_percent"],
        "memory_free_percent_after": after["memory_free_percent"],
        "ardagent_rss_before_bytes": before["ardagent_rss_bytes"],
        "ardagent_rss_after_bytes": after["ardagent_rss_bytes"],
        "windowserver_rss_before_bytes": before["windowserver_rss_bytes"],
        "windowserver_rss_after_bytes": after["windowserver_rss_bytes"],
        "ardagent_rss_peak_bytes": max(samples["ard"], default=None),
        "windowserver_rss_peak_bytes": max(samples["window"], default=None),
        "model_process_rss_peak_bytes": max(samples["model"], default=None),
        "retrieval_process_rss_peak_bytes": max(samples["retrieval"], default=None),
        "process_tree_physical_footprint_peak_bytes": max(samples["physical"], default=None),
        "returncode": returncode,
    }
    if policy != "idle" and returncode == 0:
        measured = last_run(root, policy, window)
        for key in ("status", "invalid_reason", "pageouts_delta", "swap_used_delta_bytes",
                    "resident_physical_footprint_bytes", "measured_physical_footprint_bytes",
                    "peak_resident_memory_bytes", "p95_tpot_ms", "p95_ttft_ms",
                    "total_retrieval_goodput_qps", "tpot_first_last_quarter_p95_ratio",
                    "ttft_first_last_quarter_p95_ratio", "retrieval_qps_first_last_quarter_ratio"):
            row[key] = measured.get(key)
    return row


def make_reference(root: Path, out: Path) -> Path:
    """Fresh LLM reference solely for normalized fields in control windows."""
    env = os.environ.copy(); env["STATIC_PHASEAWARE_ROOT"] = str(root)
    command = workload_args("llm-only", 999, root, None)
    result = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
    (out / "raw/reference_llm.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("control reference failed")
    row = last_run(root, "llm-only", 999)
    path = root / "isolated_baseline/baseline.json"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "p95_tpot_ms": row["p95_tpot_ms"], "p95_ttft_ms": row["p95_ttft_ms"],
        "median_inter_token_gap_ms": row["p50_tpot_ms"],
        "purpose": "pageout attribution controls only; not campaign baseline",
    }, indent=2) + "\n")
    return path


def classify(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    by = {name: [row for row in rows if row["policy"] == name] for name in LOW_INTENSITY | HIGH_INTENSITY}
    low = [row for name in LOW_INTENSITY for row in by[name]]
    ceiling_delta = max(int(row["global_pageout_delta"]) for row in low)
    ceiling_rate = max(float(row["pageout_rate_per_s"]) for row in low)
    normal = all((row["memory_free_percent_after"] is None or int(row["memory_free_percent_after"]) >= 10)
                 and int(row.get("swap_delta_bytes") or 0) == 0 for row in rows)
    low_positive_groups = sum(any(int(row["global_pageout_delta"]) > 0 for row in by[name])
                              for name in LOW_INTENSITY)
    high_median = {name: sorted(int(row["global_pageout_delta"]) for row in by[name])[len(by[name]) // 2]
                   for name in HIGH_INTENSITY}
    high_exceeds = any(value > ceiling_delta for value in high_median.values())
    # Predeclared decision: background needs positive pageouts in at least two
    # low-intensity classes, normal pressure/swap, and no high-intensity median
    # above the low-intensity maximum.
    background = normal and low_positive_groups >= 2 and not high_exceeds
    kind = "background_noise" if background else "workload_linked_or_inconclusive"
    rule = {
        "frozen_at": datetime.now().astimezone().isoformat(),
        "study_seed": SEED, "idle_duration_s": IDLE_DURATION_S,
        "predeclared_control_ceiling_method": "maximum observed among idle, LLM-only, and Fixed-0 windows",
        "background_pageout_delta_ceiling": ceiling_delta,
        "background_pageout_rate_per_s_ceiling": ceiling_rate,
        "classification": kind,
        "decision_protocol": {
            "background_noise": "normal pressure and zero swap; positive pageouts in at least two low-intensity classes; neither high-intensity median exceeds the low-intensity maximum",
            "otherwise": "retain zero-pageout rule and treat evidence as workload-linked or inconclusive",
        },
        "hard_invalid": ["swap_delta_bytes > 0", "memory pressure leaves normal state",
                         "pageout delta or rate exceeds frozen background ceiling",
                         "abnormal compressor or physical-footprint growth", "existing TPOT/TTFT/QPS drift failure"],
        "soft_flag": ("small positive global pageout at or below the frozen ceiling with zero swap, normal pressure, stable footprint/compression, and valid drift checks"
                      if background else None),
        "automatic_mlx_change": False,
    }
    return kind, rule


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(); out = args.out.resolve(); root = out / "workload_runs"
    (out / "raw").mkdir(parents=True, exist_ok=True)
    manifest = {"created": datetime.now().astimezone().isoformat(), "seed": SEED,
                "mlx_memory_limit_gb": 5.5, "execution_mode": "headless_ssh_tmux",
                "reference_is_not_a_campaign_block": True,
                "window_counts": {"idle": 10, "llm-only": 10, "fixed0": 10, "fixed4": 5, "phasegate4to3": 5},
                "idle_duration_s": IDLE_DURATION_S,
                "classification_protocol_predeclared": True}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    baseline = make_reference(root, out)
    schedule = [name for name, count in manifest["window_counts"].items() for _ in range(count)]
    random.Random(SEED).shuffle(schedule)
    (out / "randomized_order.json").write_text(json.dumps(schedule, indent=2) + "\n")
    rows: list[dict[str, Any]] = []
    for window, policy in enumerate(schedule):
        row = run_window(policy, window, root, baseline, out)
        rows.append(row); write_csv(out / "pageout_background_controls.csv", rows)
        (out / "progress.json").write_text(json.dumps({
            "completed_windows": len(rows), "total_windows": len(schedule),
            "last_policy": policy, "last_window": window,
            "updated": datetime.now().astimezone().isoformat(),
        }, indent=2) + "\n")
    kind, rule = classify(rows)
    (out / "frozen_pageout_validity_rule.json").write_text(json.dumps(rule, indent=2) + "\n")
    lines = ["# Pageout background attribution controls", "",
             f"- Classification: **{kind}**", f"- Random seed: {SEED}",
             f"- Frozen low-intensity pageout ceiling: {rule['background_pageout_delta_ceiling']} pages/window",
             f"- Frozen low-intensity pageout-rate ceiling: {rule['background_pageout_rate_per_s_ceiling']:.8f} pages/s",
             "", "| Policy | Windows | Positive-pageout windows | Median delta | Max delta | Median QPS |",
             "|---|---:|---:|---:|---:|---:|"]
    for policy in ("idle", "llm-only", "fixed0", "fixed4", "phasegate4to3"):
        group = [row for row in rows if row["policy"] == policy]
        deltas = sorted(int(row["global_pageout_delta"]) for row in group)
        qps = sorted(float(row["total_retrieval_goodput_qps"]) for row in group
                     if row.get("total_retrieval_goodput_qps") is not None)
        lines.append(f"| {policy} | {len(group)} | {sum(x > 0 for x in deltas)} | "
                     f"{deltas[len(deltas)//2]} | {max(deltas)} | "
                     f"{('N/A' if not qps else f'{qps[len(qps)//2]:.2f}')} |")
    lines += ["", "Controls are diagnostic only and are not eligible calibration, baseline, selection, or evaluation data."]
    (out / "pageout_background_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(out), "classification": kind, "windows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
