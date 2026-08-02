#!/usr/bin/env python3
"""Measure production FAISS-HNSW residency scaling in fresh processes."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phaseguard.cpu_task_manager import CPUTaskManager
from phaseguard.metrics import vm_snapshot
from phaseguard.retrieval import load_index

DEFAULT_INDEX = REPO / "experiments/phaseguard/index/hnsw_100k_d384.faiss"
DEFAULT_OUT = REPO / "experiments/static_phaseaware/faiss_residency_audit_apple_m2_pro_20260802"


def run(command: list[str], timeout: float = 60.0) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                            check=False)
    return result.stdout + result.stderr


def swap_used_bytes() -> int | None:
    text = run(["sysctl", "vm.swapusage"], 5)
    match = re.search(r"used\s*=\s*([0-9.]+)([BKMG])", text, re.I)
    if not match:
        return None
    scale = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(match.group(1)) * scale[match.group(2).upper()])


def host_state() -> dict[str, Any]:
    vm = vm_snapshot()
    return {
        "timestamp": time.time(),
        "pageouts": vm.get("pageouts", 0),
        "swapouts": vm.get("swapouts", 0),
        "compressions": vm.get("compressions", 0),
        "decompressions": vm.get("decompressions", 0),
        "compressor_pages": vm.get("pages_occupied_by_compressor", 0),
        "stored_compressor_pages": vm.get("pages_stored_in_compressor", 0),
        "page_size": vm.get("page_size", 16384),
        "headroom_bytes": vm.get("headroom_bytes", 0),
        "swap_used_bytes": swap_used_bytes(),
        "memory_pressure": run(["memory_pressure", "-Q"], 10).strip(),
    }


def delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    fields = ("pageouts", "swapouts", "compressions", "decompressions",
              "compressor_pages", "stored_compressor_pages")
    result = {f"{name}_delta": int(after[name]) - int(before[name]) for name in fields}
    bswap, aswap = before.get("swap_used_bytes"), after.get("swap_used_bytes")
    result["swap_used_delta_bytes"] = None if bswap is None or aswap is None else aswap - bswap
    return result


def rss_bytes(pid: int) -> int:
    text = run(["ps", "-o", "rss=", "-p", str(pid)], 5).strip()
    return int(text.split()[0]) * 1024 if text else 0


def physical_footprint_bytes(text: str) -> int | None:
    match = re.search(r"phys_footprint:\s*([0-9]+)\s+B", text)
    if match:
        return int(match.group(1))
    match = re.search(r"Footprint:\s*([0-9]+)\s+B", text)
    return int(match.group(1)) if match else None


def process_snapshot(pids: list[int], raw: Path, label: str, index: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    raw.mkdir(parents=True, exist_ok=True)
    for pid in pids:
        footprint = run(["footprint", "-p", str(pid), "-f", "bytes", "--noCategories"], 90)
        vmmap = run(["vmmap", "-summary", str(pid)], 90)
        lsof = run(["lsof", "-p", str(pid)], 90)
        (raw / f"{label}_pid{pid}_footprint.txt").write_text(footprint)
        (raw / f"{label}_pid{pid}_vmmap.txt").write_text(vmmap)
        (raw / f"{label}_pid{pid}_lsof.txt").write_text(lsof)
        rows.append({
            "pid": pid,
            "rss_bytes": rss_bytes(pid),
            "physical_footprint_bytes": physical_footprint_bytes(footprint),
            "index_open": str(index) in lsof,
        })
    return {
        "processes": rows,
        "total_rss_bytes": sum(int(row["rss_bytes"]) for row in rows),
        "total_physical_footprint_bytes": (
            sum(int(row["physical_footprint_bytes"]) for row in rows)
            if all(row["physical_footprint_bytes"] is not None for row in rows) else None
        ),
    }


class BackloggedLoad:
    def __init__(self, manager: CPUTaskManager, feeders: int, queries: int,
                 chunk: int, seed: int) -> None:
        self.manager = manager
        self.feeders = feeders
        self.queries = queries
        self.chunk = chunk
        self.seed = seed
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.completed_queries = 0
        self.errors: list[str] = []
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        def feed(slot: int) -> None:
            sequence = 0
            while not self.stop.is_set():
                task = self.manager.submit(f"audit-{slot}", self.queries, self.chunk,
                                           self.seed + slot * 1_000_000 + sequence)
                try:
                    result = self.manager.wait(task, timeout=600)
                except Exception as exc:
                    if not self.stop.is_set():
                        with self.lock:
                            self.errors.append(repr(exc))
                    return
                with self.lock:
                    self.completed_queries += int(result["queries"])
                sequence += 1
        self.threads = [threading.Thread(target=feed, args=(slot,), daemon=True)
                        for slot in range(self.feeders)]
        for thread in self.threads:
            thread.start()

    def snapshot(self) -> int:
        with self.lock:
            return self.completed_queries

    def request_stop(self) -> None:
        self.stop.set()


def one_configuration(args: argparse.Namespace) -> dict[str, Any]:
    raw = args.out / "raw" / f"workers_{args.workers}"
    raw.mkdir(parents=True, exist_ok=True)
    parent_pid = os.getpid()
    initial_host = host_state()
    before = process_snapshot([parent_pid], raw, "before_index", args.index)

    # One parent-held control copy establishes the exact index residency cost.
    control_index, meta = load_index(args.index, args.ef_search)
    assert int(control_index.ntotal) == int(meta["vectors"])
    after_index_host = host_state()
    after_index = process_snapshot([parent_pid], raw, "after_control_index", args.index)

    manager: CPUTaskManager | None = None
    load: BackloggedLoad | None = None
    qps = 0.0
    active: dict[str, Any]
    worker_pids: list[int] = []
    try:
        if args.workers:
            manager = CPUTaskManager(str(args.index), args.workers,
                                     ef_search=args.ef_search, top_k=args.top_k).start()
            worker_pids = manager.worker_pids()
            load = BackloggedLoad(manager, max(16, args.workers + 1),
                                  args.queries_per_task, args.chunk,
                                  args.seed + args.workers * 10_000)
            load.start()
            time.sleep(args.warmup_seconds)
            q0 = load.snapshot()
            t0 = time.perf_counter()
            time.sleep(args.measure_seconds)
            elapsed = time.perf_counter() - t0
            q1 = load.snapshot()
            qps = (q1 - q0) / elapsed
            active_host = host_state()
            active = process_snapshot([parent_pid, *worker_pids], raw, "workers_active", args.index)
            demand = manager.demand_snapshot()
        else:
            time.sleep(args.warmup_seconds + args.measure_seconds)
            active_host = host_state()
            active = process_snapshot([parent_pid], raw, "workers_active", args.index)
            demand = {}
    finally:
        if load is not None:
            load.request_stop()
        if manager is not None:
            manager.close()
        del control_index
        gc.collect()
        time.sleep(args.release_seconds)

    released_host = host_state()
    released = process_snapshot([parent_pid], raw, "after_shutdown", args.index)
    result = {
        "captured": datetime.now().astimezone().isoformat(),
        "workers": args.workers,
        "parent_pid": parent_pid,
        "worker_pids": worker_pids,
        "index_path": str(args.index),
        "index_bytes": args.index.stat().st_size,
        "index_vectors": meta.get("vectors"),
        "index_dimensions": meta.get("dimensions"),
        "ef_search": args.ef_search,
        "top_k": args.top_k,
        "queries_per_task": args.queries_per_task,
        "chunk": args.chunk,
        "retrieval_qps": qps,
        "before_index": before,
        "after_control_index": after_index,
        "workers_active": active,
        "after_shutdown": released,
        "host_initial": initial_host,
        "host_after_index": after_index_host,
        "host_active": active_host,
        "host_released": released_host,
        "index_load_delta": delta(initial_host, after_index_host),
        "active_delta": delta(initial_host, active_host),
        "full_delta": delta(initial_host, released_host),
        "demand_snapshot": demand,
        "worker_errors": [] if load is None else load.errors,
        "retrieval_architecture": ("control_only" if manager is None
                                   else manager.retrieval_architecture),
    }
    (raw / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def write_summary(out: Path, rows: list[dict[str, Any]]) -> None:
    baseline = next(row for row in rows if row["workers"] == 0)
    base_rss = baseline["workers_active"]["total_rss_bytes"]
    base_phys = baseline["workers_active"]["total_physical_footprint_bytes"]
    csv_rows = []
    for row in rows:
        active = row["workers_active"]
        worker_processes = [p for p in active["processes"] if p["pid"] in row["worker_pids"]]
        phys = active["total_physical_footprint_bytes"]
        csv_rows.append({
            "workers": row["workers"],
            "parent_pid": row["parent_pid"],
            "worker_pids": ";".join(map(str, row["worker_pids"])),
            "index_bytes": row["index_bytes"],
            "before_index_total_rss_bytes": row["before_index"]["total_rss_bytes"],
            "after_control_index_total_rss_bytes": row["after_control_index"]["total_rss_bytes"],
            "active_tree_rss_bytes": active["total_rss_bytes"],
            "active_tree_rss_increase_vs_worker0_bytes": active["total_rss_bytes"] - base_rss,
            "active_tree_physical_footprint_bytes": phys,
            "active_tree_physical_increase_vs_worker0_bytes": (
                None if phys is None or base_phys is None else phys - base_phys),
            "worker_rss_bytes": sum(p["rss_bytes"] for p in worker_processes),
            "worker_physical_footprint_bytes": (
                sum(p["physical_footprint_bytes"] for p in worker_processes)
                if all(p["physical_footprint_bytes"] is not None for p in worker_processes) else None),
            "after_shutdown_rss_bytes": row["after_shutdown"]["total_rss_bytes"],
            "retrieval_qps": row["retrieval_qps"],
            "pageout_delta": row["active_delta"]["pageouts_delta"],
            "swap_delta_bytes": row["active_delta"]["swap_used_delta_bytes"],
            "compressor_pages_delta": row["active_delta"]["compressor_pages_delta"],
            "compressed_bytes_delta": row["active_delta"]["compressor_pages_delta"] * row["host_initial"]["page_size"],
            "queue_depth": row["demand_snapshot"].get("retrieval_queue_depth"),
            "alive_workers": row["demand_snapshot"].get("active_retrievals"),
            "worker_errors": ";".join(row["worker_errors"]),
        })
    fields = list(csv_rows[0])
    with (out / "faiss_residency_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    index_bytes = int(baseline["index_bytes"])
    increments = [int(row["active_tree_rss_increase_vs_worker0_bytes"]) for row in csv_rows[1:]]
    slopes = [(increments[i] - (increments[i - 1] if i else 0)) for i in range(len(increments))]
    median_slope = sorted(slopes)[len(slopes) // 2]
    likely = median_slope >= 0.5 * index_bytes
    verdict = "confirmed" if likely and all(s >= 0.35 * index_bytes for s in slopes) else ("likely" if likely else "absent")
    lines = [
        "# FAISS residency audit", "",
        f"- Timestamp: {datetime.now().astimezone().isoformat()}",
        f"- Exact index: `{baseline['index_path']}` ({index_bytes:,} bytes)",
        f"- Production retrieval architecture: `{rows[-1]['retrieval_architecture']}`.",
        "- Measurement control: the parent retains one identical index copy in every configuration; "
        "0→N increments isolate the production retrieval owner and its logical concurrency.",
        f"- Duplication decision: **{verdict}**", "",
        "| Workers | Tree RSS | RSS increase vs 0 | Physical footprint | Retrieval QPS | Pageouts | Swap delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in csv_rows:
        phys = row["active_tree_physical_footprint_bytes"]
        lines.append(
            f"| {row['workers']} | {row['active_tree_rss_bytes'] / 2**20:.1f} MiB | "
            f"{row['active_tree_rss_increase_vs_worker0_bytes'] / 2**20:.1f} MiB | "
            f"{('N/A' if phys is None else f'{phys / 2**20:.1f} MiB')} | "
            f"{row['retrieval_qps']:.1f} | {row['pageout_delta']} | {row['swap_delta_bytes']} |")
    lines += ["", "Per-additional-worker tree RSS increments: " +
              ", ".join(f"{value / 2**20:.1f} MiB" for value in slopes) + ".", "",
              "Raw `footprint`, `vmmap -summary`, `lsof`, host snapshots, and per-PID results "
              "are retained under `raw/workers_N/`."]
    (out / "faiss_residency_audit.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--run-one", action="store_true")
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--queries-per-task", type=int, default=4096)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=810260801)
    parser.add_argument("--warmup-seconds", type=float, default=4.0)
    parser.add_argument("--measure-seconds", type=float, default=10.0)
    parser.add_argument("--release-seconds", type=float, default=5.0)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.index = args.index.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.run_one:
        if args.workers is None or not 0 <= args.workers <= 4:
            raise SystemExit("--run-one requires --workers 0..4")
        one_configuration(args)
        return

    rows = []
    for workers in range(5):
        command = [sys.executable, str(Path(__file__).resolve()), "--run-one",
                   "--workers", str(workers), "--out", str(args.out), "--index", str(args.index),
                   "--ef-search", str(args.ef_search), "--top-k", str(args.top_k),
                   "--queries-per-task", str(args.queries_per_task), "--chunk", str(args.chunk),
                   "--seed", str(args.seed), "--warmup-seconds", str(args.warmup_seconds),
                   "--measure-seconds", str(args.measure_seconds),
                   "--release-seconds", str(args.release_seconds)]
        started = time.monotonic()
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                                timeout=900, check=False)
        (args.out / f"workers_{workers}_subprocess.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"workers={workers} failed after {time.monotonic()-started:.1f}s; "
                               f"see {args.out / f'workers_{workers}_subprocess.log'}")
        rows.append(json.loads((args.out / "raw" / f"workers_{workers}" / "result.json").read_text()))
    write_summary(args.out, rows)
    print(args.out / "faiss_residency_audit.md")


if __name__ == "__main__":
    main()
