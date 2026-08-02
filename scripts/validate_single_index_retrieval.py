#!/usr/bin/env python3
"""Validate equivalence and concurrency of the single-index retrieval owner."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from phaseguard.cpu_task_manager import CPUTaskManager
from phaseguard.retrieval import load_index, make_queries, search_chunks

INDEX = REPO / "experiments/phaseguard/index/hnsw_100k_d384.faiss"
OUT = REPO / "experiments/static_phaseaware/single_index_retrieval_validation_apple_m2_pro_20260803"


class Backlog:
    def __init__(self, manager: CPUTaskManager, feeders: int = 16,
                 queries_per_task: int = 1024, chunk: int = 16,
                 seed: int = 820260801) -> None:
        self.manager = manager
        self.feeders = feeders
        self.queries_per_task = queries_per_task
        self.chunk = chunk
        self.seed = seed
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.completed_queries = 0
        self.completed_tasks = 0
        self.errors: list[str] = []
        self.threads: list[threading.Thread] = []

    def start(self) -> "Backlog":
        def feed(slot: int) -> None:
            seq = 0
            while not self.stop.is_set():
                task = self.manager.submit(f"cap-feed-{slot}", self.queries_per_task,
                                           self.chunk, self.seed + slot * 100_000 + seq)
                try:
                    result = self.manager.wait(task, timeout=600)
                except Exception as exc:
                    if not self.stop.is_set():
                        with self.lock:
                            self.errors.append(repr(exc))
                    return
                with self.lock:
                    self.completed_queries += int(result["queries"])
                    self.completed_tasks += 1
                seq += 1
        self.threads = [threading.Thread(target=feed, args=(i,), daemon=True)
                        for i in range(self.feeders)]
        for thread in self.threads:
            thread.start()
        return self

    def snapshot(self) -> int:
        with self.lock:
            return self.completed_queries

    def close(self) -> None:
        self.stop.set()


def sample_cap(manager: CPUTaskManager, backlog: Backlog, cap: int,
               settle_s: float, measure_s: float) -> dict[str, Any]:
    manager.set_permits(cap)
    transition_start = time.perf_counter()
    while manager.active_retrievals() > cap:
        if time.perf_counter() - transition_start > 5:
            raise RuntimeError(f"cap {cap} failed to converge")
        time.sleep(.0002)
    convergence_ms = (time.perf_counter() - transition_start) * 1000
    time.sleep(settle_s)
    q0 = backlog.snapshot()
    t0 = time.perf_counter()
    active: list[int] = []
    queue_depth: list[int] = []
    while time.perf_counter() - t0 < measure_s:
        state = manager.demand_snapshot()
        active.append(int(state["active_retrievals"]))
        queue_depth.append(int(state["retrieval_queue_depth"]))
        time.sleep(.001)
    elapsed = time.perf_counter() - t0
    q1 = backlog.snapshot()
    return {
        "cap": cap,
        "samples": len(active),
        "active_mean": float(np.mean(active)),
        "active_p95": float(np.percentile(active, 95)),
        "active_max": max(active),
        "queue_nonempty_fraction": float(np.mean(np.asarray(queue_depth) > 0)),
        "retrieval_qps": (q1 - q0) / elapsed,
        "transition_convergence_ms": convergence_ms,
        "cap_never_exceeded_after_convergence": max(active) <= cap,
    }


def fixed_query_equivalence(index_path: Path) -> dict[str, Any]:
    import faiss
    faiss.omp_set_num_threads(1)
    old_index, meta = load_index(index_path, 128)
    shared_index, _ = load_index(index_path, 128)
    queries = make_queries(821260801, 256, int(meta["dimensions"]))
    old_distances, old_ids = old_index.search(queries, 10)
    shared_distances, shared_ids = shared_index.search(queries, 10)

    def search_slice(bounds: tuple[int, int]) -> tuple[int, np.ndarray, np.ndarray]:
        faiss.omp_set_num_threads(1)
        start, end = bounds
        distances, ids = shared_index.search(queries[start:end], 10)
        return start, distances, ids

    bounds = [(start, min(start + 64, len(queries))) for start in range(0, len(queries), 64)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        pieces = sorted(executor.map(search_slice, bounds), key=lambda item: item[0])
    concurrent_distances = np.concatenate([item[1] for item in pieces])
    concurrent_ids = np.concatenate([item[2] for item in pieces])
    max_abs = float(np.max(np.abs(old_distances - concurrent_distances)))
    return {
        "query_count": len(queries),
        "top_k": 10,
        "old_vs_shared_ids_exact": bool(np.array_equal(old_ids, shared_ids)),
        "old_vs_concurrent_ids_exact": bool(np.array_equal(old_ids, concurrent_ids)),
        "old_vs_shared_distances_close": bool(np.allclose(old_distances, shared_distances,
                                                            rtol=1e-6, atol=1e-7)),
        "old_vs_concurrent_distances_close": bool(np.allclose(
            old_distances, concurrent_distances, rtol=1e-6, atol=1e-7)),
        "maximum_absolute_distance_difference": max_abs,
    }


def manager_equivalence_and_accounting(index_path: Path) -> dict[str, Any]:
    index, meta = load_index(index_path, 128)
    dimensions = int(meta["dimensions"])
    seed = 822260801
    query_count = 4096
    expected_checksum = sum(value for _, value in search_chunks(
        index, make_queries(seed, query_count, dimensions), 10, 16))
    with CPUTaskManager(str(index_path), 4, 128, 10) as manager:
        manager.set_permits(4)
        one = manager.submit("equivalence", query_count, 16, seed)
        result = manager.wait(one)
        baseline = manager.demand_snapshot()
        task_ids = [manager.submit(f"accounting-{i:03d}", 256, 16,
                                   seed + 1000 + i) for i in range(32)]
        results = [manager.wait(task, timeout=600) for task in task_ids]
        final = manager.demand_snapshot()
        owner_pids = manager.worker_pids()
        architecture = manager.retrieval_architecture
        alive_logical_workers = manager.alive_workers()
    expected_queries = 32 * 256
    accounted = int(final["completed_queries"]) - int(baseline["completed_queries"])
    request_ids = [str(row["request_id"]) for row in results]
    return {
        "architecture": architecture,
        "owner_pid_count": len(owner_pids),
        "logical_worker_count": alive_logical_workers,
        "manager_query_count_match": int(result["queries"]) == query_count,
        "manager_checksum_absolute_difference": abs(float(result["checksum"]) - expected_checksum),
        "submitted_task_ids_unique": len(set(task_ids)) == len(task_ids),
        "completed_request_ids_unique": len(set(request_ids)) == len(request_ids),
        "completed_request_ids_exact": set(request_ids) == {f"accounting-{i:03d}" for i in range(32)},
        "expected_accounting_queries": expected_queries,
        "accounted_queries": accounted,
        "no_query_loss_or_duplicate_completion": accounted == expected_queries,
        "result_errors": [row.get("error") for row in results if row.get("error")],
    }


def concurrency_validation(index_path: Path) -> dict[str, Any]:
    with CPUTaskManager(str(index_path), 4, 128, 10) as manager:
        backlog = Backlog(manager).start()
        time.sleep(1)
        fixed = [sample_cap(manager, backlog, cap, .25, 3.0) for cap in (0, 1, 2, 3, 4)]
        phasegate = []
        for decode_cap in (0, 1, 2, 3):
            prefill = sample_cap(manager, backlog, 4, .15, 1.0)
            decode = sample_cap(manager, backlog, decode_cap, .15, 1.0)
            phasegate.append({"policy": f"4to{decode_cap}", "prefill": prefill,
                              "decode": decode})
        backlog.close()
        errors = backlog.errors
    by_cap = {row["cap"]: row for row in fixed}
    qps1 = float(by_cap[1]["retrieval_qps"])
    return {
        "fixed_caps": fixed,
        "phasegate_caps": phasegate,
        "errors": errors,
        "all_fixed_caps_respected": all(row["cap_never_exceeded_after_convergence"] for row in fixed),
        "all_fixed_concurrency_p95_matches": all(abs(row["active_p95"] - row["cap"]) <= .1
                                                    for row in fixed),
        "all_phase_caps_respected": all(
            item[phase]["cap_never_exceeded_after_convergence"]
            for item in phasegate for phase in ("prefill", "decode")),
        "gil_parallelism_ratio_2_vs_1": (float(by_cap[2]["retrieval_qps"]) / qps1 if qps1 else 0),
        "gil_parallelism_ratio_4_vs_1": (float(by_cap[4]["retrieval_qps"]) / qps1 if qps1 else 0),
        "faiss_search_concurrent_not_gil_serialized": (
            qps1 > 0 and float(by_cap[2]["retrieval_qps"]) / qps1 >= 1.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.index = args.index.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "captured": datetime.now().astimezone().isoformat(),
        "index": str(args.index),
        "index_bytes": args.index.stat().st_size,
        "faiss_internal_threads_per_query": 1,
        "fixed_query_equivalence": fixed_query_equivalence(args.index),
        "manager_equivalence_and_accounting": manager_equivalence_and_accounting(args.index),
        "concurrency": concurrency_validation(args.index),
    }
    checks = [
        result["fixed_query_equivalence"]["old_vs_shared_ids_exact"],
        result["fixed_query_equivalence"]["old_vs_concurrent_ids_exact"],
        result["fixed_query_equivalence"]["old_vs_shared_distances_close"],
        result["fixed_query_equivalence"]["old_vs_concurrent_distances_close"],
        result["manager_equivalence_and_accounting"]["manager_query_count_match"],
        result["manager_equivalence_and_accounting"]["no_query_loss_or_duplicate_completion"],
        result["manager_equivalence_and_accounting"]["owner_pid_count"] == 1,
        result["manager_equivalence_and_accounting"]["logical_worker_count"] == 4,
        result["concurrency"]["all_fixed_caps_respected"],
        result["concurrency"]["all_fixed_concurrency_p95_matches"],
        result["concurrency"]["all_phase_caps_respected"],
        result["concurrency"]["faiss_search_concurrent_not_gil_serialized"],
        not result["manager_equivalence_and_accounting"]["result_errors"],
        not result["concurrency"]["errors"],
    ]
    result["passed"] = all(checks)
    (args.out / "single_index_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Single-index retrieval validation", "",
             f"- Result: **{'PASS' if result['passed'] else 'FAIL'}**",
             f"- Index: `{args.index}` ({args.index.stat().st_size:,} bytes)",
             "- Architecture: one spawned FAISS index-owner process, four query threads",
             "- FAISS internal threads per query: 1", "",
             "## Equivalence", "", "```json",
             json.dumps(result["fixed_query_equivalence"], indent=2), "```", "",
             "## Accounting", "", "```json",
             json.dumps(result["manager_equivalence_and_accounting"], indent=2), "```", "",
             "## Fixed-cap concurrency", "",
             "| Cap | Active mean | Active p95 | Active max | QPS | Queue non-empty |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in result["concurrency"]["fixed_caps"]:
        lines.append(f"| {row['cap']} | {row['active_mean']:.3f} | {row['active_p95']:.1f} | "
                     f"{row['active_max']} | {row['retrieval_qps']:.1f} | "
                     f"{row['queue_nonempty_fraction']:.3f} |")
    lines += ["", f"2-vs-1 QPS ratio: {result['concurrency']['gil_parallelism_ratio_2_vs_1']:.3f}",
              f"4-vs-1 QPS ratio: {result['concurrency']['gil_parallelism_ratio_4_vs_1']:.3f}",
              "", "PhaseGate stable PREFILL and DECODE samples are retained in the JSON output."]
    (args.out / "single_index_validation.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"passed": result["passed"], "out": str(args.out)}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
