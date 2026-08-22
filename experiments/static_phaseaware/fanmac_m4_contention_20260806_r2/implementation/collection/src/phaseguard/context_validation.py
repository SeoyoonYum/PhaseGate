"""Focused context-validation measurement and host-state helpers."""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from common import cpuload, measure
from exp2_contention import build_cache
from phaseguard.metrics import percentile, vm_snapshot


@dataclass(frozen=True)
class HostState:
    timestamp: float
    vm: dict[str, int]
    swap_total_bytes: int | None
    swap_used_bytes: int | None
    swap_free_bytes: int | None
    memory_free_percent: int | None
    experiment_rss_bytes: int
    temperature_c: float | None = None
    gpu_frequency_mhz: float | None = None


def _size_bytes(value: str, unit: str) -> int:
    scale = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[unit.upper()]
    return int(float(value) * scale)


def swap_usage() -> tuple[int | None, int | None, int | None]:
    raw = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True,
                         text=True, timeout=5).stdout
    values: dict[str, int] = {}
    for name in ("total", "used", "free"):
        match = re.search(rf"{name}\s*=\s*([0-9.]+)([BKMG])", raw, re.I)
        if match: values[name] = _size_bytes(match.group(1), match.group(2))
    return values.get("total"), values.get("used"), values.get("free")


def memory_free_percent() -> int | None:
    raw = subprocess.run(["memory_pressure", "-Q"], capture_output=True,
                         text=True, timeout=10).stdout
    match = re.search(r"free percentage:\s*(\d+)%", raw)
    return int(match.group(1)) if match else None


def total_rss_bytes(pids: list[int]) -> int:
    total = 0
    for pid in sorted(set(pids)):
        raw = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        if raw:
            try: total += int(raw.split()[0]) * 1024
            except ValueError: pass
    return total


def capture_state(pids: list[int]) -> HostState:
    total, used, free = swap_usage()
    return HostState(time.time(), vm_snapshot(), total, used, free,
                     memory_free_percent(), total_rss_bytes([os.getpid(), *pids]))


def state_delta(before: HostState, after: HostState) -> dict[str, Any]:
    vm_keys = ("pageins", "pageouts", "swapins", "swapouts", "compressions",
               "decompressions", "pages_occupied_by_compressor", "pages_stored_in_compressor")
    delta: dict[str, Any] = {f"{key}_delta": after.vm.get(key, 0) - before.vm.get(key, 0)
                             for key in vm_keys}
    delta.update({
        "swap_used_delta_bytes": None if before.swap_used_bytes is None or after.swap_used_bytes is None
                                 else after.swap_used_bytes - before.swap_used_bytes,
        "headroom_before_bytes": before.vm.get("headroom_bytes", 0),
        "headroom_after_bytes": after.vm.get("headroom_bytes", 0),
        "memory_free_percent_before": before.memory_free_percent,
        "memory_free_percent_after": after.memory_free_percent,
        "experiment_rss_before_bytes": before.experiment_rss_bytes,
        "experiment_rss_after_bytes": after.experiment_rss_bytes,
    })
    return delta


class HNSWLoad:
    """Continuously refill short tasks in an existing persistent worker pool."""
    def __init__(self, manager: Any, workers: int, queries_per_task: int, seed: int) -> None:
        self.manager, self.workers = manager, workers
        self.queries_per_task, self.seed = queries_per_task, seed
        self.stop_event = threading.Event(); self.lock = threading.Lock()
        self.queries = 0; self.threads: list[threading.Thread] = []

    def start(self) -> "HNSWLoad":
        self.manager.set_permits(self.workers)
        def feed(slot: int) -> None:
            sequence = 0
            while not self.stop_event.is_set():
                task = self.manager.submit(f"validation-{slot}", self.queries_per_task, 1,
                                           self.seed + slot * 100_000 + sequence)
                result = self.manager.wait(task)
                with self.lock: self.queries += int(result["queries"])
                sequence += 1
        self.threads = [threading.Thread(target=feed, args=(i,), daemon=True)
                        for i in range(self.workers)]
        for thread in self.threads: thread.start()
        time.sleep(0.2)
        return self

    def snapshot(self) -> int:
        with self.lock: return self.queries

    def stop(self) -> None:
        self.stop_event.set(); self.manager.set_permits(self.manager.max_workers)
        for thread in self.threads: thread.join(timeout=120)
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError("HNSW feeder failed to drain")
        self.manager.set_permits(0)


class AlwaysBackloggedHNSW:
    """Keep more HNSW tasks outstanding than the worker pool can claim.

    One feeder thread owns one outstanding task at a time.  With feeders greater
    than the process-pool size, the manager's task queue remains non-empty even
    when every permitted worker is searching.  This avoids an arrival-limited
    CPU-throughput measurement.
    """
    def __init__(self, manager: Any, feeders: int, queries_per_task: int,
                 chunk: int, seed: int) -> None:
        if feeders <= manager.max_workers or queries_per_task < 1 or chunk < 1:
            raise ValueError("feeders must exceed workers; queries and chunk must be positive")
        self.manager, self.feeders = manager, feeders
        self.queries_per_task, self.chunk, self.seed = queries_per_task, chunk, seed
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.submitted_tasks = 0
        self.completed_tasks = 0
        self.completed_queries = 0
        self.latency_s = 0.0
        self.latencies_s: list[float] = []
        self.threads: list[threading.Thread] = []

    def start(self) -> "AlwaysBackloggedHNSW":
        def feed(slot: int) -> None:
            sequence = 0
            while not self.stop_event.is_set():
                task = self.manager.submit(f"steady-{slot:03d}", self.queries_per_task,
                                           self.chunk, self.seed + slot * 1_000_000 + sequence)
                with self.lock:
                    self.submitted_tasks += 1
                try:
                    result = self.manager.wait(task, timeout=3600)
                except RuntimeError:
                    if self.stop_event.is_set():
                        return
                    raise
                with self.lock:
                    latency = float(result["ended"]) - float(result["started"])
                    self.completed_tasks += 1
                    self.completed_queries += int(result["queries"])
                    self.latency_s += latency
                    self.latencies_s.append(latency)
                sequence += 1
        self.threads = [threading.Thread(target=feed, args=(slot,), daemon=True,
                                         name=f"steady-hnsw-{slot}")
                        for slot in range(self.feeders)]
        for thread in self.threads:
            thread.start()
        return self

    def snapshot(self) -> dict[str, float]:
        with self.lock:
            return {"submitted_tasks": float(self.submitted_tasks),
                    "completed_tasks": float(self.completed_tasks),
                    "completed_queries": float(self.completed_queries),
                    "latency_s": self.latency_s,
                    "latencies_s": list(self.latencies_s)}

    def stop(self) -> None:
        self.stop_event.set()
        self.manager.set_permits(self.manager.max_workers)
        for thread in self.threads:
            # Outstanding 4k-query tasks need not drain for a bounded benchmark.
            # The persistent manager shuts workers down immediately afterwards;
            # feeder threads are daemon threads and must not extend the timed block.
            thread.join(timeout=0.2)


class SyntheticRandomLoad:
    """One persistent validated native random-gather kernel, stopped gracefully."""
    def __init__(self, mb: int = 96) -> None:
        self.mb = mb; self.load: cpuload.CpuBandwidthLoad | None = None

    def start(self) -> "SyntheticRandomLoad":
        self.load = cpuload.CpuBandwidthLoad(100, threads=1, mb=self.mb,
                                            seconds=3600.0, mode="random").start()
        time.sleep(0.5); return self

    def throughput(self) -> float:
        return self.load.gbps if self.load is not None else 0.0

    def stop(self) -> None:
        if self.load is not None: self.load.stop()


def prefill_iterations(model: Any, context: int, warmups: int, iterations: int,
                       on_iteration: Callable[[int, float], None]) -> list[float]:
    timings: list[float] = []
    for index in range(warmups + iterations):
        cache = measure.make_prompt_cache(model); tokens = measure.make_tokens(context); mx.eval(tokens)
        start = time.perf_counter(); output = measure.forward_body(model, tokens, cache); mx.eval(output)
        latency = (time.perf_counter() - start) * 1e3
        if index >= warmups:
            timings.append(latency); on_iteration(index - warmups, latency)
        del cache, tokens, output; measure.free_buffers()
    return timings


def decode_iterations(model: Any, context: int, warmups: int, tokens: int,
                      on_iteration: Callable[[int, float], None]) -> list[float]:
    cache = build_cache(model, context); token = mx.array([[7]], dtype=mx.int32); mx.eval(token)
    output = None
    for _ in range(warmups): output = model(token, cache=cache); mx.eval(output)
    timings: list[float] = []
    for index in range(tokens):
        start = time.perf_counter(); output = model(token, cache=cache); mx.eval(output)
        latency = (time.perf_counter() - start) * 1e3
        timings.append(latency); on_iteration(index, latency)
    del cache, token, output; measure.free_buffers()
    return timings


def distribution_metrics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.all(np.isfinite(array)): raise ValueError("invalid timing distribution")
    q25, q75 = np.percentile(array, [25, 75])
    first = float(np.median(array[:max(1, len(array) // 4)]))
    last = float(np.median(array[-max(1, len(array) // 4):]))
    return {"median_ms": float(np.median(array)), "p95_ms": percentile(array, 95),
            "p99_ms": percentile(array, 99), "mean_ms": float(array.mean()),
            "cv": float(array.std(ddof=1) / array.mean()), "iqr_ms": float(q75 - q25),
            "thermal_drift_ratio": last / first if first else float("nan"),
            "spread_ratio": float(array.max() / np.median(array))}


def classify_clean(delta: dict[str, Any], metrics: dict[str, float],
                   minimum_headroom_bytes: int) -> dict[str, bool]:
    pageout_zero = delta["pageouts_delta"] == 0
    swap_known = delta["swap_used_delta_bytes"] is not None
    swap_increased = swap_known and delta["swap_used_delta_bytes"] > 0
    compressor_growth = (delta["pages_occupied_by_compressor_delta"] > 0 or
                         delta["pages_stored_in_compressor_delta"] > 0)
    low_headroom = min(delta["headroom_before_bytes"], delta["headroom_after_bytes"]) < minimum_headroom_bytes
    pressure_ok = all(value is None or value >= 10 for value in
                      (delta["memory_free_percent_before"], delta["memory_free_percent_after"]))
    memory_stable = pageout_zero and swap_known and not swap_increased and not low_headroom and pressure_ok
    thermal_suspect = metrics["thermal_drift_ratio"] > 1.10 or metrics["spread_ratio"] > 1.50
    return {"pageout_delta_zero": pageout_zero, "pageout_observed": not pageout_zero,
            "swap_usage_increased": bool(swap_increased),
            "compressor_growth_observed": compressor_growth,
            "low_memory_headroom": low_headroom, "memory_state_stable": memory_stable,
            "thermal_clean": not thermal_suspect, "thermal_suspect": thermal_suspect,
            "thermal_contaminated": metrics["thermal_drift_ratio"] > 1.25,
            "clean": memory_stable and not thermal_suspect}
