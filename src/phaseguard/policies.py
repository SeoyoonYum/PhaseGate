"""CPU permit-selection policies; the GPU schedule is deliberately unchanged."""
from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .phase_monitor import GPUPhase


class Policy:
    name = "base"

    def __init__(self, max_workers: int) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers

    def select(self, phase: GPUPhase, context: int, recent_tpot_ms: float | None = None) -> int:
        raise NotImplementedError


class UncoordinatedPolicy(Policy):
    name = "uncoordinated"

    def select(self, phase: GPUPhase, context: int, recent_tpot_ms: float | None = None) -> int:
        return self.max_workers


class FixedWorkerPolicy(Policy):
    """Phase-oblivious fixed CPU concurrency baseline."""
    name = "fixed_workers"

    def __init__(self, max_workers: int, workers: int) -> None:
        super().__init__(max_workers)
        if not 0 <= workers <= max_workers:
            raise ValueError("fixed workers outside [0, max_workers]")
        self.workers = workers

    def select(self, phase: GPUPhase, context: int, recent_tpot_ms: float | None = None) -> int:
        return self.workers


class SerializedPolicy(Policy):
    name = "serialized"

    def select(self, phase: GPUPhase, context: int, recent_tpot_ms: float | None = None) -> int:
        return self.max_workers if phase == GPUPhase.IDLE else 0


class StaticPhasePolicy(Policy):
    name = "static_phase"

    def __init__(self, max_workers: int, decode_workers: int) -> None:
        super().__init__(max_workers)
        if not 0 <= decode_workers <= max_workers:
            raise ValueError("decode_workers outside [0, max_workers]")
        self.decode_workers = decode_workers

    def select(self, phase: GPUPhase, context: int, recent_tpot_ms: float | None = None) -> int:
        return self.decode_workers if phase == GPUPhase.DECODE else self.max_workers


class StaticCapsPolicy(Policy):
    """Explicit static prefill/decode caps; IDLE always drains at max concurrency."""
    name = "static_caps"

    def __init__(self, max_workers: int, prefill_workers: int, decode_workers: int) -> None:
        super().__init__(max_workers)
        if not 0 <= prefill_workers <= max_workers:
            raise ValueError("prefill workers outside [0, max-workers]")
        if not 0 <= decode_workers <= max_workers:
            raise ValueError("decode workers outside [0, max-workers]")
        self.prefill_workers = prefill_workers
        self.decode_workers = decode_workers

    def select(self, phase: GPUPhase, context: int, recent_tpot_ms: float | None = None) -> int:
        if phase == GPUPhase.PREFILL:
            return self.prefill_workers
        if phase == GPUPhase.DECODE:
            return self.decode_workers
        return self.max_workers


@dataclass(frozen=True)
class ProfilePoint:
    model: str
    context: int
    phase: str
    workload: str
    workers: int
    p95_ms: float
    logical_qps: float


def load_profile(path: str | Path) -> list[ProfilePoint]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty profile: {path}")
    return [ProfilePoint(r["model"], int(r["context"]), r["phase"], r["workload"],
                         int(r["workers"]), float(r["p95_ms"]),
                         float(r.get("logical_qps", 0.0))) for r in rows]


class ProfilePolicy(Policy):
    name = "phaseguard"

    def __init__(self, max_workers: int, profile: list[ProfilePoint], model: str,
                 workload: str, baseline_tpot_ms: float, slo_multiplier: float) -> None:
        super().__init__(max_workers)
        if slo_multiplier < 1:
            raise ValueError("SLO multiplier must be >= 1")
        self.profile, self.model, self.workload = profile, model, workload
        self.slo_ms = baseline_tpot_ms * slo_multiplier

    def profiled_decode_cap(self, context: int) -> int:
        """Return the largest profiled worker count satisfying the TPOT SLO."""
        candidates = [p for p in self.profile if p.model == self.model and
                      p.workload == self.workload and p.phase == "DECODE"]
        if not candidates:
            raise ValueError("no matching decode profile points")
        nearest = min({p.context for p in candidates}, key=lambda n: abs(n - context))
        safe = [p.workers for p in candidates if p.context == nearest and p.p95_ms <= self.slo_ms]
        return min(self.max_workers, max(safe, default=0))

    def select(self, phase: GPUPhase, context: int, recent_tpot_ms: float | None = None) -> int:
        if phase != GPUPhase.DECODE:
            return self.max_workers
        return self.profiled_decode_cap(context)


class DemandAwareProfilePolicy(ProfilePolicy):
    """Apply the profiled decode cap only when application demand exceeds it."""
    name = "phaseguard_demand"

    def __init__(self, max_workers: int, profile: list[ProfilePoint], model: str,
                 workload: str, baseline_tpot_ms: float, slo_multiplier: float,
                 demand_source: Callable[[], dict[str, int]]) -> None:
        super().__init__(max_workers, profile, model, workload, baseline_tpot_ms, slo_multiplier)
        self.demand_source = demand_source

    def select(self, phase: GPUPhase, context: int, recent_tpot_ms: float | None = None) -> int:
        safe = super().select(phase, context, recent_tpot_ms)
        if phase != GPUPhase.DECODE:
            return self.max_workers
        snapshot = self.demand_source()
        outstanding = int(snapshot["outstanding_tasks"])
        return self.max_workers if outstanding <= safe else safe


class DemandAwareCapPolicy(Policy):
    """Demand-aware phase gate using a cap selected on separate calibration runs."""
    name = "phaseguard_target"

    def __init__(self, max_workers: int, decode_cap: int,
                 demand_source: Callable[[], dict[str, int]]) -> None:
        super().__init__(max_workers)
        if not 0 <= decode_cap <= max_workers:
            raise ValueError("decode cap outside [0, max_workers]")
        self.decode_cap = decode_cap
        self.demand_source = demand_source

    def select(self, phase: GPUPhase, context: int, recent_tpot_ms: float | None = None) -> int:
        if phase != GPUPhase.DECODE:
            return self.max_workers
        outstanding = int(self.demand_source()["outstanding_tasks"])
        return self.max_workers if outstanding <= self.decode_cap else self.decode_cap


class BandwidthThresholdPolicy(Policy):
    """Baseline using logical query throughput as the primary interference proxy."""
    name = "bandwidth_only"

    def __init__(self, max_workers: int, profile: list[ProfilePoint], threshold_qps: float) -> None:
        super().__init__(max_workers)
        self.threshold_qps = threshold_qps
        by_worker: dict[int, list[float]] = {}
        for p in profile:
            if p.phase == "DECODE":
                by_worker.setdefault(p.workers, []).append(p.logical_qps)
        self.qps = {k: sum(v) / len(v) for k, v in by_worker.items()}

    def select(self, phase: GPUPhase, context: int, recent_tpot_ms: float | None = None) -> int:
        if phase != GPUPhase.DECODE:
            return self.max_workers
        safe = [k for k, qps in self.qps.items() if k <= self.max_workers and qps <= self.threshold_qps]
        return max(safe, default=0)


class PolicyController:
    def __init__(self, policy: Policy, manager: object, context: int) -> None:
        self.policy, self.manager, self.context = policy, manager, context
        self.changes = 0
        self.overhead_s = 0.0
        self.last: int | None = None
        self._lock = threading.Lock()

    def __call__(self, phase: GPUPhase, request_id: str | None) -> int:
        with self._lock:
            t0 = time.perf_counter()
            permit = self.policy.select(phase, self.context)
            if permit != self.last:
                self.manager.set_permits(permit)
                self.changes += 1
                self.last = permit
            self.overhead_s += time.perf_counter() - t0
            return permit
