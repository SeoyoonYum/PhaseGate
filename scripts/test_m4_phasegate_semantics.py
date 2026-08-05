#!/usr/bin/env python3
"""Functional checks required before the M4 campaign."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import faiss  # noqa: E402
from phaseguard.phase_monitor import GPUPhase  # noqa: E402
from phaseguard.policies import FixedWorkerPolicy, StaticCapsPolicy, TimeGateController  # noqa: E402
from phaseguard.shared_index_manager import SharedIndexTaskManager  # noqa: E402


class PermitRecorder:
    def __init__(self) -> None:
        self.values: list[int] = []

    def set_permits(self, value: int) -> None:
        self.values.append(value)


def main() -> None:
    for cap in (0, 1, 2):
        fixed = FixedWorkerPolicy(2, cap)
        assert all(fixed.select(phase, 2048) == cap for phase in GPUPhase)
    gate = StaticCapsPolicy(2, 2, 1)
    assert gate.select(GPUPhase.PREFILL, 2048) == 2
    assert gate.select(GPUPhase.DECODE, 2048) == 1

    recorder = PermitRecorder()
    timer = TimeGateController(recorder, 2, 1, [(2, .01), (1, .01)], .005).start()
    time.sleep(.045)
    timer.stop()
    assert {1, 2}.issubset(set(recorder.values))
    assert timer.audit()["phase_state_consulted"] is False

    index = REPO.parent / "kv-uma-research/experiments/phaseguard/index/hnsw_smoke_10k_d384.faiss"
    faiss.omp_set_num_threads(1)
    with SharedIndexTaskManager(index, 2, ef_search=128) as manager:
        assert manager.index_load_count == 1
        assert faiss.omp_get_max_threads() == 1
        tasks = [manager.submit(f"t{i}", 1024, 16, 9000 + i) for i in range(8)]
        deadline = time.monotonic() + 10
        while manager.active_retrievals() < 2 and time.monotonic() < deadline:
            time.sleep(.001)
        assert manager.active_retrievals() == 2
        assert manager.retrieval_queue_depth() > 0
        manager.set_permits(1)
        # The second active query drains non-preemptively; no new query may be
        # admitted by worker 1 after it completes while cap remains one.
        saw_overshoot = manager.active_retrievals() == 2
        assert saw_overshoot
        deadline = time.monotonic() + 10
        while manager.active_retrievals() > 1 and time.monotonic() < deadline:
            time.sleep(.001)
        assert manager.active_retrievals() <= 1
        time.sleep(.02)
        assert manager.active_retrievals() <= 1
        manager.set_permits(2)
        results = [manager.wait(task, timeout=60) for task in tasks]
        snap = manager.demand_snapshot()
        assert sum(int(row["queries"]) for row in results) == snap["completed_queries"]
        assert snap["admitted_queries"] == snap["completed_queries"]
        assert snap["submitted_tasks"] == snap["completed_tasks"]
    print("M4 PhaseGate semantics: PASS")


if __name__ == "__main__":
    main()
