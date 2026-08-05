#!/usr/bin/env python3
"""Functional checks for the production event observer and token fast path."""
from __future__ import annotations

import inspect
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import faiss  # noqa: E402
from phaseguard.observer import (EventBuffer, NativeMemoryMonitor, finalize_event_log,
    reconstruct_event_log)  # noqa: E402
from phaseguard.phase_monitor import GPUPhase, PhaseMonitor  # noqa: E402
from phaseguard.policies import TimeGateController  # noqa: E402
from phaseguard.shared_index_manager import SharedIndexTaskManager  # noqa: E402


class PermitRecorder:
    def __init__(self) -> None:
        self.values: list[int] = []

    def set_permits(self, value: int) -> None:
        self.values.append(value)


def request_fixture(start: float) -> dict[str, object]:
    tokens = [start + .020, start + .030, start + .040]
    return {"request_id": "r0", "prefill_start": start, "prefill_end": start + .015,
            "decode_end": start + .045, "token_timestamps": tokens}


def main() -> None:
    monitor = PhaseMonitor()
    source = inspect.getsource(PhaseMonitor.record_token)
    assert "_lock" not in source and "subprocess" not in source and "open(" not in source
    with patch.object(subprocess, "run", side_effect=AssertionError("token path subprocess")):
        first = monitor.record_token("r0")
        second = monitor.record_token("r0")
    assert second > first

    buffer = EventBuffer()
    buffer.emit("cap_change", timestamp=first, requested_cap=0, active_query_count=0)
    phase_events = [
        {"timestamp": first, "request_id": None, "phase": "IDLE", "event": "start"},
        {"timestamp": first + .001, "request_id": "r0", "phase": "PREFILL", "event": "start"},
        {"timestamp": first + .016, "request_id": "r0", "phase": "DECODE", "event": "start"},
        {"timestamp": first + .046, "request_id": None, "phase": "IDLE", "event": "start"},
    ]
    events = finalize_event_log(
        run_id="run", run_key="key", repeat=0, policy="llm-only",
        phase_events=phase_events, requests=[request_fixture(first + .001)],
        runtime_events=buffer.events(), default_cap=0,
    )
    audit = reconstruct_event_log(events)
    assert audit["sequence_exact"] and audit["timestamps_monotonic"]
    assert audit["token_events"] == 3
    assert [event["event_type"] for event in events].count("request_complete") == 1

    with patch.object(subprocess, "run", side_effect=AssertionError("event monitor subprocess")):
        memory = NativeMemoryMonitor(1.0).start()
        time.sleep(.03)
        memory.stop()
    assert memory.audit()["subprocess_count"] == 0
    assert memory.audit()["observed_rate_hz"] <= 1.01

    recorder = PermitRecorder()
    time_events = EventBuffer()
    gate = TimeGateController(recorder, 2, 1, [(2, .01), (1, .01)],
                              event_sink=time_events.emit).start()
    time.sleep(.035)
    gate.stop()
    assert gate.audit()["phase_state_consulted"] is False
    assert all(event["event_type"] == "timegate_transition"
               for event in time_events.events())

    smoke_index = (REPO.parent /
                   "kv-uma-research/experiments/phaseguard/index/hnsw_smoke_10k_d384.faiss")
    faiss.omp_set_num_threads(1)
    query_events = EventBuffer()
    with SharedIndexTaskManager(smoke_index, 2, event_sink=query_events.emit) as manager:
        tasks = [manager.submit(f"t{i}", 128, 16, 6000 + i) for i in range(4)]
        results = [manager.wait(task, timeout=60) for task in tasks]
        runtime = query_events.events()
        snap = manager.demand_snapshot()
    query_log = finalize_event_log(
        run_id="query-run", run_key="query-key", repeat=0, policy="fixed2",
        phase_events=[{"timestamp": runtime[0]["timestamp"] - .001,
                       "request_id": None, "phase": GPUPhase.IDLE.value, "event": "start"}],
        requests=[], runtime_events=runtime, default_cap=2,
    )
    query_audit = reconstruct_event_log(query_log)
    expected = sum(int(result["queries"]) for result in results)
    assert query_audit["query_accounting_exact"]
    assert query_audit["completed_queries"] == expected == snap["completed_queries"]
    assert query_audit["active_never_negative"] and query_audit["admissions_within_cap"]
    assert manager.index_load_count == 1 and faiss.omp_get_max_threads() == 1

    preserved = EventBuffer()
    preserved.emit("cap_change", requested_cap=1)
    try:
        raise RuntimeError("orderly diagnostic exception")
    except RuntimeError:
        pass
    assert len(preserved.events()) == 1
    print("observer semantics: PASS")


if __name__ == "__main__":
    main()
