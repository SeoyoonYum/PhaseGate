#!/usr/bin/env python3
"""Functional and audit checks for the frozen bursty DemandGate path."""
from __future__ import annotations

import inspect
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from phaseguard.bursty_audit import audit_bursty_block  # noqa: E402
from phaseguard.demand_gate import (DemandGateController, DemandInterval,
    FrozenDemandTrace, generate_demand_trace)  # noqa: E402
from phaseguard.observer import EventBuffer, finalize_event_log, reconstruct_event_log  # noqa: E402
from phaseguard.shared_index_manager import SharedIndexTaskManager  # noqa: E402


SMOKE_INDEX = REPO / "experiments/phaseguard/index/hnsw_smoke_10k_d384.faiss"


def event_fields(sequence: int, demand_on: bool) -> dict[str, object]:
    return {"run_id": "manual", "demand_sequence": sequence, "demand_level": 0.25,
            "trace_seed": 99, "trace_offset_s": 0.0, "demand_on": demand_on}


def deterministic_trace_checks() -> None:
    for level in (0.05, 0.25, 1.0):
        first = generate_demand_trace(demand_level=level, seed=2026082100)
        second = generate_demand_trace(demand_level=level, seed=2026082100)
        assert first.to_dict() == second.to_dict()
        assert first.duration_s >= 7200.0
        assert min(item.duration_s for item in first.intervals) >= 0.25 - 1e-9
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            first.write(path)
            replayed = FrozenDemandTrace.load(path)
            assert replayed.to_dict() == first.to_dict()
            for offset in (0.0, 0.125, 17.0, 7199.75, 7201.0):
                assert replayed.state_at(offset) == first.state_at(offset)
                assert replayed.position(offset) == first.position(offset)
    continuous = generate_demand_trace(demand_level=1.0, seed=7)
    assert all(continuous.state_at(value) for value in (0.0, 10.0, 7199.9, 9000.0))
    assert continuous.scheduled_on_fraction(123.0, 456.0) == 1.0


def dependency_checks() -> None:
    signature = inspect.signature(DemandGateController.__init__)
    assert "phase" not in signature.parameters
    assert "policy" not in signature.parameters
    selection_source = inspect.getsource(DemandGateController._run)
    assert "GPUPhase" not in selection_source
    assert "policy" not in selection_source.lower()
    pilot = (REPO / "scripts/run_static_phaseaware_pilot.py").read_text()
    measured = pilot[pilot.index("if demand_trace is not None:\n            if manager is None"):
                     pilot.index("stop.set()", pilot.index("if demand_trace is not None:\n            if manager is None"))]
    assert "subprocess" not in measured and "total_rss_bytes" not in measured

    class FakeManager:
        def __init__(self) -> None:
            self.demand_on = True
            self.events: list[tuple[str, bool]] = []

        def set_demand_state(self, demand_on: bool, *,
                             events: list[tuple[str, dict[str, object]]]) -> float:
            self.demand_on = demand_on
            self.events.extend((kind, bool(fields["demand_on"])) for kind, fields in events)
            return time.perf_counter()

    trace = FrozenDemandTrace(0.25, 11, 0.08, (
        DemandInterval(0.0, 0.02, True), DemandInterval(0.02, 0.04, False),
        DemandInterval(0.04, 0.06, True), DemandInterval(0.06, 0.08, False)),
        {"kind": "controller-test"})
    fake = FakeManager()
    controller = DemandGateController(fake, trace, 0.0, run_id="dependency-test")
    controller.start(); time.sleep(.055); controller.stop()
    assert fake.events[0][0] == "demand_trace_start"
    assert any(kind == "demand_off" for kind, _ in fake.events)
    assert fake.events[-1] == ("demand_trace_end", False)
    assert controller.audit()["phase_state_consulted"] is False
    assert controller.audit()["policy_identity_consulted"] is False


def manual_gate_and_drain_checks() -> None:
    buffer = EventBuffer()
    with SharedIndexTaskManager(
        SMOKE_INDEX, 2, ef_search=128, top_k=10,
        event_sink=buffer.emit, demand_enabled=True,
    ) as manager:
        manager.set_permits(1)
        off_time = manager.set_demand_state(
            False, events=[("demand_off", event_fields(1, False))])
        task = manager.submit("bursty", 10_000, 5_000, 7100)
        time.sleep(.03)
        assert not [event for event in buffer.events()
                    if event["event_type"] == "query_chunk_started"
                    and float(event["timestamp"]) >= off_time]

        manager.set_demand_state(True, events=[("demand_on", event_fields(2, True))])
        deadline = time.monotonic() + 10
        while manager.active_retrievals() == 0 and time.monotonic() < deadline:
            time.sleep(.0005)
        assert manager.active_retrievals() == 1
        drain_off = manager.set_demand_state(
            False, events=[("demand_off", event_fields(3, False))])
        deadline = time.monotonic() + 10
        while manager.active_retrievals() > 0 and time.monotonic() < deadline:
            time.sleep(.0005)
        assert manager.active_retrievals() == 0
        time.sleep(.02)
        during_off = [event for event in buffer.events()
                      if event["event_type"] == "query_chunk_started"
                      and float(event["timestamp"]) >= drain_off]
        assert not during_off
        assert manager.demand_snapshot()["completed_queries"] in (5_000, 10_000)

        manager.set_demand_state(True, events=[("demand_on", event_fields(4, True))])
        result = manager.wait(task, timeout=60)
        runtime_snapshot = manager.demand_snapshot()
        assert int(result["queries"]) == 10_000
        assert runtime_snapshot["admitted_queries"] == runtime_snapshot["completed_queries"]

    runtime = buffer.events()
    start = min(float(event["timestamp"]) for event in runtime)
    end = max(float(event["timestamp"]) for event in runtime) + 1e-6
    events = finalize_event_log(
        run_id="manual", run_key="manual", repeat=0, policy="fixed1",
        phase_events=[{"timestamp": start - .001, "request_id": None,
                       "phase": "IDLE", "event": "start"}],
        requests=[], runtime_events=runtime, default_cap=1)
    reconstructed = reconstruct_event_log(events)
    assert reconstructed["query_accounting_exact"]
    assert reconstructed["admissions_within_cap"]
    assert reconstructed["admissions_within_demand"]
    assert reconstructed["completed_queries"] == runtime_snapshot["completed_queries"]
    chunk_events = [event for event in events
                    if event["event_type"] in ("query_chunk_started",
                                                "query_chunk_completed")]
    assert chunk_events
    assert all(event["run_id"] == "manual" and int(event["seq"]) > 0
               and event["demand_level"] == 0.25 and event["trace_seed"] == 99
               and event["trace_offset_s"] == 0.0
               and isinstance(event["demand_on"], bool) for event in chunk_events)

    trace = FrozenDemandTrace(0.25, 99, 0.2, (
        DemandInterval(0.0, 0.05, False), DemandInterval(0.05, 0.10, True),
        DemandInterval(0.10, 0.15, False), DemandInterval(0.15, 0.20, True),
    ), {"kind": "test"})
    audit, overlap = audit_bursty_block(
        events=events, trace=trace, offset_s=0.0, block_start=start,
        block_end=end, llm_requests=1)
    assert audit["new_chunk_starts_during_off"] == 0
    assert audit["off_active_excluding_bounded_drain_s"] <= 1e-9
    assert overlap


def warmup_chunk_trace_identity_check() -> None:
    runtime = [
        {"timestamp": 1.0, "event_type": "query_chunk_started",
         "query_id": "warmup", "query_count": 16, "requested_cap": 1,
         "active_query_count": 1, "demand_on": True},
        {"timestamp": 1.1, "event_type": "query_chunk_completed",
         "query_id": "warmup", "query_count": 16, "requested_cap": 1,
         "active_query_count": 0, "demand_on": True},
        {"timestamp": 2.0, "event_type": "demand_trace_start",
         **event_fields(1, False)},
        {"timestamp": 2.0, "event_type": "demand_off", **event_fields(2, False)},
    ]
    events = finalize_event_log(
        run_id="warmup", run_key="warmup", repeat=0, policy="fixed1",
        phase_events=[], requests=[], runtime_events=runtime, default_cap=1)
    chunks = [event for event in events if event["event_type"].startswith("query_chunk_")]
    assert chunks
    assert all(event["demand_level"] == 0.25 and event["trace_seed"] == 99
               and event["trace_offset_s"] == 0.0 and event["demand_on"] is True
               for event in chunks)


def always_on_equivalence_check() -> None:
    def execute(demand_enabled: bool) -> tuple[int, int, int]:
        events = EventBuffer()
        with SharedIndexTaskManager(
            SMOKE_INDEX, 1, event_sink=events.emit, demand_enabled=demand_enabled,
        ) as manager:
            if demand_enabled:
                manager.set_demand_state(
                    True, events=[("demand_trace_start", event_fields(1, True)),
                                  ("demand_on", event_fields(2, True))])
            task = manager.submit("equivalence", 512, 16, 8100)
            result = manager.wait(task, timeout=60)
            snapshot = manager.demand_snapshot()
        return int(result["queries"]), int(snapshot["admitted_queries"]), int(
            snapshot["completed_queries"])
    assert execute(False) == execute(True) == (512, 512, 512)


def main() -> None:
    deterministic_trace_checks()
    dependency_checks()
    manual_gate_and_drain_checks()
    warmup_chunk_trace_identity_check()
    always_on_equivalence_check()
    print("bursty DemandGate semantics: PASS")


if __name__ == "__main__":
    main()
