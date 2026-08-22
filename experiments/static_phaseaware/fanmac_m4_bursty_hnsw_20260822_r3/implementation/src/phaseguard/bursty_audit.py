"""Offline reconstruction and audit metrics for frozen bursty-demand blocks."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .demand_gate import FrozenDemandTrace


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def _runtime_segments(events: list[dict[str, Any]], start: float,
                      end: float) -> list[dict[str, Any]]:
    ordered = sorted(events, key=lambda item: (float(item["timestamp"]), int(item["seq"])))
    state = {"phase": "IDLE", "cap": 0, "demand_on": True, "active": 0}
    for event in ordered:
        if float(event["timestamp"]) > start:
            break
        state = {"phase": str(event.get("llm_phase", state["phase"])),
                 "cap": int(event.get("requested_cap", state["cap"])),
                 "demand_on": bool(event.get("demand_on", state["demand_on"])),
                 "active": int(event.get("active_query_count", state["active"]))}
    cursor = start
    segments: list[dict[str, Any]] = []
    for event in ordered:
        timestamp = float(event["timestamp"])
        if timestamp <= start:
            continue
        if timestamp >= end:
            break
        if timestamp > cursor:
            segments.append({"start": cursor, "end": timestamp, **state})
        state = {"phase": str(event.get("llm_phase", state["phase"])),
                 "cap": int(event.get("requested_cap", state["cap"])),
                 "demand_on": bool(event.get("demand_on", state["demand_on"])),
                 "active": int(event.get("active_query_count", state["active"]))}
        cursor = timestamp
    if cursor < end:
        segments.append({"start": cursor, "end": end, **state})
    return segments


def audit_bursty_block(*, events: list[dict[str, Any]], trace: FrozenDemandTrace,
                       offset_s: float, block_start: float, block_end: float,
                       llm_requests: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    duration = block_end - block_start
    if duration <= 0:
        raise ValueError("bursty block duration must be positive")
    scheduled = trace.scheduled_segments(offset_s, duration)
    runtime = _runtime_segments(events, block_start, block_end)
    scheduled_on_s = sum(end - start for start, end, state in scheduled if state)
    scheduled_on_fraction = scheduled_on_s / duration
    actual_on_s = sum(item["end"] - item["start"] for item in runtime
                      if item["demand_on"])
    active_s = sum(item["end"] - item["start"] for item in runtime
                   if int(item["active"]) > 0)
    active_worker_s = sum((item["end"] - item["start"]) * int(item["active"])
                          for item in runtime)
    prefill_active_worker_s = sum(
        (item["end"] - item["start"]) * int(item["active"])
        for item in runtime if item["phase"] == "PREFILL")
    decode_active_worker_s = sum(
        (item["end"] - item["start"]) * int(item["active"])
        for item in runtime if item["phase"] == "DECODE")

    overlap: dict[tuple[str, bool, int], dict[str, float]] = defaultdict(
        lambda: {"wall_time_s": 0.0, "active_worker_time_s": 0.0})
    for item in runtime:
        span = item["end"] - item["start"]
        key = (str(item["phase"]), bool(item["demand_on"]), int(item["cap"]))
        overlap[key]["wall_time_s"] += span
        overlap[key]["active_worker_time_s"] += span * int(item["active"])
    overlap_rows = [
        {"llm_phase": phase, "demand_on": demand_on, "policy_cap": cap,
         "wall_time_s": values["wall_time_s"],
         "active_worker_time_s": values["active_worker_time_s"]}
        for (phase, demand_on, cap), values in sorted(overlap.items())
    ]

    # Split runtime state at ideal scheduled transitions for the predeclared
    # cap-binding denominator rather than substituting controller wake-up time.
    scheduled_absolute = [
        (block_start + start, block_start + end, state)
        for start, end, state in scheduled
    ]
    cap_binding_s = 0.0
    for item in runtime:
        for start, end, scheduled_on in scheduled_absolute:
            overlap_start = max(item["start"], start)
            overlap_end = min(item["end"], end)
            if (scheduled_on and overlap_end > overlap_start and int(item["cap"]) > 0
                    and int(item["active"]) >= int(item["cap"])):
                cap_binding_s += overlap_end - overlap_start

    measured_events = [event for event in events
                       if block_start <= float(event["timestamp"]) <= block_end]
    starts = [event for event in measured_events
              if event["event_type"] == "query_chunk_started"]
    completions = [event for event in measured_events
                   if event["event_type"] == "query_chunk_completed"]
    completed_queries = sum(int(event.get("query_count", 0)) for event in completions)
    starts_during_off = [event for event in starts if not bool(event.get("demand_on", True))]
    demand_transitions = [event for event in measured_events
                          if event["event_type"] in ("demand_on", "demand_off")]

    actual_state_segments = [(item["start"], item["end"], bool(item["demand_on"]),
                              int(item["active"])) for item in runtime]
    demand_periods: list[list[float | bool]] = []
    for start, end, state, _ in actual_state_segments:
        if demand_periods and bool(demand_periods[-1][2]) == state:
            demand_periods[-1][1] = end
        else:
            demand_periods.append([start, end, state])
    on_durations = [float(end) - float(start) for start, end, state in demand_periods
                    if bool(state)]
    off_durations = [float(end) - float(start) for start, end, state in demand_periods
                     if not bool(state)]

    off_events = [event for event in demand_transitions if event["event_type"] == "demand_off"]
    drain_times: list[float] = []
    drain_windows: list[tuple[float, float]] = []
    for event in off_events:
        off_time = float(event["timestamp"])
        if int(event.get("active_query_count", 0)) == 0:
            zero_time = off_time
        else:
            zero_candidates = [
                float(candidate["timestamp"]) for candidate in measured_events
                if float(candidate["timestamp"]) >= off_time
                and candidate["event_type"] == "query_chunk_completed"
                and int(candidate.get("active_query_count", -1)) == 0
            ]
            zero_time = min(zero_candidates, default=block_end)
        drain_times.append(max(0.0, zero_time - off_time))
        drain_windows.append((off_time, zero_time))

    off_active_s = sum(end - start for start, end, state, active in actual_state_segments
                       if not state and active > 0)
    bounded_drain_active_s = 0.0
    for start, end, state, active in actual_state_segments:
        if state or active <= 0:
            continue
        for drain_start, drain_end in drain_windows:
            bounded_drain_active_s += max(0.0, min(end, drain_end) - max(start, drain_start))
    off_active_excluding_drain_s = max(0.0, off_active_s - bounded_drain_active_s)

    demand_sequences = [int(event["demand_sequence"]) for event in measured_events
                        if event["event_type"].startswith("demand_")
                        and "demand_sequence" in event]
    audit = {
        "demand_level": trace.demand_level,
        "trace_seed": trace.seed,
        "trace_offset_s": float(offset_s),
        "block_start": block_start,
        "block_end": block_end,
        "block_duration_s": duration,
        "scheduled_on_fraction": scheduled_on_fraction,
        "scheduled_on_time_s": scheduled_on_s,
        "actual_demand_on_fraction": actual_on_s / duration,
        "actual_hnsw_active_fraction": active_s / duration,
        "active_worker_time_s": active_worker_s,
        "prefill_active_worker_time_s": prefill_active_worker_s,
        "decode_active_worker_time_s": decode_active_worker_s,
        "demand_transition_count": len(demand_transitions),
        "demand_on_transition_count": sum(event["event_type"] == "demand_on"
                                           for event in demand_transitions),
        "demand_off_transition_count": len(off_events),
        "mean_on_duration_s": float(np.mean(on_durations)) if on_durations else 0.0,
        "p95_on_duration_s": _percentile(on_durations, 95),
        "mean_off_duration_s": float(np.mean(off_durations)) if off_durations else 0.0,
        "p95_off_duration_s": _percentile(off_durations, 95),
        "mean_on_off_to_zero_active_s": float(np.mean(drain_times)) if drain_times else 0.0,
        "p95_on_off_to_zero_active_s": _percentile(drain_times, 95),
        "max_on_off_to_zero_active_s": max(drain_times, default=0.0),
        "off_active_time_s": off_active_s,
        "off_active_excluding_bounded_drain_s": off_active_excluding_drain_s,
        "new_chunk_starts_during_off": len(starts_during_off),
        "completed_retrieval_queries": completed_queries,
        "retrieval_qps_full_wall_time": completed_queries / duration,
        "retrieval_qps_scheduled_on_time": (
            completed_queries / scheduled_on_s if scheduled_on_s > 0 else 0.0),
        "completed_queries_per_llm_request": completed_queries / llm_requests,
        "cap_binding_fraction_during_scheduled_on": (
            cap_binding_s / scheduled_on_s if scheduled_on_s > 0 else 0.0),
        "demand_sequence_strict": all(b > a for a, b in zip(
            demand_sequences, demand_sequences[1:])),
    }
    return audit, overlap_rows
