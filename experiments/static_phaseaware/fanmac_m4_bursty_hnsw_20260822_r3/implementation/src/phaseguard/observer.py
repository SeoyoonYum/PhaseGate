"""Low-overhead event and memory observers for measured PhaseGate blocks."""
from __future__ import annotations

import bisect
import resource
import threading
import time
from typing import Any, Iterable


class EventBuffer:
    """Thread-safe buffer for state-change events outside the token fast path."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []

    def emit(self, event_type: str, timestamp: float | None = None, **fields: Any) -> None:
        event = {"timestamp": time.perf_counter() if timestamp is None else float(timestamp),
                 "event_type": event_type, **fields}
        with self._lock:
            self._events.append(event)

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]


class NativeMemoryMonitor:
    """At-most-1 Hz in-process peak-RSS sampling without fork/exec."""

    def __init__(self, interval_s: float = 1.0) -> None:
        if interval_s < 1.0:
            raise ValueError("production memory sampling interval must be at least one second")
        self.interval_s = float(interval_s)
        self.samples: list[dict[str, float | int]] = []
        self.missed_intervals = 0
        self.max_scheduling_delay_s = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _peak_rss_bytes() -> int:
        # Darwin reports bytes; Linux reports KiB. This campaign is Darwin-only,
        # but retain a portable fallback for functional tests.
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        import sys
        return value if sys.platform == "darwin" else value * 1024

    def start(self) -> "NativeMemoryMonitor":
        if self._thread is not None:
            raise RuntimeError("memory monitor already started")
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="phasegate-memory-observer")
        self._thread.start()
        return self

    def _run(self) -> None:
        target = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            delay = max(0.0, now - target)
            missed = int(delay // self.interval_s)
            self.missed_intervals += missed
            self.max_scheduling_delay_s = max(self.max_scheduling_delay_s, delay)
            self.samples.append({"timestamp": now, "peak_rss_bytes": self._peak_rss_bytes(),
                                 "scheduling_delay_s": delay})
            target += (missed + 1) * self.interval_s
            if self._stop.wait(max(0.0, target - time.perf_counter())):
                break

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("memory monitor did not stop")

    def audit(self) -> dict[str, Any]:
        timestamps = [float(sample["timestamp"]) for sample in self.samples]
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        return {
            "sample_count": len(self.samples),
            "interval_s": self.interval_s,
            "observed_rate_hz": ((len(timestamps) - 1) / duration if duration > 0 else 0.0),
            "missed_intervals": self.missed_intervals,
            "max_scheduling_delay_ms": self.max_scheduling_delay_s * 1e3,
            "subprocess_count": 0,
            "api": "resource.getrusage(RUSAGE_SELF)",
        }


def _phase_starts(phase_events: Iterable[dict[str, Any]]) -> list[tuple[float, str, str | None]]:
    starts = []
    for event in phase_events:
        if event.get("event") == "start":
            starts.append((float(event["timestamp"]), str(event["phase"]),
                           event.get("request_id")))
    return sorted(starts)


def finalize_event_log(*, run_id: str, run_key: str, repeat: int, policy: str,
                       phase_events: list[dict[str, Any]], requests: list[dict[str, Any]],
                       runtime_events: list[dict[str, Any]], default_cap: int) -> list[dict[str, Any]]:
    """Merge buffered state changes and request-local timestamps after measurement."""
    merged = [dict(event) for event in runtime_events]
    phase_starts = _phase_starts(phase_events)
    for timestamp, phase, request_id in phase_starts:
        event_type = {"PREFILL": "phase_enter_prefill", "DECODE": "phase_enter_decode",
                      "IDLE": "phase_exit_or_idle"}[phase]
        merged.append({"timestamp": timestamp, "event_type": event_type,
                       "request_id": request_id})
    for request in requests:
        rid = str(request["request_id"])
        merged.extend([
            {"timestamp": float(request["prefill_start"]), "event_type": "request_start",
             "request_id": rid},
            {"timestamp": float(request["prefill_end"]), "event_type": "prefill_complete",
             "request_id": rid},
            {"timestamp": float(request["decode_end"]), "event_type": "request_complete",
             "request_id": rid},
        ])
        # Timestamps were already acquired and buffered on the unchanged token path.
        merged.extend({"timestamp": float(timestamp), "event_type": "token_ready",
                       "request_id": rid, "token_index": index}
                      for index, timestamp in enumerate(request["token_timestamps"]))

    priority = {
        "demand_trace_start": 0, "demand_on": 1, "demand_off": 1,
        "query_admitted": 2, "query_started": 3, "query_chunk_started": 3,
        "query_completed": 4, "query_chunk_completed": 4,
        "demand_trace_end": 9,
    }
    merged.sort(key=lambda event: (float(event["timestamp"]),
                                   priority.get(str(event.get("event_type", "")), 5),
                                   str(event.get("event_type", "")),
                                   str(event.get("query_id", ""))))
    phase_times = [item[0] for item in phase_starts]
    cap = int(default_cap)
    active = 0
    demand_on = True
    # Demand-enabled managers emit chunk events during the fixed warm-up that
    # precedes measured trace replay. Pre-seed immutable trace identity from
    # the later start marker so those buffered chunk events are fully auditable.
    trace_marker = next((event for event in merged
                         if event.get("event_type") == "demand_trace_start"), {})
    demand_level = trace_marker.get("demand_level")
    trace_seed = trace_marker.get("trace_seed")
    trace_offset_s = trace_marker.get("trace_offset_s")
    output: list[dict[str, Any]] = []
    for sequence, event in enumerate(merged, 1):
        if event.get("event_type") == "cap_change":
            cap = int(event["requested_cap"])
        if event.get("event_type") in ("demand_on", "demand_off"):
            demand_on = bool(event["demand_on"])
        if event.get("event_type", "").startswith("demand_"):
            demand_level = event.get("demand_level", demand_level)
            trace_seed = event.get("trace_seed", trace_seed)
            trace_offset_s = event.get("trace_offset_s", trace_offset_s)
        if "active_query_count" in event:
            active = int(event["active_query_count"])
        timestamp = float(event["timestamp"])
        index = bisect.bisect_right(phase_times, timestamp) - 1
        phase = phase_starts[index][1] if index >= 0 else "IDLE"
        item = {
            "seq": sequence, "timestamp": timestamp, "run_id": run_id,
            "run_key": run_key, "repeat": repeat, "policy": policy,
            "request_id": event.get("request_id"), "query_id": event.get("query_id"),
            "event_type": event["event_type"], "requested_cap": int(event.get("requested_cap", cap)),
            "active_query_count": active,
            "demand_on": bool(event.get("demand_on", demand_on)),
            "demand_level": event.get("demand_level", demand_level),
            "trace_seed": event.get("trace_seed", trace_seed),
            "trace_offset_s": event.get("trace_offset_s", trace_offset_s),
            "llm_phase": phase,
        }
        item.update({key: value for key, value in event.items() if key not in item})
        output.append(item)
    return output


def reconstruct_event_log(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate ordering/accounting and summarize an event-driven trace."""
    sequences = [int(event["seq"]) for event in events]
    timestamps = [float(event["timestamp"]) for event in events]
    active_values = [int(event.get("active_query_count", 0)) for event in events
                     if event["event_type"] in (
                         "query_admitted", "query_started", "query_completed",
                         "query_chunk_started", "query_chunk_completed", "cap_change")]
    admitted = sum(int(event.get("query_count", 0)) for event in events
                   if event["event_type"] == "query_admitted")
    started = sum(int(event.get("query_count", 0)) for event in events
                  if event["event_type"] in ("query_started", "query_chunk_started"))
    completed = sum(int(event.get("query_count", 0)) for event in events
                    if event["event_type"] in ("query_completed", "query_chunk_completed"))
    tokens = sum(event["event_type"] == "token_ready" for event in events)
    admissions_within_cap = all(
        int(event.get("active_query_count", 0)) <= int(event.get("requested_cap", 0))
        for event in events if event["event_type"] == "query_admitted"
    )
    admissions_within_demand = all(
        bool(event.get("demand_on", True))
        for event in events if event["event_type"] == "query_admitted"
    )
    return {
        "sequence_exact": sequences == list(range(1, len(events) + 1)),
        "timestamps_monotonic": all(b >= a for a, b in zip(timestamps, timestamps[1:])),
        "active_never_negative": all(value >= 0 for value in active_values),
        "admitted_queries": admitted, "started_queries": started,
        "completed_queries": completed, "token_events": tokens,
        "query_accounting_exact": admitted == started == completed,
        "admissions_within_cap": admissions_within_cap,
        "admissions_within_demand": admissions_within_demand,
        "max_active_queries": max(active_values, default=0),
        "event_count": len(events),
    }


def event_state_samples(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert exact state-change events to the legacy analysis sample schema."""
    relevant = {
        "phase_enter_prefill", "phase_enter_decode", "phase_exit_or_idle", "cap_change",
        "query_admitted", "query_completed", "query_chunk_completed",
        "retrieval_task_submitted", "retrieval_task_completed",
        "demand_trace_start", "demand_on", "demand_off", "demand_trace_end",
    }
    admitted = completed = submitted_tasks = completed_tasks = 0
    outstanding = active = 0
    phase = "IDLE"
    demand_on = True
    phase_started = float(events[0]["timestamp"]) if events else 0.0
    cap = int(events[0].get("requested_cap", 0)) if events else 0
    output: list[dict[str, Any]] = []
    for event in events:
        kind = str(event["event_type"])
        if kind not in relevant:
            continue
        if kind == "phase_enter_prefill":
            phase, phase_started = "PREFILL", float(event["timestamp"])
        elif kind == "phase_enter_decode":
            phase, phase_started = "DECODE", float(event["timestamp"])
        elif kind == "phase_exit_or_idle":
            phase, phase_started = "IDLE", float(event["timestamp"])
        elif kind == "cap_change":
            cap = int(event["requested_cap"])
        elif kind in ("demand_on", "demand_off"):
            demand_on = bool(event["demand_on"])
        elif kind == "query_admitted":
            admitted += int(event.get("query_count", 0))
        elif kind in ("query_completed", "query_chunk_completed"):
            completed += int(event.get("query_count", 0))
        elif kind == "retrieval_task_submitted":
            submitted_tasks += 1; outstanding += 1
        elif kind == "retrieval_task_completed":
            completed_tasks += 1; outstanding = max(0, outstanding - 1)
        active = int(event.get("active_query_count", active))
        output.append({
            "timestamp": float(event["timestamp"]), "phase": phase,
            "phase_started": phase_started, "request_id": event.get("request_id"),
            "demand_on": demand_on,
            "permitted_workers": cap, "active_retrievals": active,
            "inflight_tasks": active, "retrieval_queue_depth": max(0, outstanding - active),
            "paused_inflight_tasks": max(0, outstanding - active),
            "effective_backlog_tasks": max(0, outstanding - active),
            "outstanding_tasks": outstanding, "submitted_tasks": submitted_tasks,
            "completed_tasks": completed_tasks, "admitted_queries": admitted,
            "completed_queries": completed, "rss_bytes": 0,
        })
    return output
