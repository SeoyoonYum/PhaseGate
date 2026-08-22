"""Frozen, phase-blind external-demand traces for bursty HNSW experiments."""
from __future__ import annotations

import bisect
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DemandInterval:
    start_s: float
    end_s: float
    demand_on: bool

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class FrozenDemandTrace:
    demand_level: float
    seed: int
    duration_s: float
    intervals: tuple[DemandInterval, ...]
    generator: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "FrozenDemandTrace":
        payload = json.loads(Path(path).read_text())
        intervals = tuple(DemandInterval(
            float(item["start_s"]), float(item["end_s"]), bool(item["demand_on"]))
            for item in payload["intervals"])
        trace = cls(float(payload["demand_level"]), int(payload["seed"]),
                    float(payload["duration_s"]), intervals,
                    dict(payload.get("generator", {})))
        trace.validate()
        return trace

    def validate(self) -> None:
        if self.duration_s <= 0 or not self.intervals:
            raise ValueError("demand trace must have positive duration and intervals")
        if abs(self.intervals[0].start_s) > 1e-9:
            raise ValueError("demand trace must start at zero")
        if abs(self.intervals[-1].end_s - self.duration_s) > 1e-9:
            raise ValueError("demand trace must cover its declared duration")
        previous: DemandInterval | None = None
        for interval in self.intervals:
            if interval.duration_s <= 0:
                raise ValueError("demand intervals must be positive")
            if previous is not None:
                if abs(previous.end_s - interval.start_s) > 1e-9:
                    raise ValueError("demand intervals must be contiguous")
                if previous.demand_on == interval.demand_on:
                    raise ValueError("demand intervals must alternate")
            previous = interval

    @property
    def starts(self) -> tuple[float, ...]:
        return tuple(interval.start_s for interval in self.intervals)

    def position(self, trace_time_s: float) -> tuple[int, float]:
        wrapped = float(trace_time_s) % self.duration_s
        index = bisect.bisect_right(self.starts, wrapped) - 1
        index = max(0, index)
        remaining = self.intervals[index].end_s - wrapped
        if remaining <= 1e-12:
            index = (index + 1) % len(self.intervals)
            remaining = self.intervals[index].duration_s
        return index, remaining

    def state_at(self, trace_time_s: float) -> bool:
        index, _ = self.position(trace_time_s)
        return self.intervals[index].demand_on

    def scheduled_on_fraction(self, offset_s: float, duration_s: float) -> float:
        if duration_s <= 0:
            return 0.0
        index, remaining = self.position(offset_s)
        left = float(duration_s)
        on_time = 0.0
        while left > 1e-12:
            span = min(left, remaining)
            if self.intervals[index].demand_on:
                on_time += span
            left -= span
            if left <= 1e-12:
                break
            index = (index + 1) % len(self.intervals)
            remaining = self.intervals[index].duration_s
        return on_time / duration_s

    def scheduled_segments(self, offset_s: float,
                           duration_s: float) -> list[tuple[float, float, bool]]:
        """Return clipped schedule segments relative to a replay start at zero."""
        if duration_s <= 0:
            return []
        index, remaining = self.position(offset_s)
        elapsed = 0.0
        output: list[tuple[float, float, bool]] = []
        while elapsed < duration_s - 1e-12:
            span = min(duration_s - elapsed, remaining)
            output.append((elapsed, elapsed + span, self.intervals[index].demand_on))
            elapsed += span
            if elapsed >= duration_s - 1e-12:
                break
            index = (index + 1) % len(self.intervals)
            remaining = self.intervals[index].duration_s
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "demand_level": self.demand_level,
            "seed": self.seed,
            "duration_s": self.duration_s,
            "generator": self.generator,
            "intervals": [
                {"start_s": item.start_s, "end_s": item.end_s,
                 "demand_on": item.demand_on}
                for item in self.intervals
            ],
        }

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def generate_demand_trace(*, demand_level: float, seed: int,
                          duration_s: float = 7200.0) -> FrozenDemandTrace:
    """Generate the handoff's predeclared alternating ON/OFF process."""
    if demand_level not in (0.05, 0.25, 1.0):
        raise ValueError("demand level must be 0.05, 0.25, or 1.0")
    if duration_s < 7200.0:
        raise ValueError("frozen demand traces must cover at least two hours")
    if demand_level == 1.0:
        return FrozenDemandTrace(
            demand_level, int(seed), float(duration_s),
            (DemandInterval(0.0, float(duration_s), True),),
            {"kind": "continuous", "minimum_interval_s": 0.25},
        )
    off_scale = 18.875 if demand_level == 0.05 else 2.875
    generator = {
        "kind": "alternating_shifted_gamma",
        "minimum_interval_s": 0.25,
        "on_gamma_shape": 2.0,
        "on_gamma_scale": 0.875,
        "off_gamma_shape": 2.0,
        "off_gamma_scale": off_scale,
    }
    rng = np.random.default_rng(seed)
    intervals: list[DemandInterval] = []
    start = 0.0
    demand_on = True
    # Preserve alternation when the frozen trace repeats: retain complete
    # ON/OFF pairs and allow the realized trace length to exceed the minimum.
    while start < duration_s or not demand_on:
        scale = 0.875 if demand_on else off_scale
        sampled = 0.25 + float(rng.gamma(shape=2.0, scale=scale))
        end = start + sampled
        intervals.append(DemandInterval(start, end, demand_on))
        start = end
        demand_on = not demand_on
    trace = FrozenDemandTrace(float(demand_level), int(seed), float(start),
                              tuple(intervals), generator)
    trace.validate()
    return trace


class DemandGateController:
    """Replay a frozen trace using monotonic time and no policy or phase input."""

    def __init__(self, manager: object, trace: FrozenDemandTrace, offset_s: float,
                 *, run_id: str) -> None:
        self.manager = manager
        self.trace = trace
        self.offset_s = float(offset_s) % trace.duration_s
        self.run_id = str(run_id)
        self.phase_state_consulted = False
        self.policy_identity_consulted = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._demand_sequence = 0
        self._transition_count = 0
        self._max_scheduling_delay_s = 0.0

    def _fields(self, demand_on: bool) -> dict[str, Any]:
        self._demand_sequence += 1
        return {
            "run_id": self.run_id,
            "demand_sequence": self._demand_sequence,
            "demand_level": self.trace.demand_level,
            "trace_seed": self.trace.seed,
            "trace_offset_s": self.offset_s,
            "demand_on": bool(demand_on),
        }

    def start(self) -> float:
        if self._thread is not None or self._started_at is not None:
            raise RuntimeError("DemandGate already started")
        index, remaining = self.trace.position(self.offset_s)
        state = self.trace.intervals[index].demand_on
        events = [("demand_trace_start", self._fields(state)),
                  ("demand_on" if state else "demand_off", self._fields(state))]
        started = self.manager.set_demand_state(state, events=events)
        self._started_at = started
        if len(self.trace.intervals) > 1:
            self._thread = threading.Thread(
                target=self._run, args=(index, remaining), daemon=True,
                name="phasegate-demand-trace")
            self._thread.start()
        return started

    def _run(self, index: int, remaining_s: float) -> None:
        assert self._started_at is not None
        deadline = self._started_at + remaining_s
        while not self._stop.is_set():
            if self._stop.wait(max(0.0, deadline - time.perf_counter())):
                return
            now = time.perf_counter()
            self._max_scheduling_delay_s = max(
                self._max_scheduling_delay_s, max(0.0, now - deadline))
            index = (index + 1) % len(self.trace.intervals)
            state = self.trace.intervals[index].demand_on
            event_type = "demand_on" if state else "demand_off"
            self.manager.set_demand_state(state, events=[(event_type, self._fields(state))])
            self._transition_count += 1
            deadline += self.trace.intervals[index].duration_s

    def stop(self) -> float:
        if self._started_at is None:
            raise RuntimeError("DemandGate not started")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("DemandGate controller did not stop")
        return self.manager.set_demand_state(
            False, events=[("demand_trace_end", self._fields(False))])

    def audit(self) -> dict[str, Any]:
        return {
            "phase_state_consulted": self.phase_state_consulted,
            "policy_identity_consulted": self.policy_identity_consulted,
            "selection_inputs": "frozen trace, offset, and time.perf_counter only",
            "demand_level": self.trace.demand_level,
            "trace_seed": self.trace.seed,
            "trace_offset_s": self.offset_s,
            "transition_count": self._transition_count,
            "max_scheduling_delay_ms": self._max_scheduling_delay_s * 1e3,
        }
