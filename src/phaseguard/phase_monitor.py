"""Thread-safe GPU phase and token-timestamp instrumentation."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable


class GPUPhase(str, Enum):
    IDLE = "IDLE"
    PREFILL = "PREFILL"
    DECODE = "DECODE"


@dataclass(frozen=True)
class PhaseEvent:
    timestamp: float
    request_id: str | None
    phase: str
    event: str
    permitted_workers: int | None = None


class PhaseMonitor:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._phase = GPUPhase.IDLE
        self._request_id: str | None = None
        self._phase_started = time.perf_counter()
        self._events: list[PhaseEvent] = [
            PhaseEvent(self._phase_started, None, GPUPhase.IDLE.value, "start")
        ]
        self._token_times: dict[str, list[float]] = {}
        self._listener: Callable[[GPUPhase, str | None], int | None] | None = None

    def set_listener(self, listener: Callable[[GPUPhase, str | None], int | None]) -> None:
        with self._lock:
            self._listener = listener

    def transition(self, phase: GPUPhase, request_id: str | None = None) -> float:
        now = time.perf_counter()
        with self._lock:
            if self._phase != phase or self._request_id != request_id:
                self._events.append(PhaseEvent(now, self._request_id, self._phase.value, "end"))
                self._phase, self._request_id, self._phase_started = phase, request_id, now
                permit = self._listener(phase, request_id) if self._listener else None
                self._events.append(PhaseEvent(now, request_id, phase.value, "start", permit))
        return now

    def record_token(self, request_id: str) -> float:
        now = time.perf_counter()
        with self._lock:
            self._token_times.setdefault(request_id, []).append(now)
        return now

    def snapshot(self) -> tuple[GPUPhase, str | None, float]:
        with self._lock:
            return self._phase, self._request_id, self._phase_started

    def token_times(self, request_id: str) -> list[float]:
        with self._lock:
            return list(self._token_times.get(request_id, ()))

    def events(self) -> list[dict[str, object]]:
        with self._lock:
            return [asdict(e) for e in self._events]
