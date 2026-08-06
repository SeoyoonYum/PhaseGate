"""Metrics and selection helpers for static phase-aware experiments."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def phase_counter_delta(samples: list[dict[str, Any]], phase: str, key: str) -> int:
    """Attribute monotonic counter progress only to intervals wholly in one phase."""
    total = 0
    for before, after in zip(samples, samples[1:]):
        if before["phase"] != phase or after["phase"] != phase:
            continue
        delta = int(after[key]) - int(before[key])
        if delta >= 0:
            total += delta
    return total


def phase_duration_s(samples: list[dict[str, Any]], phase: str) -> float:
    total = 0.0
    for before, after in zip(samples, samples[1:]):
        if before["phase"] == phase and after["phase"] == phase:
            total += max(0.0, float(after["timestamp"]) - float(before["timestamp"]))
    return total


def phase_cap_applied_fraction(samples: list[dict[str, Any]], phase: str,
                               cap: int) -> float:
    """Wall-time fraction at a requested cap, never event-count weighted."""
    total = applied = 0.0
    for before, after in zip(samples, samples[1:]):
        # State at `before` is valid until the next event. Requiring `after` to
        # have the same phase incorrectly drops the entire phase for Fixed-0,
        # which intentionally generates no retrieval events inside the phase.
        if before["phase"] != phase:
            continue
        duration = max(0.0, float(after["timestamp"]) - float(before["timestamp"]))
        total += duration
        if int(before["permitted_workers"]) == cap:
            applied += duration
    return applied / total if total > 0 else 0.0


def phase_transition_metrics(samples: list[dict[str, Any]], decode_cap: int) -> dict[str, float]:
    """Measure non-preemptive worker overshoot after each decode admission change."""
    decode = [row for row in samples if row["phase"] == "DECODE"]
    if not decode:
        raise ValueError("decode samples are required")
    groups: list[list[dict[str, Any]]] = []
    for row in decode:
        if not groups or (
            row.get("request_id"), row.get("phase_started")
        ) != (
            groups[-1][-1].get("request_id"), groups[-1][-1].get("phase_started")
        ):
            groups.append([row])
        else:
            groups[-1].append(row)

    convergence_ms: list[float] = []
    for group in groups:
        active = [int(row["active_retrievals"]) for row in group]
        start = float(group[0].get("phase_started", group[0]["timestamp"]))
        stable_index = None
        for index in range(len(group)):
            if all(value <= decode_cap for value in active[index:]):
                stable_index = index
                break
        end_index = stable_index if stable_index is not None else len(group) - 1
        convergence_ms.append(max(0.0, (float(group[end_index]["timestamp"]) - start) * 1e3))

    active_array = np.asarray([int(row["active_retrievals"]) for row in decode], dtype=float)
    excess = np.maximum(active_array - decode_cap, 0.0)
    overshoot = excess > 0
    return {
        "phase_transition_to_cap_ms": float(np.median(convergence_ms)),
        "phase_transition_to_cap_max_ms": float(max(convergence_ms)),
        "decode_cap_overshoot_fraction": float(overshoot.mean()),
        "decode_cap_overshoot_worker_mean": float(excess[overshoot].mean()) if overshoot.any() else 0.0,
        "decode_cap_overshoot_worker_max": float(excess.max()),
        "decode_cap_applied_fraction": phase_cap_applied_fraction(
            samples, "DECODE", decode_cap),
    }


def latency_drift_ratio(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size < 2:
        return 1.0
    width = max(1, array.size // 3)
    first = float(np.median(array[:width]))
    last = float(np.median(array[-width:]))
    return last / first if first else float("inf")


def linear_slope(times: Iterable[float], values: Iterable[float]) -> float:
    """Least-squares value trend per second; returns zero for fewer than two samples."""
    x = np.asarray(list(times), dtype=float)
    y = np.asarray(list(values), dtype=float)
    if x.size < 2 or y.size != x.size:
        return 0.0
    return float(np.polyfit(x - x[0], y, 1)[0])


def counter_rate_drift(samples: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Compare first/last-third progress rates and report their linear drift."""
    if len(samples) < 4:
        return {"first_qps": 0.0, "last_qps": 0.0, "ratio": 1.0,
                "slope_qps_per_s": 0.0}
    width = max(2, len(samples) // 3)

    def rate(window: list[dict[str, Any]]) -> float:
        elapsed = float(window[-1]["timestamp"]) - float(window[0]["timestamp"])
        progress = int(window[-1][key]) - int(window[0][key])
        return float(progress / elapsed) if elapsed > 0 and progress >= 0 else 0.0

    first = rate(samples[:width])
    last = rate(samples[-width:])
    duration = float(samples[-1]["timestamp"]) - float(samples[0]["timestamp"])
    ratio = last / first if first > 0 else (1.0 if last == 0 else float("inf"))
    return {"first_qps": first, "last_qps": last, "ratio": ratio,
            "slope_qps_per_s": (last - first) / duration if duration > 0 else 0.0}
