"""Staggered-arrival pipeline that creates retrieval backlog during GPU decode."""
from __future__ import annotations

import threading
import time
from typing import Any

from .phase_monitor import GPUPhase, PhaseMonitor
from .request_pipeline import GPUTicket, GPUWorker


def run_staggered_backlog(
    model: Any,
    manager: Any,
    monitor: PhaseMonitor,
    total_requests: int,
    context: int,
    output_tokens: int,
    retrieval_queries: int,
    retrieval_chunk: int,
    burst_size: int,
    burst_spacing_s: float,
    interarrival_s: float,
    seed: int,
) -> list[dict[str, object]]:
    """Start request A alone, then inject the remaining trace during A's decode."""
    if total_requests < 2:
        raise ValueError("total_requests must be at least two")
    if not 1 <= burst_size < total_requests:
        raise ValueError("burst_size must be in [1, total_requests)")
    if burst_spacing_s < 0 or interarrival_s <= 0:
        raise ValueError("arrival spacing must be non-negative and interarrival positive")

    gpu = GPUWorker(model, context, output_tokens, monitor)
    gpu.start()
    rows: list[dict[str, object]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()
    threads: list[threading.Thread] = []

    def request(sequence: int, scheduled: float) -> None:
        try:
            now = time.perf_counter()
            if scheduled > now:
                time.sleep(scheduled - now)
            arrival = time.perf_counter()
            rid = f"b{sequence:03d}"
            task = manager.submit(rid, retrieval_queries, retrieval_chunk, seed + sequence)
            retrieval = manager.wait(task, timeout=3600)
            ready = time.perf_counter()
            ticket = GPUTicket(rid, sequence, arrival, retrieval, ready)
            gpu.submit(ticket)
            if not ticket.done.wait(timeout=3600):
                raise TimeoutError(rid)
            if ticket.error:
                raise ticket.error
            assert ticket.result is not None
            ticket.result["scheduled_arrival"] = scheduled
            ticket.result["arrival_lateness_ms"] = (arrival - scheduled) * 1e3
            with lock:
                rows.append(ticket.result)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    first = threading.Thread(target=request, args=(0, time.perf_counter()), name="arrival-000")
    threads.append(first)
    first.start()

    deadline = time.monotonic() + 1800
    first_decode_start: float | None = None
    while time.monotonic() < deadline:
        phase, request_id, started = monitor.snapshot()
        if phase == GPUPhase.DECODE and request_id == "b000":
            first_decode_start = started
            break
        if errors:
            break
        time.sleep(0.001)
    if first_decode_start is None:
        first.join(timeout=5)
        gpu.close()
        if errors:
            raise RuntimeError("initial request failed") from errors[0]
        raise TimeoutError("initial request never entered decode")

    # Replenish future work at every decode boundary. A one-shot burst drains too
    # early and recreates the original natural-staggering artifact.
    next_sequence = 1
    seen_decodes: set[str] = set()
    while next_sequence < total_requests and not errors:
        phase, request_id, phase_started = monitor.snapshot()
        if phase == GPUPhase.DECODE and request_id is not None and request_id not in seen_decodes:
            seen_decodes.add(request_id)
            count = min(burst_size, total_requests - next_sequence)
            for offset_index in range(count):
                sequence = next_sequence
                next_sequence += 1
                scheduled = phase_started + offset_index * burst_spacing_s
                thread = threading.Thread(target=request, args=(sequence, scheduled),
                                          name=f"arrival-{sequence:03d}")
                threads.append(thread)
                thread.start()
        time.sleep(min(interarrival_s, 0.002))

    for thread in threads:
        thread.join()
    gpu.close()
    if errors:
        raise RuntimeError("staggered pipeline request failure") from errors[0]
    return sorted(rows, key=lambda row: float(row["arrival"]))
