"""Closed-loop request pipeline with one serialized MLX GPU execution worker."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from common import measure
from .metrics import percentile
from .phase_monitor import GPUPhase, PhaseMonitor


@dataclass
class GPUTicket:
    request_id: str
    client_id: int
    arrival: float
    retrieval: dict[str, object]
    ready_for_gpu: float
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, object] | None = None
    error: BaseException | None = None


class GPUWorker:
    def __init__(self, model: Any, context: int, output_tokens: int,
                 monitor: PhaseMonitor, prompt_seed: int = 0) -> None:
        if context < 1 or output_tokens < 2:
            raise ValueError("context must be positive and output_tokens >= 2")
        self.model, self.context, self.output_tokens = model, context, output_tokens
        self.prompt_seed = prompt_seed
        self.monitor = monitor
        self.queue: queue.Queue[GPUTicket | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="phaseguard-gpu", daemon=True)

    def start(self) -> None: self.thread.start()

    def submit(self, ticket: GPUTicket) -> None: self.queue.put(ticket)

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join(timeout=60)
        if self.thread.is_alive(): raise RuntimeError("GPU worker did not stop")

    def _run(self) -> None:
        while True:
            ticket = self.queue.get()
            if ticket is None:
                self.monitor.transition(GPUPhase.IDLE)
                return
            try: ticket.result = self._execute(ticket)
            except BaseException as exc: ticket.error = exc
            finally: ticket.done.set()

    def _execute(self, ticket: GPUTicket) -> dict[str, object]:
        rid = ticket.request_id
        prefill_start = self.monitor.transition(GPUPhase.PREFILL, rid)
        cache = measure.make_prompt_cache(self.model)
        # Preserve the exact shape while allowing held-out deterministic token traces.
        x = (measure.make_tokens(self.context) + (self.prompt_seed + ticket.client_id) % 100) % 100
        mx.eval(x)
        out = measure.forward_body(self.model, x, cache)
        mx.eval(out)
        for c in cache:
            state = c.state
            if state and state[0] is not None: mx.eval(state[0], state[1])
        prefill_end = time.perf_counter()
        decode_start = self.monitor.transition(GPUPhase.DECODE, rid)
        token = mx.array([[(7 + self.prompt_seed + ticket.client_id) % 100]], dtype=mx.int32)
        mx.eval(token)
        token_times: list[float] = []
        y = None
        for _ in range(self.output_tokens):
            y = self.model(token, cache=cache)
            mx.eval(y)
            token_times.append(self.monitor.record_token(rid))
        decode_end = time.perf_counter()
        self.monitor.transition(GPUPhase.IDLE)
        intervals_ms = [(b - a) * 1e3 for a, b in zip(token_times, token_times[1:])]
        if not intervals_ms: raise RuntimeError("decode produced no TPOT intervals")
        first_token = token_times[0]
        row: dict[str, object] = {
            "request_id": rid, "client_id": ticket.client_id, "arrival": ticket.arrival,
            "retrieval_submitted": ticket.retrieval["submitted"],
            "retrieval_start": ticket.retrieval["started"],
            "retrieval_end": ticket.retrieval["ended"],
            "retrieval_queue_ms": (float(ticket.retrieval["started"]) - float(ticket.retrieval["submitted"])) * 1e3,
            "retrieval_latency_ms": (float(ticket.retrieval["ended"]) - float(ticket.retrieval["started"])) * 1e3,
            "retrieval_queries": ticket.retrieval["queries"],
            "retrieval_worker": ticket.retrieval["worker_id"],
            "ready_for_gpu": ticket.ready_for_gpu,
            "gpu_queue_ms": (prefill_start - ticket.ready_for_gpu) * 1e3,
            "prefill_start": prefill_start, "prefill_end": prefill_end,
            "prefill_ms": (prefill_end - prefill_start) * 1e3,
            "decode_start": decode_start, "first_token": first_token,
            "ttft_ms": (first_token - prefill_start) * 1e3,
            "token_timestamps": token_times, "tpot_intervals_ms": intervals_ms,
            "mean_tpot_ms": sum(intervals_ms) / len(intervals_ms),
            "p50_tpot_ms": percentile(intervals_ms, 50),
            "p95_tpot_ms": percentile(intervals_ms, 95),
            "p99_tpot_ms": percentile(intervals_ms, 99),
            "decode_end": decode_end, "completion": decode_end,
            "end_to_end_ms": (decode_end - ticket.arrival) * 1e3,
        }
        del cache, x, out, token, y
        measure.free_buffers()
        return row


def run_closed_loop(model: Any, manager: Any, monitor: PhaseMonitor, concurrency: int,
                    requests_per_client: int, context: int, output_tokens: int,
                    retrieval_queries: int, retrieval_chunk: int, seed: int) -> list[dict[str, object]]:
    if concurrency < 1 or requests_per_client < 1:
        raise ValueError("concurrency and requests_per_client must be positive")
    gpu = GPUWorker(model, context, output_tokens, monitor)
    gpu.start()
    barrier = threading.Barrier(concurrency)
    rows: list[dict[str, object]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def client(client_id: int) -> None:
        try:
            barrier.wait()
            for sequence in range(requests_per_client):
                rid = f"c{client_id:02d}-r{sequence:03d}"
                arrival = time.perf_counter()
                task = manager.submit(rid, retrieval_queries, retrieval_chunk,
                                      seed + client_id * 100_000 + sequence)
                retrieval = manager.wait(task)
                ready = time.perf_counter()
                ticket = GPUTicket(rid, client_id, arrival, retrieval, ready)
                gpu.submit(ticket)
                if not ticket.done.wait(timeout=1800): raise TimeoutError(rid)
                if ticket.error: raise ticket.error
                assert ticket.result is not None
                with lock: rows.append(ticket.result)
        except BaseException as exc:
            with lock: errors.append(exc)

    threads = [threading.Thread(target=client, args=(i,), name=f"client-{i}")
               for i in range(concurrency)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    gpu.close()
    if errors: raise RuntimeError("pipeline client failure") from errors[0]
    return sorted(rows, key=lambda row: float(row["arrival"]))
