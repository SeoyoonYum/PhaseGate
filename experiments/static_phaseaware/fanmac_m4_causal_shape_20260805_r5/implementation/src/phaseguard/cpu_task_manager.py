"""Persistent process pool with a shared, phase-controlled worker permit."""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RetrievalResult:
    task_id: str
    request_id: str
    worker_id: int
    submitted: float
    started: float
    ended: float
    queries: int
    chunks: int
    checksum: float
    error: str | None = None


def _worker_main(worker_id: int, index_path: str, ef_search: int, top_k: int,
                 permit: Any, active_flags: Any, inflight: Any, admitted_queries: Any,
                 completed_queries: Any, tasks: Any, results: Any, ready: Any, stop: Any) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    try:
        from .retrieval import load_index, make_queries, search_chunks
        index, meta = load_index(index_path, ef_search)
        dimensions = int(meta.get("dimensions", index.d))
        ready.put((worker_id, None))
    except Exception as exc:
        ready.put((worker_id, repr(exc)))
        return
    while not stop.is_set():
        if worker_id >= permit.value:
            time.sleep(0.001)
            continue
        try:
            task = tasks.get(timeout=0.05)
        except queue.Empty:
            continue
        if task is None:
            return
        task_id, request_id, submitted, n_queries, chunk, seed = task
        started, done, chunks, checksum = time.perf_counter(), 0, 0, 0.0
        error = None
        with inflight.get_lock(): inflight.value += 1
        try:
            active_flags[worker_id] = 1
            queries = make_queries(seed, n_queries, dimensions)
            iterator = search_chunks(index, queries, top_k, chunk)
            active_flags[worker_id] = 0
            while True:
                while worker_id >= permit.value and not stop.is_set():
                    active_flags[worker_id] = 0
                    time.sleep(0.001)
                if stop.is_set():
                    raise RuntimeError("worker stopped")
                active_flags[worker_id] = 1
                expected = min(chunk, n_queries - done)
                with admitted_queries.get_lock():
                    admitted_queries.value += expected
                try:
                    count, value = next(iterator)
                except StopIteration:
                    break
                finally:
                    active_flags[worker_id] = 0
                done += count
                with completed_queries.get_lock():
                    completed_queries.value += count
                chunks += 1
                checksum += value
        except Exception as exc:
            error = repr(exc)
        finally:
            active_flags[worker_id] = 0
            with inflight.get_lock(): inflight.value -= 1
        results.put(asdict(RetrievalResult(task_id, request_id, worker_id, submitted,
                                           started, time.perf_counter(), done, chunks,
                                           checksum, error)))


class CPUTaskManager:
    def __init__(self, index_path: str, max_workers: int, ef_search: int = 64,
                 top_k: int = 10, start_method: str = "spawn") -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers
        self._ctx = mp.get_context(start_method)
        self._permit = self._ctx.Value("i", max_workers, lock=True)
        self._active_flags = self._ctx.Array("b", max_workers, lock=True)
        self._inflight = self._ctx.Value("i", 0, lock=True)
        self._admitted_queries = self._ctx.Value("q", 0, lock=True)
        self._completed_queries = self._ctx.Value("q", 0, lock=True)
        self._tasks, self._results, self._ready = (self._ctx.Queue() for _ in range(3))
        self._stop = self._ctx.Event()
        self._processes = [self._ctx.Process(target=_worker_main,
                           args=(i, index_path, ef_search, top_k, self._permit,
                                 self._active_flags, self._inflight, self._admitted_queries,
                                 self._completed_queries, self._tasks,
                                 self._results, self._ready, self._stop),
                           name=f"phaseguard-cpu-{i}") for i in range(max_workers)]
        self._condition = threading.Condition()
        self._completed: dict[str, dict[str, object]] = {}
        self._submitted = 0
        self._received = 0
        self._dispatcher: threading.Thread | None = None

    def start(self, timeout: float = 120.0) -> "CPUTaskManager":
        for p in self._processes: p.start()
        failures = []
        for _ in self._processes:
            worker, error = self._ready.get(timeout=timeout)
            if error: failures.append(f"worker {worker}: {error}")
        if failures:
            self.close()
            raise RuntimeError("; ".join(failures))
        self._dispatcher = threading.Thread(target=self._dispatch, daemon=True)
        self._dispatcher.start()
        return self

    def _dispatch(self) -> None:
        while not self._stop.is_set():
            try: result = self._results.get(timeout=0.05)
            except queue.Empty: continue
            with self._condition:
                self._completed[str(result["task_id"])] = result
                self._received += 1
                self._condition.notify_all()

    @property
    def permits(self) -> int:
        return int(self._permit.value)

    def set_permits(self, count: int) -> None:
        if not 0 <= count <= self.max_workers:
            raise ValueError("permit count outside [0, max_workers]")
        with self._permit.get_lock(): self._permit.value = count

    def submit(self, request_id: str, queries: int, chunk: int = 1,
               seed: int = 20260728) -> str:
        if queries < 1 or chunk < 1:
            raise ValueError("queries and chunk must be positive")
        task_id, now = uuid.uuid4().hex, time.perf_counter()
        self._tasks.put((task_id, request_id, now, queries, chunk, seed))
        with self._condition: self._submitted += 1
        return task_id

    def wait(self, task_id: str, timeout: float = 600.0) -> dict[str, object]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while task_id not in self._completed:
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise TimeoutError(task_id)
                self._condition.wait(min(remaining, 0.5))
            result = self._completed.pop(task_id)
        if result.get("error"): raise RuntimeError(str(result["error"]))
        return result

    def queue_length(self) -> int:
        with self._condition: return max(0, self._submitted - self._received)

    def outstanding_tasks(self) -> int:
        """Submitted tasks not yet returned, including running and paused tasks."""
        return self.queue_length()

    def inflight_tasks(self) -> int:
        """Tasks already claimed by workers, including permit-paused tasks."""
        return int(self._inflight.value)

    def retrieval_queue_depth(self) -> int:
        """Tasks waiting to be claimed by a worker."""
        return max(0, self.outstanding_tasks() - self.inflight_tasks())

    def active_retrievals(self) -> int:
        """Workers currently inside a retrieval search chunk."""
        with self._active_flags.get_lock():
            return sum(int(value) for value in self._active_flags[:])

    def demand_snapshot(self) -> dict[str, int]:
        outstanding = self.outstanding_tasks()
        inflight = self.inflight_tasks()
        active = self.active_retrievals()
        return {
            "permitted_workers": self.permits,
            "active_retrievals": active,
            "inflight_tasks": inflight,
            "retrieval_queue_depth": max(0, outstanding - inflight),
            "paused_inflight_tasks": max(0, inflight - active),
            "effective_backlog_tasks": max(0, outstanding - active),
            "outstanding_tasks": outstanding,
            "admitted_queries": int(self._admitted_queries.value),
            "completed_queries": int(self._completed_queries.value),
        }

    def alive_workers(self) -> int:
        return sum(p.is_alive() for p in self._processes)

    def worker_pids(self) -> list[int]:
        return [int(p.pid) for p in self._processes if p.pid is not None]

    def close(self) -> None:
        self._stop.set()
        self.set_permits(self.max_workers)
        for _ in self._processes: self._tasks.put(None)
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive(): p.terminate()

    def __enter__(self) -> "CPUTaskManager": return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: self.close()
