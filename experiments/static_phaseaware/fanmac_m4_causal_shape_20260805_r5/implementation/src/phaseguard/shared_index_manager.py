"""One-process, one-index HNSW executor with non-preemptive cap changes."""
from __future__ import annotations

import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .retrieval import load_index, make_queries, search_chunks


@dataclass
class SharedRetrievalResult:
    task_id: str
    request_id: str
    worker_id: int
    submitted: float
    started: float
    ended: float
    queries: int
    chunks: int
    checksum: float
    query_latencies_s: list[float]
    error: str | None = None


class SharedIndexTaskManager:
    """External threads share one read-only FAISS index in the owner process.

    A worker checks the permit only before claiming a task. If the cap falls,
    already-active tasks drain without preemption while no disallowed worker can
    admit a new task.
    """

    def __init__(self, index_path: str | Path, max_workers: int, ef_search: int = 128,
                 top_k: int = 10,
                 event_sink: Callable[..., None] | None = None) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers
        self.index_path = str(index_path)
        self.index, self.index_metadata = load_index(self.index_path, ef_search)
        self.top_k = top_k
        self._permit = max_workers
        self._condition = threading.Condition()
        self._tasks: queue.Queue[tuple[str, str, float, int, int, int] | None] = queue.Queue()
        self._completed: dict[str, dict[str, Any]] = {}
        self._active = [False] * max_workers
        self._inflight = [False] * max_workers
        self._submitted_tasks = 0
        self._received_tasks = 0
        self._admitted_queries = 0
        self._completed_queries = 0
        self._stop = False
        self._executor: ThreadPoolExecutor | None = None
        self.index_load_count = 1
        self._event_sink = event_sink

    def _emit(self, event_type: str, timestamp: float | None = None, **fields: Any) -> None:
        if self._event_sink is not None:
            self._event_sink(event_type, timestamp=timestamp, **fields)

    def start(self) -> "SharedIndexTaskManager":
        if self._executor is not None:
            raise RuntimeError("manager already started")
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers,
                                            thread_name_prefix="phasegate-hnsw")
        for worker_id in range(self.max_workers):
            self._executor.submit(self._worker, worker_id)
        return self

    def _worker(self, worker_id: int) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stop or worker_id < self._permit)
                if self._stop:
                    return
            try:
                task = self._tasks.get(timeout=0.05)
            except queue.Empty:
                continue
            if task is None:
                return
            task_id, request_id, submitted, n_queries, chunk, seed = task
            started = time.perf_counter()
            self._emit("retrieval_task_started", started, query_id=task_id,
                       request_id=request_id, requested_cap=self.permits)
            result = SharedRetrievalResult(task_id, request_id, worker_id, submitted,
                                           started, started, 0, 0, 0.0, [])
            try:
                with self._condition:
                    self._inflight[worker_id] = True
                queries = make_queries(seed, n_queries, int(self.index.d))
                iterator = search_chunks(self.index, queries, self.top_k, chunk)
                chunk_index = 0
                while True:
                    if result.queries >= n_queries:
                        break
                    with self._condition:
                        self._condition.wait_for(
                            lambda: self._stop or (
                                worker_id < self._permit and sum(self._active) < self._permit
                            ))
                        if self._stop:
                            raise RuntimeError("manager stopped")
                        self._active[worker_id] = True
                        active_after_admit = sum(self._active)
                        requested_cap = self._permit
                    query_id = f"{task_id}:{chunk_index}"
                    query_count = min(chunk, n_queries - result.queries)
                    query_started = time.perf_counter()
                    self._emit("query_admitted", query_started, query_id=query_id,
                               request_id=request_id, query_count=query_count,
                               requested_cap=requested_cap,
                               active_query_count=active_after_admit, worker_id=worker_id)
                    self._emit("query_started", query_started, query_id=query_id,
                               request_id=request_id, query_count=query_count,
                               requested_cap=requested_cap,
                               active_query_count=active_after_admit, worker_id=worker_id)
                    try:
                        count, checksum = next(iterator)
                    except StopIteration:
                        with self._condition:
                            self._active[worker_id] = False
                            self._condition.notify_all()
                        break
                    query_ended = time.perf_counter()
                    with self._condition:
                        self._active[worker_id] = False
                        self._admitted_queries += count
                        self._completed_queries += count
                        active_after_complete = sum(self._active)
                        requested_cap = self._permit
                        self._condition.notify_all()
                    self._emit("query_completed", query_ended, query_id=query_id,
                               request_id=request_id, query_count=count,
                               requested_cap=requested_cap,
                               active_query_count=active_after_complete, worker_id=worker_id)
                    result.queries += count; result.chunks += 1; result.checksum += checksum
                    result.query_latencies_s.extend([(query_ended - query_started) / count] * count)
                    chunk_index += 1
            except Exception as exc:  # retain accounting evidence for invalidation
                result.error = repr(exc)
            finally:
                result.ended = time.perf_counter()
                with self._condition:
                    self._active[worker_id] = False
                    self._inflight[worker_id] = False
                    self._completed[task_id] = asdict(result)
                    self._received_tasks += 1
                    self._condition.notify_all()
                self._emit("retrieval_task_completed", result.ended, query_id=task_id,
                           request_id=request_id, query_count=result.queries,
                           requested_cap=self.permits)

    @property
    def permits(self) -> int:
        with self._condition:
            return self._permit

    def set_permits(self, count: int) -> None:
        if not 0 <= count <= self.max_workers:
            raise ValueError("permit count outside [0, max_workers]")
        with self._condition:
            self._permit = count
            active = sum(self._active)
            self._condition.notify_all()
        self._emit("cap_change", requested_cap=count, active_query_count=active)

    def submit(self, request_id: str, queries: int, chunk: int = 16,
               seed: int = 20260728) -> str:
        if queries < 1 or chunk < 1:
            raise ValueError("queries and chunk must be positive")
        task_id = uuid.uuid4().hex
        with self._condition:
            self._submitted_tasks += 1
            outstanding = self._submitted_tasks - self._received_tasks
        self._tasks.put((task_id, request_id, time.perf_counter(), queries, chunk, seed))
        self._emit("retrieval_task_submitted", query_id=task_id, request_id=request_id,
                   query_count=queries, requested_cap=self.permits,
                   active_query_count=self.active_retrievals(), outstanding_tasks=outstanding)
        return task_id

    def wait(self, task_id: str, timeout: float = 600.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while task_id not in self._completed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(task_id)
                self._condition.wait(min(remaining, 0.5))
            result = self._completed.pop(task_id)
        if result.get("error"):
            raise RuntimeError(str(result["error"]))
        return result

    def demand_snapshot(self) -> dict[str, int]:
        with self._condition:
            active = sum(self._active)
            inflight = sum(self._inflight)
            outstanding = self._submitted_tasks - self._received_tasks
            return {
                "permitted_workers": self._permit,
                "active_retrievals": active,
                "inflight_tasks": inflight,
                "retrieval_queue_depth": max(0, outstanding - inflight),
                "paused_inflight_tasks": max(0, inflight - active),
                "effective_backlog_tasks": max(0, outstanding - active),
                "outstanding_tasks": outstanding,
                "submitted_tasks": self._submitted_tasks,
                "completed_tasks": self._received_tasks,
                "admitted_queries": self._admitted_queries,
                "completed_queries": self._completed_queries,
            }

    def active_retrievals(self) -> int:
        return self.demand_snapshot()["active_retrievals"]

    def queue_length(self) -> int:
        return self.demand_snapshot()["outstanding_tasks"]

    def outstanding_tasks(self) -> int:
        return self.queue_length()

    def inflight_tasks(self) -> int:
        return self.demand_snapshot()["inflight_tasks"]

    def retrieval_queue_depth(self) -> int:
        return self.demand_snapshot()["retrieval_queue_depth"]

    def alive_workers(self) -> int:
        return self.max_workers if self._executor is not None and not self._stop else 0

    def worker_pids(self) -> list[int]:
        return []  # all FAISS work is in the single owner process

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        for _ in range(self.max_workers):
            self._tasks.put(None)
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

    def __enter__(self) -> "SharedIndexTaskManager":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
