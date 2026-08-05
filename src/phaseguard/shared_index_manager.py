"""One-process, one-index HNSW executor with non-preemptive cap changes."""
from __future__ import annotations

import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
                 top_k: int = 10) -> None:
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
        self._submitted_tasks = 0
        self._received_tasks = 0
        self._admitted_queries = 0
        self._completed_queries = 0
        self._stop = False
        self._executor: ThreadPoolExecutor | None = None
        self.index_load_count = 1

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
            result = SharedRetrievalResult(task_id, request_id, worker_id, submitted,
                                           started, started, 0, 0, 0.0, [])
            try:
                with self._condition:
                    self._active[worker_id] = True
                    self._admitted_queries += n_queries
                queries = make_queries(seed, n_queries, int(self.index.d))
                for count, checksum in search_chunks(self.index, queries, self.top_k, chunk):
                    query_ended = time.perf_counter()
                    result.queries += count
                    result.chunks += 1
                    result.checksum += checksum
                    result.query_latencies_s.extend(
                        [(query_ended - started) / count] * count
                        if result.chunks == 1 else [0.0] * count
                    )
                    started = query_ended
                with self._condition:
                    self._completed_queries += result.queries
            except Exception as exc:  # retain accounting evidence for invalidation
                result.error = repr(exc)
            finally:
                result.ended = time.perf_counter()
                with self._condition:
                    self._active[worker_id] = False
                    self._completed[task_id] = asdict(result)
                    self._received_tasks += 1
                    self._condition.notify_all()

    @property
    def permits(self) -> int:
        with self._condition:
            return self._permit

    def set_permits(self, count: int) -> None:
        if not 0 <= count <= self.max_workers:
            raise ValueError("permit count outside [0, max_workers]")
        with self._condition:
            self._permit = count
            self._condition.notify_all()

    def submit(self, request_id: str, queries: int, chunk: int = 16,
               seed: int = 20260728) -> str:
        if queries < 1 or chunk < 1:
            raise ValueError("queries and chunk must be positive")
        task_id = uuid.uuid4().hex
        with self._condition:
            self._submitted_tasks += 1
        self._tasks.put((task_id, request_id, time.perf_counter(), queries, chunk, seed))
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
            outstanding = self._submitted_tasks - self._received_tasks
            return {
                "permitted_workers": self._permit,
                "active_retrievals": active,
                "inflight_tasks": active,
                "retrieval_queue_depth": max(0, outstanding - active),
                "paused_inflight_tasks": max(0, active - self._permit),
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
