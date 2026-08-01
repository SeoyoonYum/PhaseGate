#!/usr/bin/env python3
"""Stage B: profile isolated MLX prefill/decode under real HNSW retrieval."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import threading
import time
import uuid
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROFILE_ROOT = Path(os.environ.get(
    "PHASEGUARD_PROFILE_ROOT", REPO / "experiments/phaseguard"
)).resolve()
sys.path.insert(0, str(REPO / "src"))
from common import measure, models, thermal  # noqa: E402
from exp2_contention import build_cache  # noqa: E402
from phaseguard.cpu_task_manager import CPUTaskManager  # noqa: E402
from phaseguard.metrics import append_jsonl, environment_metadata, percentile, process_rss_bytes, vm_snapshot  # noqa: E402


class BackgroundRetrieval:
    """Keep at most one short retrieval task queued per persistent worker."""
    def __init__(self, manager: CPUTaskManager, queries_per_task: int, seed: int) -> None:
        self.manager, self.queries_per_task, self.seed = manager, queries_per_task, seed
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.queries = 0
        self.threads: list[threading.Thread] = []

    def start(self) -> "BackgroundRetrieval":
        def feed(slot: int) -> None:
            sequence = 0
            while not self.stop_event.is_set():
                tid = self.manager.submit(f"profile-{slot}", self.queries_per_task, 1,
                                          self.seed + slot * 100_000 + sequence)
                result = self.manager.wait(tid)
                with self.lock: self.queries += int(result["queries"])
                sequence += 1
        self.threads = [threading.Thread(target=feed, args=(i,), daemon=True)
                        for i in range(self.manager.max_workers)]
        for thread in self.threads: thread.start()
        return self

    def snapshot(self) -> int:
        with self.lock: return self.queries

    def stop(self) -> None:
        self.stop_event.set()
        self.manager.set_permits(self.manager.max_workers)
        for thread in self.threads: thread.join(timeout=120)
        if any(t.is_alive() for t in self.threads): raise RuntimeError("retrieval feeder did not drain")


def timed_prefill(model: object, context: int) -> list[float]:
    return measure.time_prefill(model, context, reps=1, warmup=0)


def timed_decode(model: object, context: int, steps: int) -> list[float]:
    cache = build_cache(model, context)
    token = mx.array([[7]], dtype=mx.int32)
    mx.eval(token)
    y = None
    for _ in range(3):
        y = model(token, cache=cache); mx.eval(y)
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        y = model(token, cache=cache); mx.eval(y)
        times.append((time.perf_counter() - t0) * 1e3)
    del cache, token, y
    measure.free_buffers()
    return times


def load_existing(path: Path) -> tuple[list[dict[str, object]], set[tuple[int, int, str]]]:
    rows, keys = [], set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line); rows.append(row)
            if row.get("status") == "ok": keys.add((int(row["repeat"]), int(row["workers"]), str(row["phase"])))
    return rows, keys


def write_profile(rows: list[dict[str, object]], path: Path, model_name: str,
                  context: int, workload: str) -> None:
    good = [r for r in rows if r.get("status") == "ok"]
    fields = ["model", "context", "phase", "workload", "workers", "p50_ms", "p95_ms",
              "p99_ms", "slowdown", "logical_qps", "repetitions"]
    output = []
    for phase in ("PREFILL", "DECODE"):
        baseline = [float(x) for r in good if r["phase"] == phase and int(r["workers"]) == 0
                    for x in r["timings_ms"]]
        if not baseline: continue
        baseline_p95 = percentile(baseline, 95)
        for workers in sorted({int(r["workers"]) for r in good if r["phase"] == phase}):
            subset = [r for r in good if r["phase"] == phase and int(r["workers"]) == workers]
            timings = [float(x) for r in subset for x in r["timings_ms"]]
            output.append({"model": model_name, "context": context, "phase": phase,
                           "workload": workload, "workers": workers,
                           "p50_ms": percentile(timings, 50), "p95_ms": percentile(timings, 95),
                           "p99_ms": percentile(timings, 99),
                           "slowdown": percentile(timings, 95) / baseline_p95,
                           "logical_qps": float(np.median([float(r["logical_qps"]) for r in subset])),
                           "repetitions": len(subset)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="experiments/phaseguard/index/hnsw_100k_d384.faiss")
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--workers", default="0,1,2,4")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--queries-per-task", type=int, default=32)
    ap.add_argument("--cooldown", type=float, default=4.0)
    ap.add_argument("--mem-limit-gb", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.workers, args.repetitions, args.decode_steps, args.cooldown = "0,1", 1, 16, 0.2
    workers = [int(x) for x in args.workers.split(",")]
    if any(w < 0 or w > args.max_workers for w in workers): raise SystemExit("invalid --workers")
    index = Path(args.index); index = index if index.is_absolute() else REPO / index
    power_ok = thermal.assert_power(args.allow_battery)
    measure.set_mem_limit_gb(args.mem_limit_gb)
    raw = PROFILE_ROOT / "raw" / f"profile_{args.tag}.jsonl"
    profile = PROFILE_ROOT / "processed" / f"profile_{args.tag}.csv"
    rows, complete = load_existing(raw)
    manifest = {"run_id": uuid.uuid4().hex, "kind": "profile", "arguments": vars(args),
                "environment": environment_metadata(REPO), "power_ok": power_ok,
                "index": json.loads(index.with_suffix(index.suffix + ".json").read_text()),
                "start_vm": vm_snapshot()}
    (PROFILE_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (PROFILE_ROOT / "logs" / f"profile_{args.tag}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[load] {models.resolve_model_id(args.model)}")
    model, _ = models.load_model(args.model); mx.eval(model.parameters())
    measure.global_warmup(model)
    rng = random.Random(args.seed)
    with CPUTaskManager(str(index), args.max_workers, args.ef_search, args.top_k) as manager:
        order = [(rep, w, phase) for rep in range(args.repetitions) for w in workers
                 for phase in ("PREFILL", "DECODE") if (rep, w, phase) not in complete]
        rng.shuffle(order)
        for rep, permitted, phase in order:
            before_vm = vm_snapshot(); manager.set_permits(permitted)
            bg = BackgroundRetrieval(manager, args.queries_per_task,
                                     args.seed + rep * 10_000 + permitted * 100).start() if permitted else None
            if bg: time.sleep(0.15)
            q0, t0 = (bg.snapshot() if bg else 0), time.perf_counter()
            status, error = "ok", None
            try: timings = timed_prefill(model, args.context) if phase == "PREFILL" else timed_decode(model, args.context, args.decode_steps)
            except Exception as exc: timings, status, error = [], "failed", repr(exc)
            elapsed = time.perf_counter() - t0
            q1 = bg.snapshot() if bg else 0
            if bg: bg.stop()
            after_vm = vm_snapshot()
            pageouts_delta = after_vm.get("pageouts", 0) - before_vm.get("pageouts", 0)
            swapouts_delta = after_vm.get("swapouts", 0) - before_vm.get("swapouts", 0)
            if status == "ok" and (pageouts_delta != 0 or swapouts_delta != 0):
                status, error = "invalid", "pageout_or_swapout_contamination"
            row = {"run_id": manifest["run_id"], "repeat": rep, "workers": permitted,
                   "phase": phase, "model": args.model, "context": args.context,
                   "ef_search": args.ef_search, "timings_ms": timings,
                   "p50_ms": percentile(timings, 50) if timings else None,
                   "p95_ms": percentile(timings, 95) if timings else None,
                   "logical_qps": (q1 - q0) / elapsed if elapsed else 0.0,
                   "elapsed_s": elapsed, "rss_bytes": process_rss_bytes(),
                   "headroom_bytes": after_vm.get("headroom_bytes", 0),
                   "pageouts_delta": pageouts_delta, "swapouts_delta": swapouts_delta,
                   "power_contaminated": not power_ok, "thermal_spread":
                   (max(timings) / float(np.median(timings))) if timings else None,
                   "status": status, "error": error}
            append_jsonl(raw, row); rows.append(row); write_profile(rows, profile, args.model, args.context, "faiss-hnsw")
            print(f"[{phase}] rep={rep} workers={permitted} status={status} "
                  f"p95={row['p95_ms']:.3f}ms qps={row['logical_qps']:.1f}")
            time.sleep(args.cooldown)
    print(f"[out] {raw}\n[out] {profile}")


if __name__ == "__main__": main()
