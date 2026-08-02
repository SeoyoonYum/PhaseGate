#!/usr/bin/env python3
"""Headless quick-preflight model/index load and lazy warm-up check."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from common import measure, models  # noqa: E402
from phaseguard.context_validation import capture_state, state_delta, total_rss_bytes  # noqa: E402
from phaseguard.cpu_task_manager import CPUTaskManager  # noqa: E402


def process_rss(name: str) -> int:
    found = subprocess.run(
        ["pgrep", "-x", name], capture_output=True, text=True, timeout=5
    ).stdout.splitlines()
    if not found:
        return 0
    raw = subprocess.run(
        ["ps", "-p", found[0], "-o", "rss="], capture_output=True, text=True, timeout=5
    ).stdout.strip()
    return int(raw) * 1024 if raw else 0


def snapshot(pids: list[int]) -> dict[str, object]:
    state = asdict(capture_state(pids))
    state.update({
        "tracked_pids": [os.getpid(), *pids],
        "ARDAgent_rss_bytes": process_rss("ARDAgent"),
        "WindowServer_rss_bytes": process_rss("WindowServer"),
    })
    return state


def clean(delta: dict[str, object]) -> bool:
    return int(delta["pageouts_delta"]) == 0 and int(delta["swap_used_delta_bytes"] or 0) == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--mem-limit-gb", type=float, default=6.0)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--ef-search", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--query-seed", type=int, default=54265801)
    args = ap.parse_args()

    worker_pids: list[int] = []
    peak_rss = [0]
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.wait(0.1):
            peak_rss[0] = max(peak_rss[0], total_rss_bytes([os.getpid(), *worker_pids]))

    sampler = threading.Thread(target=sample_rss, daemon=True)
    before = snapshot([])
    sampler.start()
    result: dict[str, object] = {
        "configuration": vars(args),
        "before_load": before,
        "status": "running",
    }
    try:
        measure.set_mem_limit_gb(args.mem_limit_gb)
        model, _ = models.load_model(args.model)
        mx.eval(model.parameters())
        model_loaded = snapshot([])
        result["model_loaded"] = model_loaded
        result["model_load_delta"] = state_delta_from_dict(before, model_loaded)
        if not clean(result["model_load_delta"]):
            result["status"] = "no-go-model-load"
            return

        with CPUTaskManager(
            str(args.index), args.max_workers, args.ef_search, args.top_k
        ) as manager:
            worker_pids[:] = manager.worker_pids()
            after_load = snapshot(worker_pids)
            result["after_model_index_load"] = after_load
            result["model_index_load_delta"] = state_delta_from_dict(before, after_load)
            if not clean(result["model_index_load_delta"]):
                result["status"] = "no-go-index-load"
                return

            measure.global_warmup(model)
            manager.set_permits(args.max_workers)
            tasks = [
                manager.submit(f"headless-warmup-{slot}", 16, 16, args.query_seed + slot)
                for slot in range(args.max_workers)
            ]
            for task in tasks:
                manager.wait(task)
            manager.set_permits(0)
            after_warmup = snapshot(worker_pids)
            result["after_warmup"] = after_warmup
            result["warmup_delta"] = state_delta_from_dict(after_load, after_warmup)
            result["cumulative_delta"] = state_delta_from_dict(before, after_warmup)
            result["status"] = "pass" if clean(result["cumulative_delta"]) else "no-go-warmup"
    finally:
        stop.set()
        sampler.join(timeout=2)
        peak_rss[0] = max(peak_rss[0], total_rss_bytes([os.getpid(), *worker_pids]))
        result["peak_tracked_rss_bytes"] = peak_rss[0]
        print(json.dumps(result, indent=2, default=str))


def state_delta_from_dict(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    from phaseguard.context_validation import HostState

    return state_delta(HostState(**{
        key: before[key] for key in HostState.__dataclass_fields__
    }), HostState(**{
        key: after[key] for key in HostState.__dataclass_fields__
    }))


if __name__ == "__main__":
    main()
