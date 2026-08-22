"""Dependency-free statistics and host-state capture."""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable

import numpy as np


def percentile(values: Iterable[float], q: float) -> float:
    a = np.asarray(list(values), dtype=float)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError("percentile requires non-empty finite values")
    return float(np.percentile(a, q))


def append_jsonl(path: str | Path, row: dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def vm_snapshot() -> dict[str, int]:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    page_size = 16384
    result: dict[str, int] = {}
    for line in out.splitlines():
        if "page size of" in line:
            try: page_size = int(line.split("page size of", 1)[1].split()[0])
            except Exception: pass
        elif ":" in line:
            key, value = line.split(":", 1)
            try: result[key.strip().lower().replace(" ", "_").strip('"')] = int(value.strip().rstrip("."))
            except ValueError: pass
    result["page_size"] = page_size
    free_pages = sum(result.get(k, 0) for k in ("pages_free", "pages_inactive", "pages_speculative", "pages_purgeable"))
    result["headroom_bytes"] = free_pages * page_size
    return result


def process_rss_bytes(pid: int | None = None) -> int:
    target = str(pid or os.getpid())
    out = subprocess.run(["ps", "-o", "rss=", "-p", target], capture_output=True,
                         text=True, timeout=5).stdout.strip()
    return int(out) * 1024 if out else 0


def environment_metadata(repo: str | Path) -> dict[str, object]:
    def version(name: str) -> str:
        try: return metadata.version(name)
        except metadata.PackageNotFoundError: return "unavailable"
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                         text=True, timeout=5).stdout.strip()
    sw = subprocess.run(["sw_vers"], capture_output=True, text=True, timeout=5).stdout.strip()
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                           text=True, timeout=5).stdout.strip()
    return {"git_commit": git, "hostname": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "software": sw, "power": power,
            "dependencies": {n: version(n) for n in
                             ("mlx", "mlx-lm", "faiss-cpu", "numpy", "matplotlib")}}
