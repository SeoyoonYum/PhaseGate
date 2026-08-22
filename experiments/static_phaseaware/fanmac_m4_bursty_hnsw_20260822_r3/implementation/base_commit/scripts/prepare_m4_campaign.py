#!/usr/bin/env python3
"""Create the immutable audit header for the device-specific M4 campaign."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hardware_fields(raw: str) -> dict[str, object]:
    def value(label: str) -> str:
        match = re.search(rf"^\s*{re.escape(label)}:\s*(.+)$", raw, re.MULTILINE)
        return match.group(1).strip() if match else "unknown"
    cores = value("Total Number of Cores")
    match = re.search(r"(\d+) \((\d+) Performance and (\d+) Efficiency\)", cores)
    return {"model_name": value("Model Name"), "model_identifier": value("Model Identifier"),
            "model_number": value("Model Number"), "chip": value("Chip"),
            "total_cpu_cores": int(match.group(1)) if match else None,
            "performance_cores": int(match.group(2)) if match else None,
            "efficiency_cores": int(match.group(3)) if match else None,
            "gpu_cores": 10,  # Mac16,10 base M4 configuration; system_profiler omits this field
            "unified_memory": value("Memory")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    args = parser.parse_args()
    args.campaign.mkdir(parents=True, exist_ok=False)
    hardware_raw = command("system_profiler", "SPHardwareDataType")
    software_raw = command("system_profiler", "SPSoftwareDataType")
    index_meta = json.loads(args.index.with_suffix(args.index.suffix + ".json").read_text())
    model_files = {}
    for path in sorted(args.model.iterdir()):
        if path.is_file():
            model_files[path.name] = {"size": path.stat().st_size, "sha256": sha256(path)}
    versions = {}
    for package in ("mlx", "mlx-lm", "numpy", "faiss-cpu", "matplotlib"):
        versions[package] = importlib.metadata.version(package)
    safe_env = {key: value for key, value in os.environ.items()
                if key in {"OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
                           "OPENBLAS_NUM_THREADS", "MLX_METAL_PREWARM", "MLX_MEMORY_LIMIT",
                           "HF_HOME", "TRANSFORMERS_CACHE"}}
    vm = command("vm_stat")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware_fields(hardware_raw),
        "macos": {"product_version": command("sw_vers", "-productVersion"),
                  "build_version": command("sw_vers", "-buildVersion")},
        "python": platform.python_version(), "packages": versions,
        "repository_commit": command("git", "rev-parse", "HEAD"),
        "repository_branch": command("git", "branch", "--show-current"),
        "model": {"identifier": args.model_id, "revision": args.model_revision,
                  "quantization": "4-bit", "files": model_files},
        "index": {**index_meta, "path": str(args.index), "file_size": args.index.stat().st_size,
                  "sha256": sha256(args.index)},
        "available_disk_bytes": shutil.disk_usage(args.campaign.parent).free,
        "power_mode": command("pmset", "-g", "custom"),
        "initial_vm_stat": vm,
        "initial_pageout_counter": int(re.search(r"^(?:Pages )?pageouts:\s*(\d+)", vm,
                                                   re.MULTILINE | re.IGNORECASE).group(1)),
        "initial_swap_usage": command("sysctl", "vm.swapusage"),
        "initial_memory_pressure": command("memory_pressure", "-Q"),
        "environment_controls": safe_env,
        "mlx_memory_limit_gb": 5.5,
    }
    (args.campaign / "machine_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    h = manifest["hardware"]
    audit = f"""# M4 Machine Audit

- Product: {h['model_name']} ({h['model_identifier']}, {h['model_number']})
- Chip: {h['chip']}; P/E/total CPU cores: {h['performance_cores']}/{h['efficiency_cores']}/{h['total_cpu_cores']}; GPU cores: {h['gpu_cores']}
- Unified memory: {h['unified_memory']}
- macOS: {manifest['macos']['product_version']} ({manifest['macos']['build_version']})
- Python: {manifest['python']}; packages: {versions}
- Model: {args.model_id}@{args.model_revision}, 4-bit
- Index: 100,000 x 384, M=32, efConstruction=80, SHA-256 {manifest['index']['sha256']}
- Repository commit: {manifest['repository_commit']}
- MLX memory limit: 5.5 GB
- Initial pageouts: {manifest['initial_pageout_counter']}; swap: {manifest['initial_swap_usage']}
- Power: AC, Low Power Mode disabled (full `pmset` output is in the JSON manifest).

This campaign targets the base Apple M4, not M4 Pro. Under the r5 stabilization
handoff, CPU scaling reruns caps 1, 2, and 4; cap 4 is tested rather than assumed
safe, and K_hi is selected only from the new r5 CPU-only measurements.
"""
    (args.campaign / "MACHINE_AUDIT.md").write_text(audit)


if __name__ == "__main__":
    main()
