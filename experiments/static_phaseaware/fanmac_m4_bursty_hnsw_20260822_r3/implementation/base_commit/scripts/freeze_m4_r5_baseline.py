#!/usr/bin/env python3
"""Freeze both baseline trace sets before the first r5 baseline block."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def traces(base: int) -> list[dict[str, int]]:
    return [{"repeat": repeat, "prompt_seed": base + repeat * 10007,
             "query_seed": base + 500003 + repeat * 10007} for repeat in range(5)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--index", required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    manifest = json.loads((campaign / "machine_manifest.json").read_text())
    hardware = manifest["hardware"]
    if hardware["chip"] != "Apple M4" or hardware["model_identifier"] != "Mac16,10":
        raise SystemExit(f"expected base-M4 Mac16,10, found {hardware}")
    freeze_path = campaign / "R5_BASELINE_FREEZE.json"
    if freeze_path.exists():
        raise SystemExit(f"refusing to replace existing freeze: {freeze_path}")
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_head(),
        "machine": {"model_identifier": hardware["model_identifier"],
                    "chip": hardware["chip"], "performance_cores": 4,
                    "efficiency_cores": 6, "gpu_cores": 10,
                    "unified_memory": hardware["unified_memory"]},
        "artifacts": {
            "model_id": args.model_id, "model_revision": args.model_revision,
            "model_local_path": str(args.model.resolve()), "quantization": "4-bit",
            "index_path": str(args.index.resolve()), "index_sha256": sha256(args.index.resolve()),
        },
        "workload": {
            "observer_mode": "event", "context_tokens": 2048, "output_tokens": 128,
            "measured_requests": 300, "valid_repeats": 5,
            "fresh_process_per_repeat": True, "cpu_retrieval": False,
            "warmup": "model global warmup plus stabilized long-context prefill sentinel",
            "token_timestamp_semantics": "unchanged: after mx.eval on the model step",
            "mlx_memory_limit_gb": 5.5,
        },
        "trace_sets": {"A": traces(3500000017), "B": traces(3600000023)},
        "acceptance": {
            "definition": "every run within +/-3% of its set median for official run p95 TPOT and TTFT",
            "relative_tolerance": 0.03,
            "hard_invalid": ["crash_or_hang", "swap_growth", "warning_or_critical_memory_pressure",
                             "token_or_event_corruption", "wrong_token_count",
                             "non_monotonic_timestamps", "event_reconstruction_mismatch",
                             "production_observer_subprocess_launch"],
            "hard_invalid_retry_limit": 1,
            "set_policy": "run A first; if it fails, diagnose and permit only frozen B after a documented environmental correction",
        },
        "diagnostic_threshold_ms": 14.0,
        "diagnostic_threshold_status": "derived from r4; diagnostic only, not a paper metric",
        "next_stage_if_pass": "r5 CPU-only scaling caps 1,2,4",
    }
    freeze_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(freeze_path)
    print(sha256(freeze_path))


if __name__ == "__main__":
    main()
