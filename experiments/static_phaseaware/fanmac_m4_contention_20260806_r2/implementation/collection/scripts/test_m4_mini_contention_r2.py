#!/usr/bin/env python3
"""Protocol regression tests for the 300-request Mini contention r2 campaign."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import run_static_phaseaware_pilot as pilot  # noqa: E402
from run_m4_mini_contention import export_baseline  # noqa: E402


def test_pageout_preflight_soft_flag() -> None:
    state = {
        "pageouts_delta": 7,
        "swap_used_delta_bytes": 0,
        "headroom_before_bytes": 8 * 1024**3,
        "headroom_after_bytes": 8 * 1024**3,
    }
    with (patch.object(pilot, "capture_state", return_value={}),
          patch.object(pilot, "state_delta", return_value=state),
          patch.object(pilot.time, "sleep")):
        assert not pilot.memory_preflight([], 0, 3, allow_pageout=False)["passed"]
        assert pilot.memory_preflight([], 0, 3, allow_pageout=True)["passed"]
    swap_state = {**state, "swap_used_delta_bytes": 1}
    with (patch.object(pilot, "capture_state", return_value={}),
          patch.object(pilot, "state_delta", return_value=swap_state),
          patch.object(pilot.time, "sleep")):
        assert not pilot.memory_preflight([], 0, 3, allow_pageout=True)["passed"]


def test_baseline_gate_uses_300_requests_and_soft_pageouts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        campaign = Path(directory)
        raw = campaign / "contention_baseline_A/raw"
        raw.mkdir(parents=True)
        rows = []
        for repeat, pageouts in enumerate((0, 1, 2, 0, 3)):
            rows.append({
                "status": "valid", "repeat": repeat, "run_key": f"r{repeat}",
                "p95_tpot_ms": 12.0, "p95_ttft_ms": 1800.0,
                "pageouts_delta": pageouts, "swap_used_delta_bytes": 0,
                "memory_pressure_clean": True, "duration_s": 1,
                "observer_mode": "event", "observer_subprocess_count_during_block": 0,
                "llm_requests": 300,
                "observer_reconstruction_audit": {
                    "token_events": 300 * 128, "query_accounting_exact": True,
                    "timestamps_monotonic": True,
                },
            })
        (raw / "runs.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
        assert export_baseline(campaign, "A")
        result = json.loads((campaign / "CONTENTION_BASELINE_A_RESULT.json").read_text())
        assert result["passed"] and result["pageout_soft_flag_blocks"] == 3


def main() -> None:
    parsed = pilot.parser().parse_args(["--allow-pageout-preflight"])
    assert parsed.allow_pageout_preflight
    test_pageout_preflight_soft_flag()
    test_baseline_gate_uses_300_requests_and_soft_pageouts()
    print("M4 Mini contention r2 protocol: PASS")


if __name__ == "__main__":
    main()
