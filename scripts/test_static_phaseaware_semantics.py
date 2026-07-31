#!/usr/bin/env python3
"""Fast policy and non-preemptive phase-transition semantics validation."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from analyze_static_phaseaware import select_with_tie_rule  # noqa: E402
from phaseguard.phase_monitor import GPUPhase  # noqa: E402
from phaseguard.policies import FixedWorkerPolicy, StaticCapsPolicy  # noqa: E402
from phaseguard.static_phaseaware import counter_rate_drift, phase_transition_metrics  # noqa: E402


def main() -> None:
    fixed = FixedWorkerPolicy(4, 2)
    assert [fixed.select(phase, 2048) for phase in GPUPhase] == [2, 2, 2]
    phasegate = StaticCapsPolicy(4, 4, 1)
    assert phasegate.select(GPUPhase.IDLE, 2048) == 4
    assert phasegate.select(GPUPhase.PREFILL, 2048) == 4
    assert phasegate.select(GPUPhase.DECODE, 2048) == 1
    decode_zero = StaticCapsPolicy(4, 2, 0)
    assert decode_zero.select(GPUPhase.IDLE, 2048) == 4
    assert decode_zero.select(GPUPhase.PREFILL, 2048) == 2
    assert decode_zero.select(GPUPhase.DECODE, 2048) == 0
    fixed_zero = StaticCapsPolicy(4, 0, 0)
    assert fixed_zero.select(GPUPhase.IDLE, 2048) == 4
    assert fixed_zero.select(GPUPhase.PREFILL, 2048) == 0
    assert fixed_zero.select(GPUPhase.DECODE, 2048) == 0
    samples = [
        {"timestamp": 1.001, "phase": "DECODE", "phase_started": 1.0,
         "request_id": "r0", "active_retrievals": 4, "permitted_workers": 1},
        {"timestamp": 1.006, "phase": "DECODE", "phase_started": 1.0,
         "request_id": "r0", "active_retrievals": 2, "permitted_workers": 1},
        {"timestamp": 1.011, "phase": "DECODE", "phase_started": 1.0,
         "request_id": "r0", "active_retrievals": 1, "permitted_workers": 1},
        {"timestamp": 1.016, "phase": "DECODE", "phase_started": 1.0,
         "request_id": "r0", "active_retrievals": 1, "permitted_workers": 1},
    ]
    metrics = phase_transition_metrics(samples, 1)
    assert abs(metrics["phase_transition_to_cap_ms"] - 11.0) < 1e-6
    assert metrics["decode_cap_overshoot_fraction"] == 0.5
    assert metrics["decode_cap_overshoot_worker_mean"] == 2.0
    assert metrics["decode_cap_overshoot_worker_max"] == 3.0
    assert metrics["decode_cap_applied_fraction"] == 1.0
    progress = [
        {"timestamp": 0.0, "completed_queries": 0},
        {"timestamp": 1.0, "completed_queries": 100},
        {"timestamp": 2.0, "completed_queries": 200},
        {"timestamp": 3.0, "completed_queries": 300},
        {"timestamp": 4.0, "completed_queries": 400},
        {"timestamp": 5.0, "completed_queries": 500},
    ]
    qps = counter_rate_drift(progress, "completed_queries")
    assert qps["first_qps"] == 100.0 and qps["last_qps"] == 100.0
    assert qps["ratio"] == 1.0 and qps["slope_qps_per_s"] == 0.0
    candidates = [
        {"decode_cap": 1, "joint_slo_pass": True, "retrieval_qps": 100.0},
        {"decode_cap": 2, "joint_slo_pass": True, "retrieval_qps": 102.9},
        {"decode_cap": 3, "joint_slo_pass": True, "retrieval_qps": 111.0},
        {"decode_cap": 4, "joint_slo_pass": False, "retrieval_qps": 130.0},
    ]
    selected = select_with_tie_rule(candidates, "decode_cap")
    assert selected is not None and selected["decode_cap"] == 3
    print("static phase-aware policy semantics: PASS")


if __name__ == "__main__":
    main()
