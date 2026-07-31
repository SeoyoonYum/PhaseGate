# Static phase-aware scheduler experiment

This experiment asks whether the best static `PhaseGate(prefill_cap, decode_cap)`
delivers more always-backlogged HNSW goodput than the best phase-oblivious
`Fixed-k` policy under the same normalized TPOT and TTFT SLO.

Calibration and held-out evaluation use different deterministic prompt-token and
HNSW query seeds. Each campaign first measures its own isolated LLM baseline.
The primary joint SLO is 1.10 times that campaign's isolated request-level p95
TPOT and p95 TTFT. The old absolute 19 ms TPOT threshold is retained only as a
secondary column.

Smoke data are written to separate `smoke_*.jsonl` and `*_smoke_summary.csv`
files and must not be interpreted as research results.

```bash
# Functional smoke only
.venv/bin/python scripts/run_static_phaseaware_pilot.py \
  --smoke --repeats 1 --allow-fanless-pilot
.venv/bin/python scripts/analyze_static_phaseaware.py

# Calibration on the target device
.venv/bin/python scripts/run_static_phaseaware_pilot.py --repeats 3
.venv/bin/python scripts/analyze_static_phaseaware.py

# Held-out evaluation after the selection file is frozen
.venv/bin/python scripts/run_static_phaseaware_evaluation.py --repeats 5
.venv/bin/python scripts/analyze_static_phaseaware.py
```

Raw attempts are append-only and resume by `(policy, repeat, attempt)`. Invalid
memory, pageout, swap, sentinel, drift, backlog, and cap-semantics attempts remain
in the corresponding JSONL.

## Stability-first paired pilot

The fanless-M4 quick check runs only `Fixed-1` versus `PhaseGate 4→1` and
`Fixed-2` versus `PhaseGate 4→2`. It uses separate paired sessions, fresh
processes per block, and checkpoints accepted pairs so an interrupted run can
resume without repeating them.

```bash
.venv/bin/python scripts/run_static_phaseaware_paired_pilot.py --max-attempts 16
.venv/bin/python scripts/analyze_static_phaseaware_paired_pilot.py
```

Results live under `paired_pilot/`. This command does not run the calibration
matrix, select a best policy, or launch held-out evaluation.

## Strict-SLO decode-cap-zero pilot

This focused pilot tests only the operating region where decode admission is
zero: `Fixed-0`, `PhaseGate 1→0`, `PhaseGate 2→0`, and `PhaseGate 4→0`.
Each comparison is a separate paired session with its own clean isolated
baseline. It does not retest decode-cap-positive policies.

```bash
.venv/bin/python scripts/run_decodecap0_pilot.py --max-attempts 16
.venv/bin/python scripts/analyze_decodecap0_pilot.py
```

The runner treats any decode-window retrieval admission as invalid for this
pilot and preserves invalid attempts under `decodecap0_pilot/raw/`.
