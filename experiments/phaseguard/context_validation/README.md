# Context-4096 prefill validation

This directory contains the focused validation of the apparent context-4096
prefill slowdown. It does not alter the original Phase 0 or PhaseGuard data.

## Layout

- `EXISTING_RESULT_AUDIT.md`: provenance and metric-semantics audit performed
  before new measurements.
- `configs/primary.json`: primary isolated configuration.
- `raw/isolated_runs.jsonl`, `raw/iterations.jsonl`: run and per-iteration
  isolated phase records, including failed/suspect flags.
- `raw/pipeline_runs.jsonl`, `raw/pipeline_requests.jsonl`: uncoordinated
  pipeline run and request records.
- `processed/`: all-run, strict-clean, stationary-clean, contaminated,
  isolated-versus-pipeline, and memory diagnostic tables.
- `plots/`: Figures 1--6 in PNG and PDF.
- `logs/`: command/environment manifests and before/after idle records.

## Reproduction

The harness resumes by `run_key`; use a new attempt number for a genuinely new
independent attempt.

```bash
.venv/bin/python scripts/run_context_validation.py --attempt 1 \
  --workloads none,hnsw1,hnsw4,random --phases prefill,decode \
  --prefill-warmups 3 --prefill-iterations 20 --decode-tokens 128 \
  --cooldown 10 --idle-seconds 30 --allow-battery

.venv/bin/python scripts/run_context_pipeline.py --attempt 2 \
  --context 4096 --workload none --requests 12 --idle-seconds 15 \
  --allow-battery

.venv/bin/python scripts/analyze_context_validation.py
```

Metal execution and unrestricted `sysctl vm.swapusage` access may require running
outside a headless filesystem sandbox. Root permission is not required. The
thermal flags are derived from within-run latency drift because this device did
not expose temperature or GPU-frequency telemetry through the repository's
unprivileged controls.

## Interpretation rule

`clean` retains the predeclared strict memory rule (zero experiment-window
pageouts, zero swap growth, adequate headroom) and the latency-drift rule. No run
is relabeled. `stationary_pair` is an additional processed field requiring both
the loaded point and its same-attempt no-load baseline to pass and remain within
10% first/last drift. The fixed first-five-iteration window is reported alongside,
not instead of, the full run.
