# PhaseGate backlog-demand experiment handoff

## Status

The new workload structure, demand instrumentation, thermal sentinel, randomized
calibration/evaluation orchestration, held-out fixed-k selection, and analysis are
implemented. Functional smoke tests completed on the current Apple M4 MacBook Air.

The primary matrix is intentionally **not** executed on this host: `Mac16,13`
MacBook Air is fanless, while this experiment requires a fan-cooled Mac. The runner
fails loudly on a MacBook Air unless both `--smoke` and
`--allow-fanless-smoke` are supplied. Fanless smoke rows are excluded from primary
processed tables.

## What changed

- `CPUTaskManager` now exposes actual search-active workers, in-flight tasks,
  queued tasks, claimed-but-paused tasks, effective backlog, outstanding tasks,
  and current permits through shared process-safe counters.
- `run_staggered_backlog` starts request A alone and injects 1/2/4 future requests
  at each decode boundary for requested low/medium/high demand.
- A 2 ms sampler records actual activity rather than configured workers.
- Metrics are reported both across the complete finite trace and within the
  offered-load-open steady window; the GPU drain is retained, never deleted.
- The demand-aware PhaseGuard applies its profiled decode cap dynamically when
  outstanding demand exceeds the safe cap.
- Serialization, fixed-k, uncoordinated, static PhaseGate, and PhaseGuard use the
  same arrival rule.
- Every primary block requires a no-load prefill sentinel within ±5% of its stored
  reference. Block order is randomized and durable.
- Fixed-k is selected only from calibration runs satisfying the steady-window
  TPOT SLO, then frozen for held-out evaluation.

## Fanless functional smoke evidence

These values validate mechanics only; output length was eight tokens and the host
has no fan.

| Requested demand | Policy | Steady active retrievals | Decode overlap | Scheduler cap binding | Steady p95 TPOT |
|---|---|---:|---:|---:|---:|
| Low | Uncoordinated | 0.98 | 98.0% | 0% | 14.78 ms |
| Medium | Uncoordinated | 2.26 | 98.3% | 0% | 16.76 ms |
| High | Uncoordinated | 3.19 | 99.4% | 0% | 19.54 ms |
| High | PhaseGuard | 1.05 | 78.9% | 91.7% | 13.68 ms |
| High | Serialization | 0.00 | 0% | 98.9% | 12.63 ms |
| High | Fixed-1 | 1.00 | 100% | 95.2% | 13.09 ms |
| High | Static PhaseGate | 0.39 | 21.4% | 99.0% | 12.98 ms |

Static PhaseGate is not exactly zero-active because an already running FAISS chunk
is allowed to finish before the worker pauses. PhaseGuard's multiprocessing queue
can read zero while tasks are claimed-but-paused; the validated High smoke measured
2.07 such tasks and 2.07 effective backlog on average.

Do not cite policy differences in this smoke table as a result. They are single,
non-randomized fanless runs. The useful result is that the requested demand bands
and real cap binding now exist.

## 19 ms TPOT-budget smoke

The runner accepts an absolute target through `--target-tpot-ms`. Decode CPU
progress is measured directly from completed FAISS query chunks, rather than
inferred from the configured worker count. A separate cap-3 calibration block
met the 19 ms target, after which three randomized functional evaluation blocks
were run per policy.

| Policy | Decode cap | Steady p95 TPOT | Decode retrieval progress | Whole-block retrieval goodput | 19 ms passes |
|---|---:|---:|---:|---:|---:|
| Uncoordinated | 4 | 19.06 ms | 1,242 queries/s | 673.3 queries/s | 1/3 |
| PhaseGuard, conservative | 1 | 14.75 ms | 721 queries/s | 681.0 queries/s | 3/3 |
| PhaseGuard, 19 ms target | 3 | 17.93 ms | 1,189 queries/s | 674.0 queries/s | 3/3 |

These are medians of three eight-token blocks on the fanless M4 Air. Relative
to conservative PhaseGuard, spending the TPOT headroom raised decode-window CPU
progress by 65.0%. Relative to uncoordinated, it retained 95.8% of decode-window
CPU progress and improved whole-block goodput by only 0.1%, while lowering steady
p95 TPOT by 5.9% and satisfying the 19 ms target in all three blocks. The latter
goodput difference is effectively zero at smoke scale and must not be presented
as a primary result.

All nine blocks passed the baseline sentinel within +2.15%/-0.07%, and none
increased swap usage. Three blocks observed a nonzero pageout delta, including
two of the three 19 ms PhaseGuard blocks, so this is not a clean primary
measurement and is reported only as functional evidence.

The machine-readable aggregation is
`experiments/phaseguard/backlog_validation/processed/tpot_budget_smoke_summary.csv`.
The full fan-cooled experiment must use 128 output tokens and independently
calibrated/evaluated cap sweeps.

Run that TPOT-budget matrix with:

```bash
.venv/bin/python scripts/run_phaseguard_tpot_budget_matrix.py \
  --stage calibration --repeats 5 --target-tpot-ms 19

.venv/bin/python scripts/analyze_phasegate_backlog.py --target-tpot-ms 19

.venv/bin/python scripts/run_phaseguard_tpot_budget_matrix.py \
  --stage evaluation --repeats 5 --target-tpot-ms 19

.venv/bin/python scripts/analyze_phasegate_backlog.py --target-tpot-ms 19
```

## Commands for the fan-cooled host

```bash
# The fan-cooled primary sentinel path is distinct from the smoke reference, so no
# smoke artifact needs to be deleted.

.venv/bin/python scripts/run_phasegate_backlog_matrix.py \
  --stage calibration --repeats 3

.venv/bin/python scripts/analyze_phasegate_backlog.py

# Inspect processed/demand_calibration.csv. If target_met is false, adjust only
# arrival counts/queries and repeat calibration before evaluation.

.venv/bin/python scripts/run_phasegate_backlog_matrix.py \
  --stage evaluation --repeats 3

.venv/bin/python scripts/analyze_phasegate_backlog.py
```

Primary outputs will appear under
`experiments/phaseguard/backlog_validation/{raw,processed,plots,logs}`. Exact block
order is saved in `logs/matrix_*.json`; exact commands and sentinel attempts are in
per-run manifests.

## Decision rule

Only make the proposed claim if held-out results show:

- Low: scheduling gives little or no benefit because measured demand is at most one.
- Medium or High: PhaseGuard satisfies the decode SLO while delivering more
  application/retrieval goodput than serialization and better TPOT than the selected
  phase-oblivious fixed-k or uncoordinated baseline.

If improvement appears only at High, explicitly limit the applicability claim to
sustained-backlog regimes. If Medium also improves, report that as the stronger
result. Preserve negative or SLO-violating blocks.
