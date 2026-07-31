Under the same TPOT and TTFT SLO, it is not yet known whether the best static phase-aware policy achieves higher total retrieval goodput than the best fixed policy; only functional smoke validation is complete.

# Static Phase-Aware Scheduler Evaluation

All data currently summarized below are fanless functional smoke data. They validate policy semantics and the analysis path, not the research claim.

## Calibration smoke

| Policy | Prefill cap | Decode cap | p95 TPOT norm | p95 TTFT norm | Joint SLO | Retrieval QPS |
|---|---:|---:|---:|---:|---:|---:|
| fixed1 | 1 | 1 | 1.106 | 0.991 | 0/1 | 850.5 |
| fixed2 | 2 | 2 | 1.185 | 0.973 | 0/1 | 1457.3 |
| fixed4 | 4 | 4 | 1.628 | 1.027 | 0/1 | 2160.4 |
| llm-only | 0 | 0 | 1.000 | 1.000 | 1/1 | 0.0 |
| phasegate4to0 | 4 | 0 | 0.941 | 1.015 | 0/1 | 1798.2 |
| phasegate4to1 | 4 | 1 | 1.060 | 1.022 | 0/1 | 1907.7 |
| phasegate4to2 | 4 | 2 | 1.224 | 0.994 | 0/1 | 2021.8 |

## Frozen smoke selection

- No selection: the smoke data did not yield both feasible policy classes.

## Held-out smoke

Not run. The handoff requested implementation and smoke validation first.

## Interpretation

No scheduler benefit is claimed from smoke-scale timings. The primary next step is three valid calibration repeats per candidate, followed by freezing the two winners and five paired held-out repeats on new prompt and HNSW traces.
