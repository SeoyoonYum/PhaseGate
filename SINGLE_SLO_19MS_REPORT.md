# Single-SLO 19 ms CPU Throughput Feasibility Check

Fanless M4 Air preliminary evidence only; not a paper result.
Recorded 24 attempts: 15 valid clean runs; 7 older attempts observed a nonzero experiment-window pageout delta.

- Selected cap: 2
- Gain vs cap1: 12.3%
- Retained uncoordinated throughput: 85.0%
- TPOT change vs uncoordinated: -25.7%
- CPU-only 1-to-3/4 worker gain: 169.9%
- Go/no-go: strong positive

## Summary

| Setting | p95 TPOT | 19ms pass | Total retrieval QPS | Gain vs cap1 | % of uncoord |
|---|---:|---:|---:|---:|---:|
| PhaseGuard cap1 | 16.38 ms | 3/3 | 1495.9 | baseline | 75.7% |
| PhaseGuard cap2 | 18.13 ms | 3/3 | 1679.7 | +12.3% | 85.0% |
| PhaseGuard cap3 | 21.35 ms | 0/3 | 1842.9 | +23.2% | 93.2% |
| PhaseGuard cap4 | 24.22 ms | 0/3 | 1976.1 | +32.1% | 100.0% |
| Uncoordinated | 24.40 ms | 0/3 | 1976.9 | +32.2% | 100.0% |

All 15 current-campaign repeats had queue-nonempty fraction 1.000, zero experiment-window pageouts, zero swap-used growth, and passed the thermal gate.
Older failed/contaminated attempts remain preserved in the raw JSONL and are excluded from the current-campaign medians.
