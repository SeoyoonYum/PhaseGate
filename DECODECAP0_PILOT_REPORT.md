# Decode-Cap-0 Strict-SLO Pilot

Yes. Under the strict 1.10× TPOT and TTFT SLO, prefill-only CPU execution did provide useful retrieval goodput beyond Fixed-0 in this fanless-M4 pilot.

This is a focused paired go/no-go experiment: no calibration matrix, held-out evaluation, adaptive controller, or decode-cap-positive retest was run.

| Policy | TPOT norm | TTFT norm | Joint SLO | Retrieval QPS | Additional QPS vs Fixed-0 |
|---|---:|---:|---:|---:|---:|
| fixed0 | 1.038 | 1.013 | 9/9 | 41.9 | baseline |
| phasegate1to0 | 1.026 | 1.031 | 3/3 | 527.5 | +483.7 |
| phasegate2to0 | 0.987 | 1.048 | 3/3 | 886.1 | +845.3 |
| phasegate4to0 | 1.035 | 1.060 | 3/3 | 1259.5 | +1218.7 |

## Interpretation

- phasegate1to0: feasible=True; decode admission zero=True; paired median additional QPS=+483.7; persistent negative QPS slope thermal-risk flag=True.
- phasegate2to0: feasible=True; decode admission zero=True; paired median additional QPS=+845.3; persistent negative QPS slope thermal-risk flag=True.
- phasegate4to0: feasible=True; decode admission zero=True; paired median additional QPS=+1218.7; persistent negative QPS slope thermal-risk flag=True.

## Separation from the prior decode-cap-positive evidence

1. Latency-matched: the prior `4→1` and `4→2` policies improved retrieval QPS at nearly identical TPOT relative to their Fixed counterparts, but both operating points were outside the strict 1.10× TPOT SLO.
2. Strict-SLO: this report evaluates only decode cap zero and does not combine its result with the decode-cap-positive comparison.
