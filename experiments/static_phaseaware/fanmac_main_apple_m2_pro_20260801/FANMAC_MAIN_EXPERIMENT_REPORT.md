No—on held-out traces, Claim A could not compare a best static phase-aware policy with a best fixed policy because calibration produced no strict-SLO-feasible candidate in either class; held-out evaluation therefore tested only the prespecified latency-matched pairs and references, without changing the SLO or policy grid.

# Fan-Cooled Mac Static Phase-Aware Main Experiment

## 1. Hardware and stability

- Campaign: `fanmac_main_apple_m2_pro_20260801`
- Invalid attempts preserved: 68
- Host: Mac mini (Apple M2 Pro), 6P+4E CPU cores, 16 GPU cores, 16 GiB unified memory
- Runtime: Python 3.12.13; macOS metadata and pinned package versions are in `environment.txt`
- Accepted blocks: 5 isolated baseline + 30 characterization + 45 calibration + 30 evaluation
- Accepted policy blocks with pageout/swap contamination: 0
- Invalid attempts with pageout/swap evidence: 64

## 2. Mechanism

| Workers | Prefill slowdown | Decode slowdown | Prefill retrieval QPS | Decode retrieval QPS |
|---:|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 0.0 | 0.0 |
| 1 | 0.999 | 1.052 | 1012.2 | 889.4 |
| 2 | 1.002 | 1.150 | 1941.6 | 1680.8 |
| 3 | 1.004 | 1.206 | 2598.0 | 2263.6 |
| 4 | 1.003 | 1.262 | 3258.2 | 2781.1 |

Decode slowdown rises to 1.262× at four workers while prefill remains near 1.003×, confirming the phase-asymmetry mechanism on this fan-cooled host.

## 3. Calibration

| Policy | Prefill cap | Decode cap | TPOT norm | TTFT norm | Joint pass | Retrieval QPS |
|---|---:|---:|---:|---:|---:|---:|
| fixed0 | 0 | 0 | 1.249 | 1.010 | 0/3 | 63.0 |
| fixed1 | 1 | 1 | 1.214 | 1.003 | 0/3 | 938.9 |
| fixed2 | 2 | 2 | 1.392 | 1.006 | 0/3 | 1740.9 |
| fixed3 | 3 | 3 | 1.473 | 1.009 | 0/3 | 2340.5 |
| fixed4 | 4 | 4 | 1.625 | 1.011 | 0/3 | 2869.9 |
| phasegate1to0 | 1 | 0 | 1.216 | 1.002 | 0/3 | 623.6 |
| phasegate2to0 | 2 | 0 | 1.203 | 1.006 | 0/3 | 1115.2 |
| phasegate2to1 | 2 | 1 | 1.197 | 1.006 | 0/3 | 1486.5 |
| phasegate3to0 | 3 | 0 | 1.222 | 1.009 | 0/3 | 1488.8 |
| phasegate3to1 | 3 | 1 | 1.212 | 1.009 | 0/3 | 1849.9 |
| phasegate3to2 | 3 | 2 | 1.399 | 1.008 | 0/3 | 2115.8 |
| phasegate4to0 | 4 | 0 | 1.208 | 1.013 | 0/3 | 1801.3 |
| phasegate4to1 | 4 | 1 | 1.225 | 1.013 | 0/3 | 2165.0 |
| phasegate4to2 | 4 | 2 | 1.400 | 1.013 | 0/3 | 2416.4 |
| phasegate4to3 | 4 | 3 | 1.450 | 1.011 | 0/3 | 2660.6 |

Frozen best Fixed: `none`. Frozen best PhaseGate: `none`.

No calibration policy passed the strict joint SLO in all three valid repeats; therefore Claim A has no feasible operating point to evaluate.

## 4. Held-out primary evaluation

| Policy | TPOT norm | TTFT norm | Joint SLO pass | Retrieval QPS | Gain |
|---|---:|---:|---:|---:|---:|
| llm-only | 1.000 | 1.000 | 5/5 | 0.0 | — |
| fixed4 | 1.616 | 1.010 | 0/5 | 2879.6 | — |

## 5. Latency-matched evaluation

| Pair | Fixed QPS | Gate QPS | Paired gain (95% CI) | TPOT ratio | TTFT ratio | QPS wins |
|---|---:|---:|---:|---:|---:|---:|
| fixed1 vs phasegate4to1 | 942.5 | 2174.1 | +130.6% [+129.9%, +131.8%] | 0.990 | 1.008 | 5/5 |
| fixed2 vs phasegate4to2 | 1746.2 | 2428.1 | +39.0% [+38.7%, +39.2%] | 0.995 | 1.001 | 5/5 |

Both pairs satisfy the prespecified held-out latency-matched Claim B criteria (latency ratios ≤1.03, QPS gain ≥5%, and 5/5 QPS wins), but neither operating point satisfies the absolute 1.10× TPOT SLO.

## 6. Execution mechanism

- Fixed-1 vs 4→1: prefill active workers 1.00 vs 4.00; decode active workers 1.00 vs 1.03; prefill completions 5648 vs 17408.
- Fixed-2 vs 4→2: prefill active workers 2.00 vs 4.00; decode active workers 2.00 vs 2.02; prefill completions 10544 vs 17408.

The timeline figure records phase, configured permits, actual workers, and cumulative completions.

## 7. Limitations

This is one fan-cooled Mac, one model/workload, an always-backlogged retrieval workload, and static policies only. Cross-hardware and workload validation remains necessary.
