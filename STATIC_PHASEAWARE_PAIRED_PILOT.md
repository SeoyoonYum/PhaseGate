# Stability-First Static Phase-Awareness Paired Pilot

This is a fanless M4 Air go/no-go pilot, not final policy selection or held-out evaluation.

## Setup and validity controls

- Qwen2.5 1.5B 4-bit, context 2048, 128 generated tokens, four LLM requests per block.
- Always-backlogged FAISS-HNSW retrieval; queue non-empty fraction must be at least 0.95.
- Separate sessions and isolated baselines for the cap-1 and cap-2 comparisons; randomized paired order.
- Fresh process for every policy block, 6 GB MLX memory limit, resident-memory preflight, and browser-process preflight.
- A block was accepted only with pageout delta 0, swap-used delta 0, and sentinel recovery to within ±5% before continuing.

## Per-block results

| Comparison | Repeat | Order | Policy | Retrieval QPS | p95 TPOT (norm) | p95 TTFT (norm) | Joint SLO | Queue non-empty |
|---|---:|---|---|---:|---:|---:|---:|---:|
| fixed1_vs_phasegate4to1 | 1 | fixed1 then phasegate4to1 | fixed1 | 786.6 | 16.89 (1.157) | 2118.1 (1.039) | False | 1.000 |
| fixed1_vs_phasegate4to1 | 1 | fixed1 then phasegate4to1 | phasegate4to1 | 1507.3 | 16.78 (1.149) | 2172.7 (1.065) | False | 1.000 |
| fixed1_vs_phasegate4to1 | 2 | phasegate4to1 then fixed1 | fixed1 | 787.3 | 16.64 (1.140) | 2128.9 (1.044) | False | 1.000 |
| fixed1_vs_phasegate4to1 | 2 | phasegate4to1 then fixed1 | phasegate4to1 | 1506.7 | 16.74 (1.146) | 2172.0 (1.065) | False | 1.000 |
| fixed1_vs_phasegate4to1 | 3 | fixed1 then phasegate4to1 | fixed1 | 786.1 | 16.72 (1.145) | 2102.2 (1.031) | False | 1.000 |
| fixed1_vs_phasegate4to1 | 3 | fixed1 then phasegate4to1 | phasegate4to1 | 1506.4 | 16.75 (1.147) | 2179.9 (1.069) | False | 1.000 |
| fixed2_vs_phasegate4to2 | 1 | fixed2 then phasegate4to2 | fixed2 | 1334.1 | 18.81 (1.311) | 2144.5 (1.051) | False | 1.000 |
| fixed2_vs_phasegate4to2 | 1 | fixed2 then phasegate4to2 | phasegate4to2 | 1687.6 | 19.02 (1.326) | 2176.2 (1.066) | False | 1.000 |
| fixed2_vs_phasegate4to2 | 2 | fixed2 then phasegate4to2 | fixed2 | 1328.9 | 19.14 (1.334) | 2154.7 (1.056) | False | 1.000 |
| fixed2_vs_phasegate4to2 | 2 | fixed2 then phasegate4to2 | phasegate4to2 | 1689.3 | 18.96 (1.322) | 2180.2 (1.068) | False | 1.000 |
| fixed2_vs_phasegate4to2 | 3 | phasegate4to2 then fixed2 | phasegate4to2 | 1690.5 | 18.91 (1.318) | 2193.1 (1.075) | False | 1.000 |
| fixed2_vs_phasegate4to2 | 3 | phasegate4to2 then fixed2 | fixed2 | 1334.5 | 19.07 (1.330) | 2154.9 (1.056) | False | 1.000 |

## Go/no-go

- fixed1_vs_phasegate4to1: **NO-GO**; median QPS gain +91.6%, PhaseGate higher in 3/3 pairs, joint-SLO +5% qualifying in 0/3 pairs. Failed/contaminated pair attempts preserved: 8.
- fixed2_vs_phasegate4to2: **NO-GO**; median QPS gain +26.7%, PhaseGate higher in 3/3 pairs, joint-SLO +5% qualifying in 0/3 pairs. Failed/contaminated pair attempts preserved: 1.

## Mechanism and stability detail

| Comparison | Rep | Policy | Active workers P/D | Overshoot | Converge ms | Completions P/D | Sentinel pre / post initial→recovered | Pageout / swap | TPOT slope | QPS slope |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed1_vs_phasegate4to1 | 1 | fixed1 | 1.00 / 1.00 | 0.00% | 13.3 | 7120 / 4880 | +1.2% / +5.2%→+0.9% | 0 / 0 B | -0.026 ms/s | -3.83 QPS/s |
| fixed1_vs_phasegate4to1 | 1 | phasegate4to1 | 4.00 / 1.03 | 1.34% | 41.4 | 18112 / 5088 | +0.9% / +6.5%→+1.0% | 0 / 0 B | -0.009 ms/s | -24.35 QPS/s |
| fixed1_vs_phasegate4to1 | 2 | fixed1 | 1.00 / 1.00 | 0.00% | 5.7 | 7232 / 4896 | +1.5% / +6.1%→+1.2% | 0 / 0 B | -0.004 ms/s | -3.45 QPS/s |
| fixed1_vs_phasegate4to1 | 2 | phasegate4to1 | 3.99 / 1.03 | 1.39% | 27.7 | 18192 / 5136 | +1.3% / +7.1%→+1.0% | 0 / 0 B | +0.022 ms/s | -24.35 QPS/s |
| fixed1_vs_phasegate4to1 | 3 | fixed1 | 1.00 / 1.00 | 0.00% | 20.2 | 7104 / 4912 | +0.5% / +4.9%→+4.9% | 0 / 0 B | -0.006 ms/s | -3.58 QPS/s |
| fixed1_vs_phasegate4to1 | 3 | phasegate4to1 | 3.99 / 1.02 | 0.85% | 37.5 | 18320 / 5088 | +1.5% / +6.7%→+1.0% | 0 / 0 B | +0.030 ms/s | -24.24 QPS/s |
| fixed2_vs_phasegate4to2 | 1 | fixed2 | 2.00 / 2.00 | 0.00% | 3.7 | 12448 / 9440 | +1.2% / +6.4%→+1.0% | 0 / 0 B | -0.006 ms/s | -6.30 QPS/s |
| fixed2_vs_phasegate4to2 | 1 | phasegate4to2 | 3.99 / 2.02 | 1.06% | 30.8 | 18288 / 9632 | +1.3% / +7.3%→+1.2% | 0 / 0 B | +0.016 ms/s | -16.15 QPS/s |
| fixed2_vs_phasegate4to2 | 2 | fixed2 | 2.00 / 2.00 | 0.00% | 4.9 | 12528 / 9408 | +1.5% / +7.4%→+1.5% | 0 / 0 B | -0.011 ms/s | -5.99 QPS/s |
| fixed2_vs_phasegate4to2 | 2 | phasegate4to2 | 4.00 / 2.01 | 1.10% | 30.4 | 18352 / 9552 | +1.7% / +7.0%→+1.4% | 0 / 0 B | -0.007 ms/s | -16.62 QPS/s |
| fixed2_vs_phasegate4to2 | 3 | phasegate4to2 | 4.00 / 2.00 | 0.46% | 29.6 | 18384 / 9440 | +1.2% / +7.0%→+1.3% | 0 / 0 B | -0.011 ms/s | -15.94 QPS/s |
| fixed2_vs_phasegate4to2 | 3 | fixed2 | 2.00 / 2.00 | 0.00% | 3.7 | 12496 / 9488 | +1.4% / +7.2%→+1.4% | 0 / 0 B | +0.024 ms/s | -5.83 QPS/s |

All 12 accepted blocks had queue non-empty fraction 1.000, pageout delta 0, and swap-used delta 0. 16 invalid policy blocks remain in `invalid_policy_blocks.csv` and the raw JSONL; they were not relabeled or included in the primary comparison.

Immediate post-sentinels often exceeded the ±5% gate, so the runner cooled and retried until recovery before starting the next block. The QPS slopes are diagnostics only and were not used to accept or reject a run.

No final best policy was selected and held-out evaluation was not run.
