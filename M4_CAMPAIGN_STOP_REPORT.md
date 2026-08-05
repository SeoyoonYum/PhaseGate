# Apple M4 PhaseGate Campaign Stop Report

## Outcome

The campaign stopped before policy calibration because two independent five-run isolated-baseline sets failed the preregistered ±3% run-level p95 stability gate. No calibration policy, primary SLO budget, TimeGate schedule, held-out result, output-length sweep, causal phase-alignment claim, or paper replacement result was produced.

This is a validity stop, not an unfavorable-result exclusion. All completed runs and failed pre-measurement attempts remain preserved under `experiments/static_phaseaware/fanmac_m4_causal_shape_20260805_r4/`.

## Tested configuration

- Apple M4 Mac mini (Mac16,10), 4 performance + 6 efficiency CPU cores, 10 GPU cores, 16 GB unified memory
- macOS 26.5 (25F71)
- Python 3.13.14; MLX 0.31.2; MLX-LM 0.31.3; NumPy 2.4.6; FAISS 1.14.3
- `mlx-community/Qwen2.5-1.5B-Instruct-4bit` revision `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`
- HNSW 100,000 × 384, M=32, efConstruction=80, SHA-256 `4c65bde676235523dbba2f1dc78a44de3f447470d38105488586d7ca486a51f0`
- MLX memory limit 5.5 GB; FAISS internal OpenMP threads fixed at 1
- Collection commit `cd42e9669b555953e540f052a3544483f84449b6`

## CPU-only scaling

All six 60-second blocks were valid, pageout-free, and swap-free. Observed active-concurrency p95 equaled the requested cap.

| Cap | Valid repeats | Median QPS | Median query p50 | Median query p95 | Median CPU | Median RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 1005.54 | 0.963 ms | 1.217 ms | 103.34% | 279,265,280 B |
| 2 | 3 | 1764.83 | 1.089 ms | 1.367 ms | 201.27% | 282,116,096 B |

Cap 2 reached the maximum median QPS and was frozen as K_hi. Cap 4 was not allowed because the preregistered rule reserves two of the M4's four performance cores.

## Mechanism blocks

All nine 100-request blocks were valid on their first attempt, with zero pageouts and zero swap growth.

| Policy | Valid repeats | Median p95 TPOT | Median p95 TTFT | Median retrieval QPS |
|---|---:|---:|---:|---:|
| LLM-only | 3 | 15.486 ms | 1809.11 ms | 0 |
| Fixed-1 | 3 | 14.836 ms | 1811.61 ms | 779.44 |
| Fixed-2 | 3 | 16.789 ms | 1831.27 ms | 1369.25 |

These values must not be normalized or interpreted as definitive PREFILL-versus-DECODE sensitivity because the subsequent isolated baseline stability gate failed. In particular, Fixed-1's lower p95 TPOT than the contemporaneous LLM-only median demonstrates why the unstable tail cannot support a slowdown claim.

## Baseline stop evidence

The first five-run set had median p95 TPOT 14.889 ms and median p95 TTFT 1806.32 ms. TPOT deviations were +5.5%, 0.0%, -0.6%, -0.8%, and +7.1%; TTFT deviations were all within ±2.3%.

A single fresh revalidation set used new seeds after environment checks found no swap, pageout, memory-pressure, within-block drift, or sentinel instability. Its median p95 TPOT was 15.353 ms and median p95 TTFT was 1806.18 ms. TPOT deviations were +1.7%, -0.2%, -13.1%, 0.0%, and +1.8%; TTFT deviations were all within ±0.7%.

Both sets therefore failed the symmetric ±3% gate. A third set was not attempted.

## Answers to the requested primary questions

1. PhaseGate versus duty-cycle-matched TimeGate: not evaluated; calibration was never validly reached.
2. Effect of output length: not evaluated.
3. Device-specific cap: CPU-only K_hi=2; cap 4 was disallowed on this base M4.
4. Reproduction without memory confounding: completed CPU, mechanism, and baseline blocks were pageout-free and swap-free, but no primary held-out result exists.

## Paper implications

No primary paper claim should be replaced, strengthened, or weakened from this stopped campaign. It is valid to report only that the base-M4 attempt established CPU-only K_hi=2 and encountered irreproducible run-level TPOT p95 tails despite clean memory and sentinel telemetry.

Statements that must not be made:

- PhaseGate beats or matches TimeGate on this M4.
- Phase alignment is causal on this M4.
- PhaseGate gain changes monotonically with output length on this M4.
- A held-out primary result was pageout-free.
- Cap 4 was tested or found safe under the preregistered M4 core-reservation rule.
