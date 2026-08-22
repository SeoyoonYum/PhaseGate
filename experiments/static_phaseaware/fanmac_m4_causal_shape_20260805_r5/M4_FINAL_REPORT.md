# Base-M4 PhaseGate r5 Final Report

This report uses only frozen, valid r5 measurements. Diagnostic smoke runs, invalid attempts, and the immutable r4 campaign are excluded from policy selection and confidence intervals.

## 1. Exact device configuration

The campaign ran on a fan-cooled Mac mini (Mac16,10, MU9D3FN/A) with an Apple M4, 4 performance cores, 6 efficiency cores (10 total), 10 GPU cores, and 16 GB unified memory. The OS was macOS 26.5 build 25F71. Python was 3.13.14; package versions were MLX 0.31.2, MLX-LM 0.31.3, NumPy 2.4.6, and FAISS 1.14.3. The model was mlx-community/Qwen2.5-1.5B-Instruct-4bit at revision `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`, 4-bit, with a 5.5 GB MLX limit. The shared read-only HNSW index contained 100,000 vectors of dimension 384, M=32, efConstruction=80, SHA-256 `4c65bde676235523dbba2f1dc78a44de3f447470d38105488586d7ca486a51f0`. This is a base M4, not an M4 Pro.

## 2. CPU-only scaling and K_hi

Three 60-second repeats were run at each cap. Median throughput was cap 1: 976.0 QPS, cap 2: 1751.8 QPS, cap 4: 2464.2 QPS. The frozen rule selected the smallest cap reaching at least 90% of the maximum median QPS; therefore `K_hi=4`. Observed active-concurrency p95 exactly matched each requested cap, and cap 4 had no CPU-scaling swap or pageout flag. On this device, cap 4 was justified by the preregistered CPU-only rule.

## 3. PREFILL versus DECODE sensitivity

The mechanism blocks showed substantially greater TPOT than TTFT slowdown as fixed retrieval concurrency increased:

| Policy | Retrieval QPS | normalized p95 TTFT | normalized p95 TPOT |
|---|---:|---:|---:|
| llm-only | 0.0 | 1.000x | 1.000x |
| fixed1 | 784.8 | 1.005x | 1.189x |
| fixed2 | 1379.0 | 1.015x | 1.293x |
| fixed4 | 2137.2 | 1.073x | 1.650x |

At cap 4, median p95 TPOT increased to 1.650x isolated while p95 TTFT increased to 1.073x. Thus DECODE was more sensitive than PREFILL on this base-M4 system under the measured workload. These are matched metrics within r5; they are not equated with older M2-Pro metrics.

## 4. Frozen policies at primary_B

The fresh five-repeat baseline was stable under the predeclared +/-3% gate. Calibration froze `primary_B=1.25`, `fixed1`, and `phasegate4to1` before held-out execution. The matched phase-blind control was `timegate4to1` with a calibration-derived wall-clock replay schedule.

## 5. PhaseGate versus matched TimeGate

| Policy | median retrieval QPS | median normalized p95 TPOT | median normalized p95 TTFT | joint SLO passes |
|---|---:|---:|---:|---:|
| fixed1 | 783.7 | 1.165 | 1.006 | 7/7 |
| phasegate4to1 | 1574.2 | 1.181 | 1.046 | 7/7 |
| timegate4to1 | 1537.6 | 1.603 | 1.043 | 0/7 |

Across seven paired randomized repeats, PhaseGate exceeded Fixed by a median 100.88% (95% run-level paired bootstrap interval 100.81% to 101.07%). It exceeded matched TimeGate by 2.36% (1.77% to 2.57%). PhaseGate passed the joint 1.25x SLO in 7/7 blocks; TimeGate passed 0/7, with median normalized p95 TPOT 1.603. The result supports phase alignment beyond burstiness: the phase-blind schedule delivered similar raw retrieval throughput but exposed DECODE to high concurrency and violated the latency budget.

## 6. TimeGate overlap with actual phases

PhaseGate spent a median 100.00% of actual PREFILL and only 0.003% of actual DECODE at cap 4. TimeGate spent 80.04% of PREFILL and 23.11% of DECODE at cap 4. Runtime audit fields confirm that TimeGate did not consult phase state when selecting its cap. Therefore TimeGate's high-cap intervals occurred materially during actual DECODE, as intended for the causal control.

## 7. Output-length dependence

| Output tokens | PhaseGate PREFILL fraction | Fixed QPS | PhaseGate QPS | median paired QPS gain [95% interval] |
|---:|---:|---:|---:|---:|
| 64 | 69.08% | 811.0 | 1869.5 | 130.39% [130.15, 131.19] |
| 128 | 51.60% | 782.1 | 1573.3 | 101.37% [101.10, 101.59] |
| 512 | 20.65% | 726.0 | 1041.0 | 43.29% [43.22, 43.69] |

The PhaseGate-versus-Fixed gain decreased monotonically from 64 to 512 output tokens. Policies remained fixed from the 128-token calibration; no length-specific recalibration was performed. This supports dependence on available PREFILL overlap time within the measured lengths and does not justify extrapolation beyond 64--512 tokens.

## 8. Association with measured PREFILL fraction

Across the 15 paired output-shape observations, the descriptive Pearson association between PhaseGate PREFILL fraction and paired QPS gain was r=0.9996, and rank association was 0.896. At the three length-level medians the ordering was perfectly monotonic. Because output length jointly changes phase fraction and other runtime behavior, this is descriptive mechanism evidence, not an independent causal estimate of phase fraction.

## 9. Memory and pageout audit

All 21 primary held-out blocks had zero swap growth and normal memory pressure. Only 1/7 paired triplets were completely pageout-free. Per-policy pageout deltas were Fixed median/range 0/0--20, PhaseGate 0/0--2, and TimeGate 0/0--4. The descriptive Spearman association between PhaseGate-minus-Fixed pageout imbalance and paired QPS gain was -0.286 at n=7. Excluding repeat 2, which had the largest absolute imbalance, yielded a median gain of 100.90%; leave-one-pair-out medians ranged from 100.85% to 100.93%. This robustness check does not prove absence of confounding. The campaign reproduces the result without swap or memory-pressure warnings, but it does **not** establish broadly pageout-free execution.

## 10. Paper-claim disposition

- Strengthen the device-specific claim that decode contention is more severe than prefill contention, using the matched base-M4 mechanism values above.
- Replace the old primary result with the frozen base-M4 Fixed/PhaseGate/TimeGate triplet and its run-level paired intervals.
- Strengthen the phase-alignment interpretation only in the bounded form supported by TimeGate's decode overlap, 0/7 SLO passes, and PhaseGate's 2.36% median QPS advantage.
- Weaken any universal throughput-gain statement: the observed gain fell from 130.39% at 64 tokens to 43.29% at 512 tokens.
- Do not claim pageout neutrality or a fully pageout-free primary campaign; only 1/7 triplets were completely pageout-free.
- Move the older flagged M2-Pro evidence to supporting context rather than numerically merging it with this campaign.
- Remove any M4-Pro label from these results and do not compare raw QPS across devices.

## Audit limitation

The 512-token isolated baseline p95 TPOT values were 17.018, 13.124, 17.574 ms, showing tail variability. The sweep protocol did not preregister a +/-3% stability gate for these three per-length baselines, so the valid blocks were not selectively rerun. Consequently, normalized 512-token TPOT values below 1 must not be interpreted as retrieval improving decode latency; the paired retrieval-QPS shape result and event-measured phase fractions are the defensible conclusions.
