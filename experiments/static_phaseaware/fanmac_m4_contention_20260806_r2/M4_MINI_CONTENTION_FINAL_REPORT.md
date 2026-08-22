# M4 Mini Contention Final Report

## Scope and frozen interpretation

This is a diagnostic contention campaign on the fan-cooled base-M4 Mac mini. It contains no calibration, PhaseGate, TimeGate, held-out policy selection, or output-length sweep. `K_hi=2` remains frozen from the source handoff; cap 4 is diagnostic-only regardless of outcome.

The harness used event-driven phase timing, <=1 Hz native process-memory monitoring, and no legacy 5 ms sampler or repeated `ps` subprocesses. All policies used context 2,048, output 128, 300 measured requests, and newly frozen r2 prompt/query seeds. Each repeat shared its trace across LLM-only and Fixed caps 1/2/4, with order frozen before execution and a fixed 10-second inter-block cooldown.

## Baseline stability gate

Set A ran first and failed the frozen gate. Its median p95 TPOT was 12.615 ms and median p95 TTFT was 1803.79 ms. Repeat 0 exceeded the TPOT gate at 5.12% absolute deviation; the other four TPOT values and all five TTFT values were within +/-3%.

### Set A (failed, retained)

| Repeat | p95 TPOT (ms) | abs dev. | p95 TTFT (ms) | abs dev. | pageouts | swap bytes | within +/-3% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13.261 | 5.12% | 1799.55 | 0.24% | 0 | 0 | False |
| 1 | 12.615 | 0.00% | 1805.03 | 0.07% | 8 | 0 | True |
| 2 | 12.811 | 1.55% | 1803.79 | 0.00% | 33 | 0 | True |
| 3 | 12.579 | 0.29% | 1804.49 | 0.04% | 57 | 0 | True |
| 4 | 12.342 | 2.16% | 1802.63 | 0.06% | 2 | 0 | True |

The single pre-frozen environmental correction was a 120-second quiet idle/thermal cooldown. It produced zero pageout growth during the correction interval and changed neither the workload nor observer. Set B was then run once and passed. Its median p95 TPOT was 12.600 ms and median p95 TTFT was 1804.16 ms; only Set B defines the official normalization baseline.

### Set B (passed, official normalization)

| Repeat | p95 TPOT (ms) | abs dev. | p95 TTFT (ms) | abs dev. | pageouts | swap bytes | within +/-3% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12.711 | 0.88% | 1783.25 | 1.16% | 31 | 0 | True |
| 1 | 12.257 | 2.72% | 1804.36 | 0.01% | 5 | 0 | True |
| 2 | 12.600 | 0.00% | 1803.65 | 0.03% | 1 | 0 | True |
| 3 | 12.316 | 2.26% | 1804.16 | 0.00% | 2 | 0 | True |
| 4 | 12.838 | 1.88% | 1804.63 | 0.03% | 4 | 0 | True |

| Set | passed | median p95 TPOT | median p95 TTFT | pageout flags | thermal-clean blocks |
|---|---:|---:|---:|---:|---:|
| A | False | 12.615 | 1803.79 | 4/5 | 5/5 |
| B | True | 12.600 | 1804.16 | 5/5 | 5/5 |

All ten baseline blocks had zero swap growth, normal memory pressure, exact 300-request/38,400-token accounting, zero observer subprocesses, and clean pre/post block-boundary thermal status. Positive global pageout was retained as a soft flag in 4/5 Set A blocks and 5/5 Set B blocks; it was never used as a rerun reason.

## Contention results

| Condition | median p95 TTFT | normalized TTFT | median p95 TPOT | normalized TPOT | retrieval QPS | pageout flags |
|---|---:|---:|---:|---:|---:|---:|
| llm-only | 1774.05 | 0.983x | 12.440 | 0.987x | 0.0 | 0/3 |
| fixed1 | 1808.77 | 1.003x | 13.966 | 1.108x | 782.7 | 0/3 |
| fixed2 | 1826.83 | 1.013x | 16.444 | 1.305x | 1376.8 | 1/3 |
| fixed4 | 1929.29 | 1.069x | 20.130 | 1.598x | 2134.9 | 0/3 |

Every repeat is included in `m4_mini_contention_per_run.csv`; medians are descriptive across three randomized repeats. Decode sensitivity exceeded prefill sensitivity at every positive cap: **True**. Cap 4 remains diagnostic-only and does not change `K_hi=2`.

Official normalized TTFT and TPOT use the passed five-run Set B medians, as required by the r2 handoff. Supplementary within-repeat normalization against each contention-matrix LLM-only block is also reported. The `FROZEN_EXECUTION_MATRIX.json` metadata says within-repeat normalization; that line is preserved unchanged as frozen evidence, while the higher-precedence r2 protocol governs the official endpoint.

### Every contention repeat

| Repeat | condition | p95 TTFT | official norm. | paired norm. | p95 TPOT | official norm. | paired norm. | QPS | pageouts | swap bytes | thermal clean | validity |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | llm-only | 1772.59 | 0.983x | 1.000x | 12.468 | 0.989x | 1.000x | 0.0 | 0 | 0 | True | valid |
| 0 | fixed1 | 1808.77 | 1.003x | 1.020x | 13.941 | 1.106x | 1.118x | 780.4 | 0 | 0 | True | valid |
| 0 | fixed2 | 1826.52 | 1.012x | 1.030x | 16.764 | 1.330x | 1.345x | 1376.8 | 10 | 0 | True | valid |
| 0 | fixed4 | 1929.29 | 1.069x | 1.088x | 20.192 | 1.602x | 1.620x | 2134.3 | 0 | 0 | True | valid |
| 1 | llm-only | 1774.05 | 0.983x | 1.000x | 12.321 | 0.978x | 1.000x | 0.0 | 0 | 0 | True | valid |
| 1 | fixed1 | 1808.81 | 1.003x | 1.020x | 14.170 | 1.125x | 1.150x | 783.5 | 0 | 0 | True | valid |
| 1 | fixed2 | 1826.83 | 1.013x | 1.030x | 15.910 | 1.263x | 1.291x | 1378.0 | 0 | 0 | True | valid |
| 1 | fixed4 | 1928.13 | 1.069x | 1.087x | 20.004 | 1.588x | 1.624x | 2134.9 | 0 | 0 | True | valid |
| 2 | llm-only | 1794.18 | 0.994x | 1.000x | 12.440 | 0.987x | 1.000x | 0.0 | 0 | 0 | True | valid |
| 2 | fixed1 | 1808.25 | 1.002x | 1.008x | 13.966 | 1.108x | 1.123x | 782.7 | 0 | 0 | True | valid |
| 2 | fixed2 | 1876.98 | 1.040x | 1.046x | 16.444 | 1.305x | 1.322x | 1374.5 | 0 | 0 | True | valid |
| 2 | fixed4 | 1940.56 | 1.076x | 1.082x | 20.130 | 1.598x | 1.618x | 2135.6 | 0 | 0 | True | valid |

The official median curve was: cap 1, TTFT +0.3% and TPOT +10.8%; cap 2, TTFT +1.3% and TPOT +30.5%; cap 4, TTFT +6.9% and TPOT +59.8%. Retrieval QPS increased from 782.7 to 1376.8 to 2134.9. Thus TPOT worsened more than TTFT at every tested positive cap, and the asymmetry grew with cap.

| Condition | inter-token p99 (ms) | request maximum-gap p95 (ms) | phase-transition gap p95 (ms) |
|---|---:|---:|---:|
| llm-only | 12.473 | 13.781 | 15.513 |
| fixed1 | 14.243 | 15.088 | 18.022 |
| fixed2 | 16.136 | 17.204 | 19.693 |
| fixed4 | 20.404 | 21.968 | 24.371 |

## Direction-only comparison with the M4 MacBook Air

| Condition | Air normalized prefill/TTFT proxy | Air normalized decode/TPOT proxy | Mini normalized TTFT | Mini normalized TPOT |
|---|---:|---:|---:|---:|
| llm-only | 1.000x | 1.000x | 0.983x | 0.987x |
| fixed1 | 1.027x | 1.040x | 1.003x | 1.108x |
| fixed2 | 1.043x | 1.194x | 1.013x | 1.305x |
| fixed4 | 1.057x | 1.613x | 1.069x | 1.598x |

The Air values are the previously reported normalized phase-contention curve in `PHASEGUARD_RESULTS.md` (0/1/2/4 HNSW workers). The comparison is limited to normalized curve direction; raw retrieval QPS is not compared across devices. Any difference in magnitude is not attributed solely to cooling because device form factor, scheduling, and harness details differ.

The Air cap-4 reference corresponds to decode +61.3% and prefill +5.7%. The Mini cap-4 official result was decode +59.8% and prefill +6.9%, so it has the same asymmetric direction. The Mini result is compared only for direction and normalized curve shape.

## Zero-pageout sensitivity

The official result retains every valid block. This sensitivity view filters the contention blocks to zero global pageout growth; `NA` means that a condition had no zero-pageout block. All five official Set B baseline blocks had small positive global pageout deltas, so the sensitivity values necessarily retain the frozen Set B normalization denominator and are not an end-to-end pageout-free baseline comparison. No claim of a wholly pageout-free campaign is made.

| Condition | zero-pageout n/all | normalized TTFT | normalized TPOT | retrieval QPS |
|---|---:|---:|---:|---:|
| llm-only | 3/3 | 0.983 | 0.987 | 0.000 |
| fixed1 | 3/3 | 1.003 | 1.108 | 782.676 |
| fixed2 | 2/3 | 1.026 | 1.284 | 1376.239 |
| fixed4 | 3/3 | 1.069 | 1.598 | 2134.892 |

## Validity, memory, and thermal audit

- Valid contention blocks: 12/12; invalid attempts preserved: 0.
- Swap-growth blocks: 0/12.
- Positive-pageout blocks: 1/12; pageout deltas are retained per run rather than used for result selection.
- Clean pre/post thermal status: 12/12 blocks.
- All production blocks used observer mode `event`, launched zero observer subprocesses, and retained exact 300-request/38,400-token and query/event accounting.
- Fixed-1/2/4 had phase-specific active-worker p95 equal to 1/2/4 in every repeat; one shared FAISS index was loaded once per retrieval block, FAISS OpenMP was frozen at one, and the queue-nonempty fraction was 1.0.
- Event timelines include a deliberate post-measurement IDLE drain at cap 4 so feeder threads can terminate. Completion-audit cap checks are restricted to the measured request interval; no measured Fixed-1/2 admission exceeded its fixed cap.

No valid block was rerun because of an unfavorable number. Cap 4 is not a PhaseGate candidate, and this campaign cannot revise calibration or K_hi.
