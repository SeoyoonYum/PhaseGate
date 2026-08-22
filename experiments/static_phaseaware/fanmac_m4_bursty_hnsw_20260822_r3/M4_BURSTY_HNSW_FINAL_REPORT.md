# Base-M4 Bursty HNSW Demand Sweep Final Report

This is a synthetic workload-sensitivity study, not an end-to-end agent benchmark or a claim about typical-user arrival rates.

## Frozen scope

Fixed-1, PhaseGate 4->1, TimeGate 4/1, B=1.25, 2,048/128 tokens, one shared 100k-vector HNSW index, five matched repeats per duty.

## Results

| Duty | PhaseGate QPS | Fixed QPS | PhaseGate/Fixed [95% paired bootstrap] | PhaseGate SLO | Fixed SLO |
|---:|---:|---:|---:|---:|---:|
| 5% | 79.30 | 39.91 | 2.003 [1.833, 2.039] | 5/5 | 5/5 |
| 25% | 393.55 | 196.45 | 2.009 [1.971, 2.052] | 5/5 | 5/5 |
| 100% | 1552.90 | 780.47 | 1.992 [1.987, 1.993] | 5/5 | 5/5 |

## All matched values

- 5% PhaseGate/Fixed: 1.833, 2.003, 2.039, 2.024, 1.937 (min--max 1.833--2.039)
- 5% PhaseGate/TimeGate: 0.944, 1.078, 1.110, 1.123, 1.023 (min--max 0.944--1.123)
- 5% PhaseGate-minus-Fixed QPS: 33.25, 39.70, 41.75, 40.88, 37.29 (min--max 33.25--41.75)
- 25% PhaseGate/Fixed: 2.010, 1.971, 2.052, 2.003, 2.009 (min--max 1.971--2.052)
- 25% PhaseGate/TimeGate: 1.077, 1.026, 1.146, 1.040, 1.043 (min--max 1.026--1.146)
- 25% PhaseGate-minus-Fixed QPS: 199.00, 190.36, 207.95, 197.10, 197.42 (min--max 190.36--207.95)
- 100% PhaseGate/Fixed: 1.993, 1.987, 1.988, 1.992, 1.992 (min--max 1.987--1.993)
- 100% PhaseGate/TimeGate: 1.012, 1.010, 1.008, 1.010, 1.008 (min--max 1.008--1.012)
- 100% PhaseGate-minus-Fixed QPS: 775.29, 770.92, 771.07, 774.44, 773.18 (min--max 770.92--775.29)

## Pageout sensitivity

- 5%: 3/5 pairs have no pageout soft flag; clean-pair median ratio 1.937.
- 25%: 2/5 pairs have no pageout soft flag; clean-pair median ratio 2.028.
- 100%: 1/5 pairs have no pageout soft flag; clean-pair median ratio 1.992.

## Interpretation

A positive median PhaseGate/Fixed ratio persists under the tested synthetic 5% and 25% intermittent-demand traces.
TimeGate joint-SLO passes across 5%, 25%, and 100% duty were 0/5, 0/5, and 0/5, respectively.
The three duty points do not establish a continuous causal law and do not generalize to embedding, indexing, file, or network tools.

## Integrity and validity

All SLO failures and unfavorable pairwise values are retained. Primary confidence intervals use 10,000 paired resamples of the five run-level comparisons, never individual requests.
