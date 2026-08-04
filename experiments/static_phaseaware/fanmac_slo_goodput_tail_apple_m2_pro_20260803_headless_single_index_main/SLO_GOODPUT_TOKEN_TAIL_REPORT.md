> In the inclusive non-hard-failure analysis, at the tightest predeclared joint TPOT/TTFT SLO admitting both continuous families, the frozen PhaseGate achieved higher held-out retrieval goodput.

# SLO Goodput and Token-Tail Report

- Frozen SLO grid: [1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45]
- Primary continuous budget: 1.10
- Fixed / PhaseGate: fixed1 / phasegate4to1
- Joint pass counts: 7/7 and 7/7
- Median retrieval QPS: 921.6231929732857 / 2255.1544988478863
- Median paired gain (10,000 run-level bootstrap 95% CI): 1.4590316069933595 (1.444278033600117, 1.4866325968783647)
- PhaseGate wins: 7/7

- Clean-run sensitivity paired gain: None from 0 paired clean repeats; conclusion: indeterminate_no_paired_clean_primary_repeats.
- Clean-run caveat: a zero paired-clean count means soft-flag exclusion cannot independently confirm or overturn the inclusive primary direction.
All conclusions above use only the fresh held-out evaluation. Individual requests and tokens were not treated as independent experimental repetitions. Sensitivity rows reuse policy runs and are correlated.
