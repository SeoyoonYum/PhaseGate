# M2 TimeGate Report

The initial 430.718-second cyclic replay schedule was preserved after its smoke missed the duty tolerance due to observation-window boundary effects. Before any held-out evaluation, the handoff's predeclared periodic fallback was derived only from the same M2 calibration duty and transition-rate targets.

The harness's sample-count duty diagnostic is biased by load-dependent sampler spacing, so the required wall-time fraction was computed by integrating consecutive raw sample timestamps. No evaluation outcome was used.

- `T_high`: 2.178218167 s
- `T_low`: 2.128961878 s
- Realized timestamp-weighted high-cap duty: 50.657972%
- Duty target/error: 50.571793% / 0.086179%
- Transition-rate error: 0.099549%
- Phase callback calls: 0
- Semantic audit: **PASS**
