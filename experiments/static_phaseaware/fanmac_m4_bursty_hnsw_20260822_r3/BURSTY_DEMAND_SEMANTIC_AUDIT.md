# Bursty Demand Semantic Audit

## Frozen implementation

- Final commit: `9eb88e9a95844a2a37d021a221eab72373e11f16`
- Starting r5 commit: `d5868e14e7e7dccc04dd5215dc76619891398da3`
- Observer: event mode; no production subprocess polling; memory sampling ≤1 Hz
- DemandGate and measured execution path: unchanged from stopped r2

## DemandGate information boundary

`DemandGate` receives only a frozen demand trace, an outcome-blind trace offset, a monotonic clock, and buffered event emission. It has no phase object, phase callback, policy object, policy name, cap, TPOT/TTFT input, queue-depth input, occupancy input, or completed-query feedback.

The shared-index worker loop independently requires both `demand_on` and policy permission before a chunk starts. DemandGate cannot raise or bypass the current policy cap. After ON→OFF, an already-started FAISS chunk may complete once, but the same worker cannot start another chunk until demand returns ON.

TimeGate remains phase-blind. DemandGate remains phase- and policy-blind. Offline finalization, not the runtime generator, attaches immutable trace metadata and reconstructs phase/cap/demand/active-chunk overlap.

## Events and reconstruction

Buffered events include `demand_trace_start`, `demand_on`, `demand_off`, `demand_trace_end`, `query_chunk_started`, and `query_chunk_completed`. Demand events carry monotonic time, run ID, sequence, demand level, trace seed, trace offset, and ON/OFF state. Production blocks perform no per-token file I/O or statistics.

Offline audit reconstructs phase, policy cap, demand state, active chunks, completed queries, phase-by-demand overlap, bounded drain, starts during OFF, and accounting exactness. A block is hard-invalid if sequence/timestamps/accounting are corrupt, a chunk begins during OFF, demand or policy semantics are wrong, or the observer launches a subprocess.

## r3 evidence

- DemandGate semantic tests: pass
- Event-observer tests: pass
- PhaseGate and static-policy tests: pass
- Baseline protocol boundary/immutability tests: pass
- Mechanics smoke: pass, 9/9 cells; one TimeGate 25% performance diagnostic is preserved invalid for within-block drift and is not scientific evidence
- Maximum matched scheduled-duty difference: 5%=0.0013234, 25%=0.0067842, 100%=0
- Clean 100%-ON compatibility: pass, 6/6 runtime-valid, exact semantics/accounting
- Median paired request-median TPOT change: −0.0014302
- Median paired all-token mean-gap change: −0.0019459
- Compatibility gates: both within absolute 1%

## Frozen baseline amendment

Only the aggregate five-run baseline stability tolerance changed to ±5%. This does not alter DemandGate, observer, timestamp, block-validity, SLO, or measured runtime semantics. See `BASELINE_STABILITY_PROTOCOL_AMENDMENT.md`.
