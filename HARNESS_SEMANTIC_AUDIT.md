# PhaseGate Harness Semantic Audit

## Token path

The measured decode loop remains:

1. `model(token, cache=cache)`
2. `mx.eval(y)`
3. `time.perf_counter()` through `PhaseMonitor.record_token()`
4. append to the request-local timestamp list

The fast path has no file I/O, subprocess, JSON formatting, statistics, RSS read, or
lock shared with an observer. Token counts and strict monotonicity are validated after
every block.

## Event schema

Final events contain a strictly increasing sequence number, monotonic timestamp,
run/repeat ID, request/query ID where applicable, event type, policy, requested cap,
actual active-query count, and observational LLM phase. Runtime state-change events are
buffered when they occur. Request and token events are joined from their original
timestamps after measurement and globally ordered offline.

Required events are present: `phase_enter_prefill`, `phase_enter_decode`,
`phase_exit_or_idle`, `cap_change`, `query_admitted`, `query_started`,
`query_completed`, `request_start`, `prefill_complete`, `token_ready`,
`request_complete`, and `timegate_transition`.

## Retrieval semantics

The manager loads one read-only index once. An external application-controlled thread
pool checks the current cap before each HNSW chunk. Lowering a cap does not preempt an
active chunk. A worker cannot admit a new chunk until actual active count is below the
new cap. Event reconstruction verifies admitted/started/completed query totals,
nonnegative active count, and admission-at-cap semantics.

## TimeGate separation

`TimeGateController` is not a phase policy. Its selection thread accepts only a frozen
wall-clock schedule and offset. Its runtime transition event sink receives only time
and cap. Actual LLM phase is joined offline by event timestamp. The audit field remains
`phase_state_consulted=false`.

## Memory observer

Production event mode uses a persistent Python thread calling the native
`resource.getrusage()` API no faster than once per second. It launches no subprocess
and never accesses the phase/token lock. Minimal mode has no in-block memory observer.
Legacy mode retains the old sampler solely for the frozen causal diagnostic.

## Functional evidence

- `scripts/test_observer_semantics.py`: token fast path, native monitor, ordered event
  reconstruction, TimeGate separation, one-index/one-OMP-thread query accounting, and
  orderly exception-buffer preservation.
- `scripts/test_m4_phasegate_semantics.py`: Fixed/PhaseGate caps including cap 4,
  backlog, non-preemptive drain, no admission above the lowered cap, and exact query
  accounting.
- `scripts/test_static_phaseaware_semantics.py`: policy-selection and transition
  metric behavior.

All three tests passed before observer validation. Event, minimal, and legacy short
LLM-only smoke blocks also passed; their performance is intentionally not interpreted.
