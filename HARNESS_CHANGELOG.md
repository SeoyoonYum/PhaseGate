# PhaseGate Harness Stabilization Changelog

## Scope

This change stabilizes measurement on the fan-cooled base Apple M4. It does not alter
the model, prompt shape, token timestamp location, FAISS index, retrieval policy, or
paper. The immutable r4 campaign remains diagnostic-only.

## Observer modes

- `minimal`: request-local token timestamps, phase events, and block-boundary host
  state only. No in-block memory sampler.
- `event`: production state-change events plus a native, in-process memory sample no
  faster than 1 Hz. No measured-block subprocess monitoring.
- `legacy`: the r4 5 ms polling loop and repeated `ps` RSS reads. This mode exists only
  for the frozen observer intervention and is forbidden in later campaign stages.

Every run manifest and result row records `observer_mode`.

## Implementation changes

- Added `phaseguard.observer.EventBuffer`, `NativeMemoryMonitor`, deterministic event
  finalization, event-log reconstruction, and compatibility state reconstruction.
- Removed `PhaseMonitor`'s token-path `RLock`. `GPUWorker` still calls
  `record_token()` immediately after the same `mx.eval()` barrier; the timestamp is
  appended to the same request-local in-memory list.
- Added state-change events for caps, retrieval task lifecycle, query admission/start/
  completion, TimeGate transition, LLM phases, request lifecycle, prefill completion,
  and token readiness. Token events are materialized after the measured block from
  the already-buffered timestamps.
- Production RSS monitoring now uses `resource.getrusage(RUSAGE_SELF)` at at most 1 Hz.
  Exact VM, swap, memory-pressure, RSS, and pageout measurements remain at block
  boundaries. In-block physical footprint is explicitly unavailable where no native
  API is present.
- Tightened non-preemptive draining: an already-active query may drain after a cap
  decrease, but no new query is admitted while the actual active count is at or above
  the current cap.
- Added explicit observer mode to run keys and child-process commands.

## Unchanged semantics

- Token timestamp clock and placement relative to `model()` and `mx.eval()`.
- One owner process and one read-only FAISS HNSW index.
- FAISS OpenMP thread count one.
- Fixed, PhaseGate, and phase-blind TimeGate cap definitions.
- Non-preemptive active-query completion.
- Buffered output flushed only after a block.
- Official run-level p95 TPOT and TTFT definitions.

## Validation rule

Smoke performance is not scientific evidence. A full campaign restart is allowed only
after the frozen minimal/event correctness and overhead gates pass. The legacy/event
result is diagnostic and may be positive, negative, or inconclusive.

## Sentinel stabilization amendment

The first frozen observer validation exposed a separate cold-reference failure: a
1677.70 ms stage reference moved to 1795.88 ms after one long block. The stage was
aborted after the allowed retry. New stages condition long-context prefill for at least
120 seconds and require the latest five burst medians to span no more than 1% before
freezing a reference. This happens before measurement and does not change workload or
timestamp semantics. The failed validation directory remains immutable evidence.
# Post-validation production lockout

- The legacy 5 ms observer now requires the explicit
  `--allow-legacy-observer-diagnostic` flag. Production campaign commands omit
  this flag, so the retired observer cannot be selected accidentally.
