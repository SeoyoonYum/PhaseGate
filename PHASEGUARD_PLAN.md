# PhaseGuard experimental plan

## Scope and claim boundary

PhaseGuard is an application-level co-scheduler for concurrent on-device LLM
requests on single-pool unified memory. It controls only application-owned CPU
retrieval processes. Version 1 observes a serialized MLX GPU worker and changes
CPU retrieval concurrency at GPU phase boundaries; it neither schedules kernels
nor suspends external processes.

## Reused infrastructure

- `src/common/models.py`: Qwen2.5 4-bit registry and MLX-LM loading.
- `src/common/measure.py`: forced `mx.eval` barriers, prompt-cache construction,
  body-only prefill, decode timing, peak-memory collection, and global warmup.
- `src/common/thermal.py`: AC check, process-memory snapshot, cooldown/throttle
  flag, and optional `powermetrics` logger.
- `src/rag_sidecar.py` and `src/exp_p2_loadtypes.py`: realistic retrieval precedent
  and the established random/sequential logical-bandwidth counterexample.
- Existing CSV metadata and Matplotlib conventions. Phase 0 files are read-only.

## Device and dependency constraints found during inspection

The available target is a fanless 16 GB Apple M4 MacBook Air (10 CPU cores,
4 performance + 6 efficiency) running macOS 26.5.1 and Python 3.13.12. The
existing environment has MLX 0.31.2, MLX-LM 0.31.3, NumPy 2.4.6, Matplotlib
3.11.0, and FAISS CPU 1.14.3. `hnswlib` is absent, so the documented fallback is
FAISS `IndexHNSWFlat`, which still performs genuine HNSW graph traversal. No
new dependency is required. The machine reports the AC source but a discharging
battery; runs made in that state will be retained and explicitly marked
power-contaminated rather than silently discarded.

## Architecture

1. `PhaseMonitor`: thread-safe `IDLE/PREFILL/DECODE` state and timestamped phase,
   first-token, and per-token events.
2. `CPUTaskManager`: persistent spawn-based process pool. A shared permit count is
   checked before each query/chunk; workers never respawn during a run. Each process
   loads the persistent FAISS HNSW index and reports task timestamps, query counts,
   logical bytes, and latency.
3. `RequestPipeline`: closed-loop clients execute retrieval, enqueue a GPU request,
   wait for the single GPU worker, and repeat. The GPU worker uses the existing
   body-only prefill path and forced-evaluation decode path.
4. Policies: uncoordinated, serialized, static phase-only, profile/SLO lookup, and
   logical-throughput threshold. An evaluation-only oracle selects the best worker
   count from held-out measurements.
5. JSONL is appended after every request/run for resumability. A manifest records
   configuration, git revision, platform/dependencies, memory headroom, power state,
   and contamination flags.

## Staged matrix and decision rules

### Stage B — isolated phase profile

Use Qwen2.5 1.5B 4-bit at context 2048 with 128 decode steps. Sweep FAISS-HNSW
workers 0/1/2/4 using 100k vectors, dimension 384, `efSearch` 32/64/128 as needed.
Run smoke measurements first, then three repetitions for the smallest configuration
that produces stable overlap. Go only if a practical condition slows decode by about
10% and more than prefill. Increase to 250k vectors before considering an optional
file-search workload.

### Stage C — static scheduling

Closed-loop concurrency 2 and 4, initially 6 requests/client in smoke and then enough
for at least 30 requests in key configurations if wall time permits. Compare
uncoordinated, serialized, and static decode permits 0/1/2/4. Primary metrics are
request throughput, p95 end-to-end latency, p95 TPOT, and request-level TPOT-SLO
violation rate.

### Stage D — held-out profile lookup

Build the profile from Stage B training repetitions. Evaluate SLO multipliers
1.05/1.10/1.15/1.20 on separate pipeline runs. The lookup chooses the largest permit
whose profiled p95 TPOT meets the uncontended-baseline SLO. Compare with the best
fixed policy and a logical-throughput threshold tuned only on profile data.

### Stages E/F

Implement a learned predictor only if the optimal permit varies materially and the
lookup improves the Pareto frontier. Otherwise record that ML is unjustified. Repeat
only the central profile and scheduler comparison at context 4096 or model 3B if
memory, thermal state, and runtime permit. No second Apple device is currently in
scope.

## Analysis and required artifacts

Produce `requests.jsonl`, `runs.jsonl`, profile CSV, processed summary CSV, and
PNG/PDF versions of: phase asymmetry, scheduling Pareto frontier, policy comparison,
bandwidth-proxy failure, and representative timelines. `PHASEGUARD_RESULTS.md` will
list exact commands, all completed/failed runs, contamination, numerical findings,
threats, supported claims, next work, and exactly one A/B/C/D outcome.

## Validity checks

- deterministic index/query seeds; fixed generation token IDs;
- index/model warmup and cooldown; randomized policy order;
- separate profile and evaluation run IDs;
- reject invalid/empty timings; retain failed and contaminated rows;
- record VM swap-in/out deltas and available-memory headroom;
- report medians plus run dispersion and never select only winning configurations.
