# Existing context-4096 result audit

## Scope

This audit was completed before adding the context-validation harness. It reads the
PhaseGuard plan/report, experiment README, both processed profile tables, the raw
profile JSONL, the pipeline run/request JSONL, and the measurement, retrieval, CPU
load, and pipeline implementations. Existing Phase 0 and PhaseGuard files remain
unchanged.

## Exact source of the apparent slowdown

The context-4096 statement came from
`experiments/phaseguard/raw/profile_ctx4096.jsonl`, processed into
`processed/profile_ctx4096.csv`. It was an **isolated, body-only transformer prefill
measurement**, not a full-pipeline evaluation and not a queue-inclusive metric.
There are no context-4096 rows in the existing PhaseGuard pipeline `runs.jsonl` or
`requests.jsonl`.

Configuration:

- Device: Apple M4 MacBook Air `Mac16,13`, 16 GB unified memory, fanless.
- Model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, MLX 0.31.2 / MLX-LM 0.31.3.
- Input: deterministic integer token tensor made by `measure.make_tokens(4096)`,
  shape `(1, 4096)`. This is exactly 4096 model input IDs; no text tokenizer or
  queueing is involved.
- Decode comparison: 128 forced-evaluation token steps after a separately built and
  evaluated cache. Generated-token length is not applicable to prefill itself.
- Retrieval: FAISS CPU `IndexHNSWFlat`, 100,000 vectors × 384 float dimensions,
  180,820,834-byte index, graph degree 32, `efConstruction=80`, `efSearch=128`,
  top-k=10. Four persistent spawn workers each loaded the index and consumed one
  32-query task at a time through feeder threads.
- CPU conditions: no retrieval (`workers=0`) and four HNSW workers (`workers=4`).
- Policy/concurrency: none; this was an isolated phase profile. Four workers refers
  to CPU process permits, not request concurrency.
- Repetitions: three scalar prefill measurements per condition and three 128-step
  decode bursts per condition. Global model warmup preceded the shuffled matrix;
  each prefill point itself used `warmup=0, reps=1`. Cooldown was only 2 seconds.
- Baseline: same-context, no-load condition. Context 4096 was not normalized against
  context 2048.

## Metric semantics

The report called 5.99 s and 13.07 s “prefill p95.” This requires qualification:

- Every raw prefill row contains exactly **one** latency value, so its `p50_ms` and
  `p95_ms` are identical.
- The processed p95 is NumPy p95 across the three single-call values, not p95 across
  20+ iterations and not a within-run tail estimate.
- The processed median is likewise the median of only three scalar calls.
- The measured interval begins immediately before `forward_body` and ends after
  `mx.eval(out)`. Cache allocation and input construction occur before the timer;
  GPU queueing does not exist in this harness.
- The existing full pipeline records pure `prefill_ms` separately from `gpu_queue_ms`,
  but no context-4096 pipeline run was made.

Observed scalar prefill calls:

| Condition | Values (ms, raw execution order) | Processed median | Processed p95 |
|---|---|---:|---:|
| 4096, no load | 6059.1, 4881.7, 5347.0 | 5347.0 | 5987.9 |
| 4096, HNSW-4 | 10887.1, 12701.0, 13111.4 | 12701.0 | 13070.4 |

Thus the apparent effect was not merely one extreme p95 call: all three HNSW calls
were much slower than all three baseline calls. But the sample is too small, lacks
per-configuration warmup, and is order/thermal confounded.

## Memory, pageout, and thermal evidence

No-load prefill raw pageout deltas were 149, 16, and 0 pages; HNSW-4 prefill deltas
were 0, 0, and 0 pages. End-of-point reclaimable-headroom estimates were 6.43–6.51 GB
for no load and 6.36–6.83 GB for HNSW-4. Therefore the old result does **not** fit the
simple claim that slowdown appeared only in pageout-observed points. However:

- swap-used deltas, idle pageout rates, compressor growth, memory-pressure status,
  and total experiment-process RSS were not recorded;
- `headroom_bytes` combined free/inactive/speculative/purgeable pages and is not by
  itself proof of a stable memory state;
- all four HNSW worker processes held their own FAISS index mapping/copy;
- nonzero cumulative pageout counters were never used—the recorded value was a
  per-point delta, but the previous pipeline report treated every nonzero delta as
  contamination without an idle-rate control.

Temperature and GPU clock were not logged. `thermal_spread=1.0` for prefill is
uninformative because each row contains one value. The shuffled raw order placed
no-load prefill at positions 0, 1, and 6, while HNSW-prefill occurred at positions
3, 9, and 11. The two slowest HNSW prefills were the final two points, so accumulated
heating/run order is a plausible confound. The first HNSW point was already 10.9 s,
so run order alone is not established as the explanation.

## Comparison with context 2048 and Phase 0

The context-2048 HNSW profile used the same index/model path but four worker counts.
Its three no-load prefill calls were 2048.8–2052.6 ms and HNSW-4 calls were
2153.0–2170.3 ms (+5.7% processed p95). All but two decode points had zero pageout;
prefill points all had zero pageout.

The prior synthetic random-access result used one native P-core-biased process with
a 96 MB random-gather working set at 100% duty. At context 2048 it reported +8.4%
prefill and +28.3% decode slowdown, based on median prefill timing. It did not test
context 4096 in the saved P2 load-type table. HNSW-4 and synthetic random therefore
differ in process count, working set, access implementation, metric aggregation, and
load duration; their mechanisms cannot be assumed equivalent.

## Audit conclusion

The existing artifacts establish an **observed candidate context-dependent isolated
prefill slowdown**, but they do not establish intrinsic long-context sensitivity.
The strongest deficiencies are three single calls per condition, no per-condition
warmups, a 2-second cooldown, missing swap/compressor/idle-rate telemetry, and run-order
confounding. The result is not a pipeline artifact in origin because it was never
measured in the pipeline, and it is not explained by the recorded pageout deltas
alone because every HNSW-prefill point had zero pageout during its measurement window.

The validation must measure at least 20 iterations after three warmups, interleave
contexts and conditions, compare HNSW with the existing native random gather, and
separate clean from memory/thermal-suspect runs before revising the claim.
