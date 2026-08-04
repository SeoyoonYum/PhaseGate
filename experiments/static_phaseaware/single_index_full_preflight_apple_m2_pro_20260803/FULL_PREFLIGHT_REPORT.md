# Single-index full-size preflight

Commit: `ed46a8f584943e6f312677e1498d3358a3dcc437`

Selected environment:

- one spawned retrieval process
- one read-only FAISS/HNSW index
- four logical query threads
- FAISS internal threads per query = 1
- MLX cache limit = 5.5 GB
- exact production model/index/query implementation
- always-backlogged retrieval
- 100 measured requests × 128 generated tokens
- per-token timestamp logging enabled

| Policy | Pageout | Swap | Physical footprint | Peak RSS | Retrieval QPS | PREFILL active mean/p95 | DECODE active mean/p95 | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Fixed-2 | 0 | 0 | 1309.8 MiB | 1355.8 MiB | 1719.2 | 1.998/2.0 | 1.999/2.0 | PASS |
| PhaseGate 4→3 | 0 | 0 | 1308.0 MiB | 1354.0 MiB | 2814.1 | 3.995/4.0 | 3.006/3.0 | PASS |

| Policy | TPOT p95 | TTFT p95 | ITG p95 | ITG p99 | Transition p95 | First-four p95 | Max ITG |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed-2 | 9.723 ms | 1360.513 ms | 10.659 ms | 12.594 ms | 17.271 ms | 10.563 ms | 13.272 ms |
| PhaseGate 4→3 | 10.064 ms | 1364.873 ms | 10.927 ms | 12.908 ms | 18.458 ms | 10.870 ms | 13.531 ms |

Both blocks completed exactly 100 requests, 12,800 generated tokens, and
12,700 inter-token gaps. Queue non-empty fraction was 1.0; policy semantics,
token timestamps, sentinel checks, and TPOT/TTFT/QPS stability checks passed.

Fixed-query validation found exact top-k ID equality and zero distance
difference. Task accounting was 8,192/8,192 with no duplicate completion.
Fixed cap traces produced p95 concurrency 1/2/3/4, and 2-thread QPS was 2.00×
1-thread QPS, confirming concurrent FAISS execution outside the GIL.

The cap-4 residency audit reclaimed 627.3 MiB physical footprint and 703.6 MiB
tree RSS relative to the old multiprocess pool. No run from this preflight or
any aborted multiprocess campaign is eligible for the replacement campaign.
