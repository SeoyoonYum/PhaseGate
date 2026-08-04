# Bounded MLX cache-limit search

- Retrieval architecture: `single_index_process_thread_pool`
- Model: unchanged 1.5B 4-bit MLX model
- Index: unchanged 100k × 384 FAISS HNSW index
- Output length: 128 tokens
- Measured requests per block: 100
- Selected limit: **5.5 GB**

| Limit | Policy | Pageouts | Swap delta | TPOT p95 | TTFT p95 | p99 ITG | Transition p95 | Retrieval QPS | Result |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 6.0 GB | Fixed-2 | 0 | 0 | 9.728 ms | 1360.849 ms | 12.572 ms | 18.717 ms | 1723.2 | PASS |
| 6.0 GB | PhaseGate 4→3 | 3 | 0 | 10.250 ms | 1529.142 ms | 12.985 ms | 20.433 ms | 2811.6 | FAIL |
| 5.5 GB | Fixed-2 | 0 | 0 | 9.723 ms | 1360.513 ms | 12.594 ms | 17.271 ms | 1719.2 | PASS |
| 5.5 GB | PhaseGate 4→3 | 0 | 0 | 10.064 ms | 1364.873 ms | 12.908 ms | 18.458 ms | 2814.1 | PASS |

The search stopped at the highest tested limit satisfying both full-size
preflights. No 5.0 GB or lower limit was tested. The selected 5.5 GB limit did
not materially reduce Fixed-2 or PhaseGate retrieval QPS and did not introduce
allocation failure. The replacement campaign must rerun its own logging
overhead check and seven-repeat isolated baseline; this diagnostic baseline is
not eligible campaign data.
