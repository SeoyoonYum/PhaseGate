# Single-index retrieval validation

- Result: **PASS**
- Index: `/Users/m1/kv-uma-research/experiments/phaseguard/index/hnsw_100k_d384.faiss` (180,820,834 bytes)
- Architecture: one spawned FAISS index-owner process, four query threads
- FAISS internal threads per query: 1

## Equivalence

```json
{
  "query_count": 256,
  "top_k": 10,
  "old_vs_shared_ids_exact": true,
  "old_vs_concurrent_ids_exact": true,
  "old_vs_shared_distances_close": true,
  "old_vs_concurrent_distances_close": true,
  "maximum_absolute_distance_difference": 0.0
}
```

## Accounting

```json
{
  "architecture": "single_index_process_thread_pool",
  "owner_pid_count": 1,
  "logical_worker_count": 4,
  "manager_query_count_match": true,
  "manager_checksum_absolute_difference": 0.0,
  "submitted_task_ids_unique": true,
  "completed_request_ids_unique": true,
  "completed_request_ids_exact": true,
  "expected_accounting_queries": 8192,
  "accounted_queries": 8192,
  "no_query_loss_or_duplicate_completion": true,
  "result_errors": []
}
```

## Fixed-cap concurrency

| Cap | Active mean | Active p95 | Active max | QPS | Queue non-empty |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.0 | 0 | 0.0 | 1.000 |
| 1 | 0.999 | 1.0 | 1 | 1023.7 | 1.000 |
| 2 | 1.998 | 2.0 | 2 | 2047.8 | 1.000 |
| 3 | 2.998 | 3.0 | 3 | 2729.7 | 1.000 |
| 4 | 3.995 | 4.0 | 4 | 3412.0 | 1.000 |

2-vs-1 QPS ratio: 2.000
4-vs-1 QPS ratio: 3.333

PhaseGate stable PREFILL and DECODE samples are retained in the JSON output.
