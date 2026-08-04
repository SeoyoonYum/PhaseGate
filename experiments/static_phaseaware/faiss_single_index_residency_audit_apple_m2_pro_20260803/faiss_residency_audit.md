# FAISS residency audit

- Timestamp: 2026-08-03T01:37:47.873518+02:00
- Exact index: `/Users/m1/kv-uma-research/experiments/phaseguard/index/hnsw_100k_d384.faiss` (180,820,834 bytes)
- Production retrieval architecture: one spawned index-owner process with a bounded thread pool.
- Measurement control: the parent retains one identical control copy in every configuration; 0→N increments isolate the production owner and logical thread-pool size.
- Duplication decision: **absent**

| Workers | Tree RSS | RSS increase vs 0 | Physical footprint | Retrieval QPS | Pageouts | Swap delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 224.5 MiB | 0.0 MiB | 201.5 MiB | 0.0 | 0 | 0 |
| 1 | 466.8 MiB | 242.3 MiB | 418.4 MiB | 818.4 | 0 | 0 |
| 2 | 472.8 MiB | 248.2 MiB | 424.3 MiB | 2045.9 | 0 | 0 |
| 3 | 479.1 MiB | 254.6 MiB | 430.7 MiB | 3273.5 | 0 | 0 |
| 4 | 485.1 MiB | 260.5 MiB | 436.7 MiB | 4911.7 | 0 | 0 |

Per-additional-worker tree RSS increments: 242.3 MiB, 5.9 MiB, 6.4 MiB, 6.0 MiB.

The first nonzero configuration adds the single production index owner. Raising
logical concurrency from 1 to 4 adds only 18.3 MiB physical footprint in total,
not one index-sized allocation per thread.

Relative to the old four-process pool, cap-4 physical footprint fell from
1,064.0 MiB to 436.7 MiB, reclaiming 627.3 MiB. Tree RSS fell from 1,188.7 MiB
to 485.1 MiB, a reduction of 703.6 MiB. Both comparisons include the same
parent-held diagnostic control index, so it cancels from the difference.

Raw `footprint`, `vmmap -summary`, `lsof`, host snapshots, and per-PID results are retained under `raw/workers_N/`.
