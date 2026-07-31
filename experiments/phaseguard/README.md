# PhaseGuard experiments

This directory contains the new application-level co-scheduling experiment. It
does not modify any Phase 0 result. The primary workload is genuine HNSW graph
search through FAISS `IndexHNSWFlat`, the documented fallback because `hnswlib`
was unavailable in the verified environment.

## Layout

- `configs/primary.json`: recorded primary configuration.
- `index/`: deterministic persistent 10k smoke and 100k primary indexes plus metadata.
- `raw/profile_*.jsonl`: one durable record per isolated phase measurement.
- `raw/requests.jsonl`: per-request phase and token timestamps.
- `raw/runs.jsonl`: per-run metrics, environment, SLO, memory, and contamination.
- `raw/timelines/`: GPU phase, CPU retrieval, and permit traces.
- `processed/profile_*.csv`: offline p50/p95/p99 lookup tables.
- `processed/summary.csv`: median/min/max policy summaries and observed oracle rows.
- `plots/`: Figures A-E as publication-resolution PNG and vector PDF.
- `logs/`: environment/profile manifests. Rootless thermal spread is recorded;
  sudo-only `powermetrics` was not used.

## Reproduction

Use the repository virtual environment from the repository root. Core GPU runs
need access to the host Metal device.

```bash
.venv/bin/python scripts/build_hnsw_index.py \
  --output experiments/phaseguard/index/hnsw_100k_d384.faiss \
  --vectors 100000 --dimensions 384 --ef-construction 80 --graph-degree 32

.venv/bin/python scripts/run_phaseguard_profile.py \
  --index experiments/phaseguard/index/hnsw_100k_d384.faiss \
  --model 1.5B --context 2048 --workers 0,1,2,4 --max-workers 4 \
  --ef-search 128 --decode-steps 128 --repetitions 3 \
  --queries-per-task 32 --cooldown 2 --tag primary --allow-battery

.venv/bin/python scripts/run_phaseguard_trace.py \
  --index experiments/phaseguard/index/hnsw_100k_d384.faiss \
  --profile experiments/phaseguard/processed/profile_primary.csv \
  --policy phaseguard --max-workers 4 --model 1.5B --context 2048 \
  --output-tokens 128 --concurrency 4 --requests-per-client 2 \
  --retrieval-queries 4096 --ef-search 128 --slo 1.15 \
  --tag eval --repeat 0 --allow-battery

.venv/bin/python scripts/analyze_phaseguard.py --tag eval \
  --output experiments/phaseguard/processed/summary.csv
.venv/bin/python scripts/plot_phaseguard.py \
  --profile experiments/phaseguard/processed/profile_primary.csv \
  --summary experiments/phaseguard/processed/summary.csv
```

For the other policies, replace `--policy phaseguard` with `uncoordinated`,
`serialized`, `bandwidth`, or `static --decode-workers K`. Evaluation repetitions
used `--repeat 0`, `1`, and `2` in randomized policy order. Static `K=0/2/4`, SLO
1.05/1.10/1.20, and concurrency 2 were staged follow-ups recorded in the same raw
files. All runners append partial results and skip an already completed `run_key`.

## Interpretation notes

`logical_qps` and legacy logical GB/s measure work completed by a CPU workload;
neither is asserted to measure memory-system interference. `contaminated=true`
means the run observed a nonzero VM pageout delta or non-AC power. All primary
evaluation runs observed small pageout deltas (19-160 16-KiB pages) despite
6.7-7.6 GB reclaimable headroom; they are retained and explicitly labeled.
