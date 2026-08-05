# r5 Protocol Amendment: CPU-Only Observer

Authorized by the user on 2026-08-06 after the r5 baseline A completed.

The valid r5 baseline A remains the normalization baseline. Before CPU-only
scaling began, an audit found that `scripts/run_m4_cpu_scaling.py` still used
0.1-second state polling and repeated `ps` subprocesses. No r5 CPU-scaling,
mechanism, calibration, TimeGate, or held-out block had yet run.

Commit `3194749db612b6ea1f0baa3d110e2486b9e579e2` changes only the CPU-only
scaling observer and its tests: active concurrency is reconstructed from query
events, memory is sampled natively at no more than 1 Hz, CPU utilization uses
process resource counters, and the measured block launches no observer
subprocess. The LLM baseline workload, model, prompts, token timestamp location,
event observer, metric definitions, and baseline data are unchanged.

All CPU-scaling blocks use the new commit and freeze their randomized order
before the first block. Results from the unused r6 preparation directory are
not pooled; that directory contains no measured runs.
