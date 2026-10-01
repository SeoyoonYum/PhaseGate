# PhaseGate

**Accepted at the NeurIPS 2026 Workshop on On-Device Intelligence.**

Seoyoon Yum and Sehoon Kim (KAIST)

[Camera-ready paper](paper/camera_ready/PhaseGate_CameraReady.pdf) · [LaTeX source and validation](paper/camera_ready/) · [Versioned release](https://github.com/SeoyoonYum/PhaseGate/releases/tag/camera-ready-2026-10-01)

PhaseGate is a phase-aware CPU admission policy for on-device LLM systems with
unified memory. It allows more concurrent retrieval work while the LLM processes
the prompt (prefill), then lowers retrieval concurrency while the LLM generates
tokens (decode). The caps are calibrated for each device and workload; `4 -> 1`
is the selected policy for the primary configuration, not a universal constant.

The central comparison is against the strongest fixed concurrency that satisfies
the same p95 time-per-output-token (TPOT) and time-to-first-token (TTFT) limits.
On the fan-cooled base-M4 Mac mini, PhaseGate delivers 2.01x the retrieval
throughput of Fixed-1 while both satisfy the latency limits in all seven held-out
comparisons. A phase-blind control that uses the same two caps on a calibration-derived schedule
has similar aggregate retrieval throughput but violates the TPOT limit in all seven.

## What Is Included

| Evidence | Device | Role | Result location |
|---|---|---|---|
| Primary policy campaign and output-length sweep | Mac mini, base Apple M4, 16 GB | Primary result | [`fanmac_m4_causal_shape_20260805_r5`](experiments/static_phaseaware/fanmac_m4_causal_shape_20260805_r5/) |
| Prefill/decode contention replication | Mac mini, base Apple M4, 16 GB | Mechanism check | [`fanmac_m4_contention_20260806_r2`](experiments/static_phaseaware/fanmac_m4_contention_20260806_r2/) |
| Cross-device policy replication | Mac mini, Apple M2, 16 GB | Replication | [`fanmac_m2mini_replication_20260805`](experiments/static_phaseaware/fanmac_m2mini_replication_20260805/) |
| Intermittent retrieval-demand sweep | Mac mini, base Apple M4, 16 GB | Workload sensitivity | [`fanmac_m4_bursty_hnsw_20260822_r3`](experiments/static_phaseaware/fanmac_m4_bursty_hnsw_20260822_r3/) |
| Original policy campaign | Mac mini, Apple M2 Pro, 16 GB | Earlier supporting evidence | [`fanmac_main_apple_m2_pro_20260801`](experiments/static_phaseaware/fanmac_main_apple_m2_pro_20260801/) |
| Access-pattern and contention controls | fanless M4 MacBook Air and M1 MacBook Air | Mechanism controls | [`PHASEGUARD_RESULTS.md`](PHASEGUARD_RESULTS.md) |

The base-M4 Mini contention campaign reproduces the phase asymmetry seen on the
fanless M4 Air: at four concurrent HNSW searches, p95 TTFT rises by 6.9% while
p95 TPOT rises by 59.8%. The intermittent-demand study is deliberately scoped as
a synthetic sensitivity test, not an end-to-end agent benchmark. At 5%, 25%,
and 100% scheduled demand, PhaseGate retains about 2x the Fixed-1 retrieval
throughput while both pass the frozen SLO; the phase-blind control passes none.

## Repository Guide

- [`paper/camera_ready`](paper/camera_ready/) contains the accepted paper with author information, self-contained LaTeX source, figures, and camera-ready validation tables. The main text is six pages; references and appendices are additional.
- [`paper/submission_v21`](paper/submission_v21/) preserves the earlier anonymous submission for history.
- [`experiments/static_phaseaware`](experiments/static_phaseaware/) contains
  protocol freezes, processed measurements, reports, audits, and figures.
- [`scripts`](scripts/) and [`src/phaseguard`](src/phaseguard/) contain the main
  experimental harness and policy implementation.
- Each newer M4 campaign carries the implementation used for that campaign in
  its result bundle or `implementation/` directory. This preserves exact
  provenance where campaign branches evolved independently.
- [`ARTIFACTS.md`](ARTIFACTS.md) explains what is tracked, what remains in the
  checksum-verified offline archives, and how the histories were integrated.

## Reproduction Notes

The measurement target is part of the experiment: results should not be
reproduced on an arbitrary machine and interpreted as the reported device
result. The campaign reports freeze the machine, model revision, sequence
lengths, HNSW index parameters, seeds, latency budget, retry rules, and policy
selection before held-out evaluation.

Start with the final report and protocol freeze in the campaign directory. The
M4 campaign snapshots preserve the exact scripts used on the measurement host;
local model and index paths in the frozen manifests are provenance records and
must be changed to valid local paths for a new campaign. A new configuration is
a new campaign, not a continuation of an existing frozen result.

The repository retains earlier KV-cache and PhaseGuard investigations because
they document how the project reached the phase-aware scheduling question.
[`RESEARCH.md`](RESEARCH.md), [`EXPERIMENTS.md`](EXPERIMENTS.md), and
[`DECISIONS.md`](DECISIONS.md) are the historical research record.

## Scope

The LLM workload uses controlled token IDs and forward passes; it does not evaluate natural-language output quality. Admission is non-preemptive, so calls already admitted can continue into decode.

The reported throughput is HNSW retrieval throughput under offered work, not
end-to-end assistant throughput. Gains depend on the device, model, prompt and
output lengths, retrieval configuration, and demand. The bursty sweep supports
robustness to three synthetic duty levels; it does not establish a typical-user
arrival distribution or generalize to embedding, indexing, file, or network
tools.

## Recheck the camera-ready calibration

From a normal clone, without MLX or model weights:

```bash
python3 paper/camera_ready/validation/audit_calibration_and_chunks.py \
  --summary-only --output-dir /tmp/phasegate-calibration-audit
```

Large raw event timelines remain outside Git. The public artifact includes code, experiment scripts, processed measurements, and frozen demand traces; see [ARTIFACTS.md](ARTIFACTS.md) for the exact coverage. Camera-ready source archives omit private editing notes and intermediate build logs.

## Citation

```bibtex
@inproceedings{yum2026phasegate,
  title = {PhaseGate: Phase-Aware CPU Retrieval Scheduling for On-Device LLMs on Unified Memory},
  author = {Yum, Seoyoon and Kim, Sehoon},
  booktitle = {NeurIPS 2026 Workshop on On-Device Intelligence: Foundation Models under Real-World Constraints},
  year = {2026},
  url = {https://github.com/SeoyoonYum/PhaseGate}
}
```
