# Prior pause record — campaign resumed

- Original path: `/Users/m1/kv-uma-research/experiments/static_phaseaware/fanmac_slo_goodput_tail_apple_m2_pro_20260803_headless_single_index_main`
- Archive timestamp: `2026-08-03T06:12:21+0200`
- Repository commit: `8ba6cbb3eb0d8e948f6ceef6898b753076022451`
- Calibration: 7 valid blocks and 6 invalid attempts, including 3 pageout-invalid attempts
- Isolated baseline: 7 valid blocks
- Token logging overhead: 6 valid blocks

This campaign paused after two consecutive pageout-invalid Fixed-0 attempts.
The observed low-intensity pageouts were not monotonic with workload intensity.

On 2026-08-03, the experiment owner explicitly superseded the pause directive:
the campaign was restored to its original path and continues from calibration
7/45. The model, index, MLX limit (5.5 GB), FAISS single-index architecture,
policy grid, workload, SLO grid, and token logging remain unchanged. Global
pageout-only, sentinel, and within-block stability deviations are now recorded
as soft flags; swap growth, warning/critical memory pressure, corrupt data,
invalid timestamps, incorrect outputs, or incorrect policy semantics remain
hard failures. This file preserves the provenance of the earlier pause.
