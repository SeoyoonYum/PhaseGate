# PHASEGATE camera-ready paper

**PhaseGate: Phase-Aware CPU Retrieval Scheduling for On-Device LLMs on Unified Memory**

Seoyoon Yum and Sehoon Kim, KAIST

NeurIPS 2026 Workshop on On-Device Intelligence: Foundation Models under Real-World Constraints

[Read the PDF](PhaseGate_CameraReady.pdf). The camera-ready manuscript has six main-text pages, two reference pages, and three appendix pages. Author names and the workshop-specific first-page footer are included.

## Build

```bash
cd paper/camera_ready
tectonic --keep-logs --keep-intermediates main.tex
# Alternatively: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The required official NeurIPS 2026 style, bibliography, author metadata, and PDF figures are included. A successful build produces `main.pdf`. Figure regeneration is optional (`python make_paper_figures.py`, requiring Matplotlib and NumPy); the figures summarize the reported measurements and do not launch experiments.

## Evidence and reproducibility

The repository's [artifact map](../../ARTIFACTS.md) points to the frozen campaign implementations, configuration records, processed results, and demand schedules. The primary result measures aggregate retrieval capacity with a backlogged queue under calibrated latency constraints, not end-to-end RAG latency. M4 uses nested request-p95 TPOT, while the M2 and M2 Pro protocols use the across-request p95 of request-mean token gaps; see Appendix A.1.

The `validation/` directory includes the camera-ready calibration audit and CPU transition measurements. To recheck the calibration using the processed tables in a normal repository clone:

```bash
python3 paper/camera_ready/validation/audit_calibration_and_chunks.py \
  --summary-only --output-dir /tmp/phasegate-calibration-audit
```

This standard-library command launches no model or retrieval workload. Full event-level drain-time recomputation additionally needs the archived raw M4 timelines, which are not distributed in Git or this source ZIP:

```bash
python3 paper/camera_ready/validation/audit_calibration_and_chunks.py \
  --data-root /path/to/fanmac_m4_causal_shape_20260805_r5 \
  --output-dir /tmp/phasegate-full-audit
```

See [validation/README.md](validation/README.md) for metric definitions and source provenance. Large raw event archives, model weights, and generated HNSW indexes are not included. Public materials include processed results and frozen demand traces; this release does not claim that every raw execution trace is downloadable.
