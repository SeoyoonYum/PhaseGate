# Camera-ready validation data

These tables were computed from the original recorded runs; no new performance experiments were run for the camera-ready revision.

- `calibration_audit.csv`: three runs per candidate, median QPS, maximum run-level p95 latency ratios, and eligibility at B=1.25.
- `normalization_baseline_audit.csv`: frozen M4 baseline verification.
- `retrieval_configuration_audit.csv`: 16-query FAISS call size, feeder task size, efSearch, and top-k from recorded manifests. Manifest paths retain the original local campaign layout.
- `heldout_chunk_transition_runs.csv` and `heldout_chunk_transitions.csv`: seven M4 held-out runs and 1,750 decode entries.
- `calibration_zero_cap_transitions.csv`: residual work in zero-decode-cap calibration runs; this does not establish a cause for nonmonotonic TPOT.

An active call has been admitted but has not completed. Service time excludes queue waiting. Drain time measures the delay after decode entry until at most one call is active and the count remains within that cap for the rest of decode. Completion of all calls active at decode entry is a separate initial-set statistic. Reported tail values are medians of per-run p95s, not a pooled percentile. Observed maxima are empirical measurements, not guarantees.

## Source provenance

The primary M4 campaign's preserved implementation is in `experiments/static_phaseaware/fanmac_m4_causal_shape_20260805_r5/implementation/`. Its README records source commit `d5868e14e7e7dccc04dd5215dc76619891398da3`. The preserved GPU worker matches the separately archived contention implementation (SHA-256 `e9ed4714042aafb658ba84f38bda119c69da3a5b37bc2458852bb9b425cfb6eb`). We corroborated synthetic execution details with these preserved sources and campaign records; the earlier exact execution commit `72bcef3a1516d6100ec1c975d7678abc48d0ffda` was not available as a local Git object during this check.

The M4 workload uses deterministic input token IDs, transformer-body prefill, repeated full-model forwards with a fixed single-token input and growing KV cache, and synchronization before timestamps. It excludes sampling and output feedback. This is a controlled systems workload, not an evaluation of natural-language generation quality.

## Checks

`audit_calibration_and_chunks.py --summary-only` uses only Python's standard library and the public campaign CSVs. Full transition analysis needs the archived raw timelines. The published paper was compiled with Tectonic; all fonts are embedded and there are no unresolved references or overfull boxes. `SHA256SUMS` records the PDF and source file hashes for this release.
