# PhaseGate Artifact Map

This repository tracks the material needed to inspect the claims without
turning Git history into multi-gigabyte object storage: protocol freezes,
machine manifests, processed run- and request-level tables, audit outputs,
figures, analysis code, frozen demand traces, and final reports.

## Integrated Campaigns

- The M2 Mac mini replication was already on `main` at commit `99020c4`.
- The base-M4 Mac mini r5 primary campaign, observer validation, and r2
  contention campaign were imported from remote branch
  `exp/m4-contention-r2-20260807` at commit `70c27e2`.
- The bursty-demand campaign was imported from its checksum-verified result
  bundle. Its exact implementation snapshot is stored under the campaign's
  `implementation/` directory.
- The current paper artifact is [`paper/camera_ready/`](paper/camera_ready/), updated for the six-page camera-ready allowance. `paper/submission_v21/` remains a historical submission snapshot.

The device campaigns evolved on separate branches. They are therefore retained
as campaign-scoped artifacts rather than silently replacing one campaign's
shared harness with another's. The M4 r5 result bundle also records the exact
campaign scripts and source-commit lineage.

## Large Offline Artifacts

The following archives are intentionally not committed:

| Artifact | SHA-256 | Reason |
|---|---|---|
| `fanmac_m4_bursty_hnsw_20260822_r3_full.zip` | `26610b5623e11298970c8d9d8194f75601c0d6b77365f09f99e245038ae83f94` | 188 MB compressed archive containing multi-GB event timelines |
| Bursty raw campaign directory | Per-file hashes in [`raw_event_checksums.csv`](experiments/static_phaseaware/fanmac_m4_bursty_hnsw_20260822_r3/raw_event_checksums.csv) | Approximately 5 GB of append-only timelines and raw request logs |
| Model weights and HNSW indexes | Model revision and index SHA-256 are in each machine/protocol manifest | Reproducible external inputs, not source code |

The compact bursty bundle used for this import has SHA-256
`04a51bfedd367fa40c80404cd8205d572aca123e3dbeaaea4eed4acc3865387b`.
Its contents were extracted into
`experiments/static_phaseaware/fanmac_m4_bursty_hnsw_20260822_r3/`, avoiding a
duplicate ZIP object in Git.

## Integrity Rules

Raw attempts remain append-only in the offline archives. Invalid attempts and
unfavorable valid outcomes are not deleted from the audit tables. Confidence
intervals use matched run-level comparisons as described in each final report;
individual requests are not treated as independent experimental repeats.

## Camera-ready release (2026-10-01)

The release adds the named-author PDF, buildable LaTeX source, paper figures, calibration and transition-audit CSVs, and a standard-library calibration check. The main text and evidence are aligned with the accepted camera-ready revision. Downloadable PDF/source assets are listed in the [versioned release](https://github.com/SeoyoonYum/PhaseGate/releases/tag/camera-ready-2026-10-01).

This publication keeps the existing raw-data boundary: multi-GB event timelines remain in offline archives. No raw-data download is implied by the paper's repository link. Full trace recomputation requires those archives, whereas the calibration check works directly from public processed tables. Model weights and HNSW indexes are external inputs specified by the campaign manifests.
