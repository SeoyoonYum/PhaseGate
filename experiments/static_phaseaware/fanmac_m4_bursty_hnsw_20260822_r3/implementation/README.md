# Exact Bursty-Demand Implementation Snapshot

The top-level `scripts/` and `src/` directories are the source overlay shipped
in the checksum-verified bursty result bundle. They preserve the measured
implementation, including the frozen host paths and environment assertions.
`base_commit/` contains the complete `scripts/` and `src/` tree from r5 commit
`d5868e14e7e7dccc04dd5215dc76619891398da3`.

The scripts retain their original repository-relative paths. To reconstruct the
measured tree, start with `base_commit/`, then copy the top-level `scripts/` and
`src/` over it. Provide the model and HNSW index identified in
`machine_manifest.json`, and use a new output directory for any new campaign.
Do not alter the frozen traces or append data to this result.
