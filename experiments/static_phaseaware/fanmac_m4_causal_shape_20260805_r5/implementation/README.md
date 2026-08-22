# Exact r5 Implementation Snapshot

This directory is an inspection snapshot of the source used by the base-M4 r5
campaign. The source is from commit
`d5868e14e7e7dccc04dd5215dc76619891398da3`; the result bundle's
`GIT_COMMITS.txt` records the earlier protocol and selection commits.

The scripts preserve their original repository-relative paths. To reproduce or
extend the campaign, place `scripts/` and `src/` at the root of a clean checkout,
install the frozen environment, provide the model and HNSW index identified by
the campaign manifests, and create a new campaign directory. Do not write new
measurements into this frozen result directory.
