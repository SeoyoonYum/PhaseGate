# Harness Changelog

## r3 protocol-only change

Final source commit: `9eb88e9a95844a2a37d021a221eab72373e11f16`.

Relative to stopped r2 commit `4dd97289389ff3471cb2ccc093962caac0733457`, the campaign runner now:

- reads the baseline stability tolerance from the immutable freeze;
- records a required rationale for non-default tolerance;
- rejects attempts to re-prepare an existing campaign with different tolerance or seed offset;
- applies a frozen seed offset to demand traces, baseline, primary, midpoint, policy order, and duty order;
- exports the selected tolerance and generic `within_frozen_tolerance_both` accounting;
- includes the tolerance in the normalization-baseline record.

The measured block implementation, model execution, token timestamp position, DemandGate, PhaseGate, TimeGate, HNSW worker/chunk path, event observer, memory observer, and offline event reconstruction are unchanged from r2.

## Preserved attempts

- Stopped r2 remains at `fanmac_m4_bursty_hnsw_20260820_r2` with both failed baseline sets and verified stop bundles.
- The partial `fanmac_m4_bursty_hnsw_20260821_compat_r3` directory is preserved and excluded because an operator status query may have overlapped its fourth block.
- Final r3 mechanics evidence is `fanmac_m4_bursty_hnsw_20260821_smoke_r3`.
- Final clean compatibility evidence is `fanmac_m4_bursty_hnsw_20260821_compat_r4`.

No paper file was edited.
