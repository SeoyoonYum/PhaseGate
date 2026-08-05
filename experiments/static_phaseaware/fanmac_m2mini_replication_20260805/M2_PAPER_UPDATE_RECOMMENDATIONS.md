# M2 Paper Update Recommendations

## Proposed replacement paragraph

On a fan-cooled 16 GiB Apple M2 Mac mini, we independently repeated the fixed-cap, phase-gated, and phase-blind duty-cycle control using a fresh device-local baseline and a frozen 1.25× joint p95 TPOT/TTFT budget. PhaseGate's median paired retrieval-QPS gain was +80.50% over Fixed-1 and +3.02% over matched TimeGate across five held-out paired repeats. The TimeGate placed its high cap in actual DECODE for a median 12.30% of DECODE time. These M2 runs are reported independently and are not pooled with M4 Pro or earlier M2 Pro intervals.

## Scope sentence

The M2 result is a cross-device replication under the same 100k HNSW shape, not a replacement for the primary M4 Pro evaluation and not a clean remeasurement of the earlier M2 Pro campaign.

## Required qualification

All five held-out M2 pairs were pageout-free, strengthening the within-device interpretation.
