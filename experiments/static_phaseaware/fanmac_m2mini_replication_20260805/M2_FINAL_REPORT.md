# M2 Final Report

1. **Device:** Mac mini `Mac14,3`, Apple M2, 4P+4E CPU cores, 10 GPU cores, 16 GiB unified memory.
2. **Memory mode:** `FULL_100K`; the 100k preflight had no swap growth, pressure failure, or pageout delta in any policy.
3. **CPU-only scaling:** cap 4 delivered 2.283× cap-1 median HNSW QPS.
4. **Phase asymmetry:** the directional DECODE-sensitivity criterion was met.
5. **Exact 4→1 feasibility:** feasible at the frozen device budget.
6. **PhaseGate vs Fixed-1:** median paired QPS gain +80.50%.
7. **PhaseGate vs TimeGate:** median paired QPS gain +3.02%.
8. **TimeGate DECODE overlap:** median high-cap overlap 12.30%.
9. **Pageout-free pairs:** 5/5.
10. **Scope:** M2 metrics and intervals remain separate from M4 Pro and old M2 Pro. The optional 512-token point is directional and was not recalibrated.
