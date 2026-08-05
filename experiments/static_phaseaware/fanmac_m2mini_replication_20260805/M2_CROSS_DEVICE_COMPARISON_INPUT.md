# M2 Cross-device Comparison Input

| Device/chip | Memory | Commit | Index | Caps | Baseline p95 TPOT / TTFT | Phase asymmetry | Frozen policies | Budget | Median QPS (Fixed / PhaseGate / TimeGate) | Paired gains (P/F; P/T) | Pageout-free |
|---|---:|---|---|---|---|---|---|---:|---|---|---:|
| Mac mini / Apple M2 (`Mac14,3`) | 16 GiB | `9bd57c31bd100ae5fb1590dd326f51556c71da69` | FULL_100K, 100k×384 HNSW | 1 and 4→1 | 14.012 ms / 2187.092 ms | DECODE-sensitive | `fixed1` / `phasegate4to1` / `timegate4to1` | 1.25× | 844.8 / 1525.1 / 1483.5 | +80.50%; +3.02% | 5/5 |

Scope caveat: independent fan-cooled base-M2 cross-device replication; no pooled confidence interval and no direct QPS equivalence claim across devices.
