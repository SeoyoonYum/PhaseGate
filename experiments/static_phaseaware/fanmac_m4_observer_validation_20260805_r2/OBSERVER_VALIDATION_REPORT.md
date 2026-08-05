# Base-M4 Observer Validation Report

## Scope

Diagnostic intervention only; no policy or paper claim is drawn from these blocks.

## Minimal versus event

- Median paired request-median TPOT change (event/minimal): -0.040%
- Median paired all-token mean-gap change (event/minimal): -0.065%
- Production correctness gate: PASS
- Production overhead gate: PASS

| Pair | Minimal median TPOT | Event median TPOT | Paired change | Minimal gap mean | Event gap mean | Paired change |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11.499 | 11.502 | 0.024% | 11.518 | 11.510 | -0.065% |
| 1 | 11.534 | 11.522 | -0.112% | 11.552 | 11.534 | -0.157% |
| 2 | 11.530 | 11.526 | -0.040% | 11.551 | 11.544 | -0.065% |

## Legacy versus event

The r4-derived tail incidence was similar and does not support the legacy observer as the cause of that old slow state. A common OS/runtime stall remains plausible, but these quiet validation blocks do not establish its cause.

- Median paired request-p95 median change (event/legacy): -3.387%
- Median paired all-token mean-gap change (event/legacy): -0.512%
- Median paired official p95 change (event/legacy): -2.984%
- Legacy was slower on these broad metrics even though it did not increase the diagnostic >14 ms count.

The 14 ms threshold is r4-derived and diagnostic, not preregistered paper evidence.

| Pair | Legacy tail count | Event tail count | Delta (event-legacy) | Legacy official p95 | Event official p95 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 2 | 2 | 12.465 | 12.251 |
| 1 | 0 | 0 | 0 | 12.477 | 12.247 |
| 2 | 0 | 0 | 0 | 12.495 | 12.005 |
| 3 | 0 | 0 | 0 | 12.460 | 11.988 |
| 4 | 0 | 0 | 0 | 12.514 | 12.141 |

## Restart gate

Observer correctness and overhead gate: PASS.
Legacy mode is disabled for any subsequent campaign regardless of the causal diagnostic outcome.
