# Observer Validation Abort

The first frozen validation stopped during the first condition. No paired observer
comparison was completed and no performance result from this directory is eligible for
an overhead or causal conclusion.

- Frozen commit: `5846df53e5190324435d711e5300ed9d4023b345`
- Freeze SHA-256: `c0fa32e9c5338436f8eaaafa4756a8499f17430d3fd79351cde4da479bb365dd`
- Affected condition: minimal, pair 0
- Attempt 1: complete measured block, hard-invalid because the post sentinel was
  +7.04% above the newly created stage reference.
- Attempt 2: stopped before measurement because the pre sentinel was +7.04% to +7.87%
  above the same reference.
- Pageout, swap, memory pressure, token count, timestamp, and event reconstruction were
  clean in the completed attempt.

The stage reference had been frozen at 1677.70 ms from the short initial warm-up. The
post-block and next-process measurements were 1795.8--1809.7 ms, showing that the first
reference represented a cold/boost state rather than the long-block plateau. The retry
limit was not exceeded. The freeze and raw attempts remain unchanged.

The correction changes only pre-measurement sentinel conditioning. A new commit and a
new observer-validation directory are required. The model, workload, timestamp
placement, observer modes, metric definitions, and acceptance gates remain unchanged.
