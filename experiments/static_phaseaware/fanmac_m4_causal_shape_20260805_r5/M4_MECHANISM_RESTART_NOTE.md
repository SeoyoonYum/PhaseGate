# Mechanism Stage Restart Note

The first `m4_mechanism` invocation was stopped on 2026-08-06 because harness
development and Git activity occurred on the same machine while its first block
was in progress. No measured block completed and no `runs.jsonl` result row was
produced. Its freeze and logs are retained as diagnostic evidence and are never
used in analysis.

All orchestration code was finalized before creating `M4_MECHANISM_CLEAN_FREEZE.json`.
The valid mechanism stage is named `m4_mechanism_clean` and is executed without
concurrent development or interactive activity. Calibration reads only that
clean stage.
