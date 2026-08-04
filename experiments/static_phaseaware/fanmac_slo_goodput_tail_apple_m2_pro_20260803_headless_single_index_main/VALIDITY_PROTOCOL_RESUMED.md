# Resumed campaign validity protocol

Effective 2026-08-03, this campaign resumes from calibration 7/45 without
changing the scientific workload or runtime environment.

Hard failures stop the campaign: process crash/hang, swap growth, warning or
critical memory pressure, missing/corrupt request or token data, invalid or
non-monotonic token timestamps, incorrect model/retrieval output, incorrect
policy semantics, or an unreachable/unstable host.

Soft flags are retained as metadata: positive global pageouts with zero swap
and normal memory pressure, sentinel deviation, TPOT/TTFT/retrieval-QPS drift,
slow recovery, and small background-memory fluctuation. A soft-flagged block
gets at most one cleanup/retry; both raw attempts remain preserved. The matrix
uses the clean retry when present, otherwise the latest complete soft-flagged
attempt for that paired repeat.

Final reporting will provide an inclusive non-hard-failure analysis and a
clean-run sensitivity analysis, including flag counts/types and whether the
conclusion changes.
