# M4 Mini Contention Campaign Stop Report

## Outcome

The diagnostic contention campaign stopped before its 12-block LLM-only/Fixed-1/Fixed-2/Fixed-4 matrix because both preregistered isolated-baseline sets failed. Set A failed, one documented environment-only correction was applied, and the single pre-frozen set B also failed. No set C was run. No contention curve, normalized cap comparison, MacBook-Air comparison, or cap conclusion was produced.

The completed Base-M4 PhaseGate r5 campaign is separate and remains valid; this stop applies only to `fanmac_m4_contention_20260806_r1`.

## Frozen device and workload

- Fan-cooled base-M4 Mac mini, Mac16,10, 4P+6E CPU cores, 10 GPU cores, 16 GB unified memory.
- Qwen2.5-1.5B-Instruct 4-bit, revision `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`.
- Context 2,048; output 128; 150 measured requests per baseline repeat; five repeats per set.
- Event observer, <=1 Hz memory sampling, zero repeated `ps` subprocesses.
- Measurement code freeze: `d4562a71880a79d4ee5370c9da7e82aaabc87a3f`.
- `K_hi=2` remained frozen; cap 4 was diagnostic-only and was never executed.

## Baseline set A

- Median p95 TPOT: 12.315 ms.
- Median p95 TTFT: 1798.69 ms.
- TPOT/TTFT stability failures: 1/5; repeat 1 had 3.19% absolute TPOT deviation.
- Positive-pageout blocks: 4/5; deltas: 41, 5, 54, 0, 31.
- Swap growth: 0/5; warning/critical memory pressure: 0/5.

After set A, the machine was at 93% reported free memory, zero compressor pages, zero swap, AC power, and no thermal/performance warning. A quiet idle/cooldown window was documented in `BASELINE_B_ENVIRONMENT_CORRECTION.json`; the global pageout counter remained 1494 to 1494 during that window. No code, observer, model, workload, seed, or measurement definition changed.

## Baseline set B

- Median p95 TPOT: 12.090 ms.
- Median p95 TTFT: 1794.26 ms.
- Stability failures: 2/5. Repeat 2 TPOT deviation was 5.31%; repeat 3 was 4.13%.
- Positive-pageout blocks: 2/5; deltas: 0, 3, 0, 0, 2.
- Swap growth: 0/5; warning/critical memory pressure: 0/5.

Set B independently failed both the TPOT stability requirement and the zero-pageout requirement. Pooling A and B, deleting tail requests, or running until a favorable set appeared was forbidden.

## Harness and telemetry audit

- Valid baseline blocks: 10/10; each produced exactly 150 requests and 19,200 token events.
- Observer subprocess blocks: 0/10.
- Swap-growth blocks: 0/10.
- Memory-pressure failures: 0/10.
- Clean pre/post thermal status: 10/10.
- Request-, run-, timeline-, manifest-, and thermal/power data are retained in the campaign bundle.

The set B command manifests contain a misspelled `.faissa` index argument. This is recorded as an orchestration metadata defect. The condition was LLM-only, `shared_index_load_count` was zero in every baseline block, no retrieval query was submitted, and the FAISS path was never opened; therefore the typo did not alter baseline execution. The exact index was verified during campaign preparation. No contention block was run with the misspelled argument.

## Scientific disposition

- Do not report a Mini Fixed-1/2/4 contention curve from this campaign; none was collected.
- Do not compare Mini versus MacBook-Air normalized contention slopes from this campaign.
- Do not revise `K_hi=2`; cap 4 remained diagnostic-only and unexecuted.
- Do not treat the clean swap/pressure/thermal telemetry as overriding the failed frozen baseline gate.
- Any future attempt requires a new campaign directory and a written protocol amendment; it must not append to or relabel this campaign.
