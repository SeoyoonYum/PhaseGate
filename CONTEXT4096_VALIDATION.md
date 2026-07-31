# Context-4096 prefill slowdown validation

## 1. Existing-result audit and apparent contradiction

The provenance audit is in
`experiments/phaseguard/context_validation/EXISTING_RESULT_AUDIT.md`. The old
context-4096 value was an isolated transformer-body prefill time, not queue time:
three single calls under HNSW-4 were 10.887, 12.701, and 13.111 s versus no-load
calls of 6.059, 4.882, and 5.347 s. Its reported “p95” was the p95 of those three
scalars. Each point used no local warmup and only a 2 s cooldown; loaded points
were later in the shuffled order. All three loaded points had zero pageout delta,
so pageout alone cannot explain the old observation.

At context 2048 the prior HNSW profile reported only +5.7% prefill p95, while its
decode p95 rose +61.3%. The contradiction is therefore between a small 2048
prefill effect and a large, sparsely sampled 4096 isolated effect—not between a
pipeline queue metric and Phase 0.

## 2. Experimental setup and controls

- Apple M4 MacBook Air (`Mac16,13`), 10 cores, 16 GB unified memory, fanless;
  macOS 26.5.1 (25F80), AC connected.
- Git `6b6089c53b8e66f8772b4397a611a2d28be7d741`; Python 3.13.12;
  MLX 0.31.2; MLX-LM 0.31.3; FAISS CPU 1.14.3; NumPy 2.4.6;
  Matplotlib 3.11.0.
- `mlx-community/Qwen2.5-1.5B-Instruct-4bit`; deterministic tensors with exact
  shapes `(1, 2048)` and `(1, 4096)`; forced MLX evaluation through the existing
  separated prefill/decode path.
- FAISS `IndexHNSWFlat`, 100,000 x 384 vectors, M=32,
  `efConstruction=80`, `efSearch=128`, top-k=10; persistent spawned processes;
  32 queries per background task.
- Synthetic control: the existing native 96 MiB random-gather loop. Its process
  now exits gracefully on SIGTERM so measured logical throughput is retained.
- Primary isolated attempt: 3 warmups and 20 measured prefills per point; 128
  decode intervals; randomized context/workload/phase order. Targeted attempt 9
  used 5 warmups, 20 iterations, 120 s cooldown, and 60 s idle windows. Targeted
  attempt 12 used five measured prefills to remain before the observed thermal
  transition.
- Pipeline: one GPU worker, concurrency 4, 128 output tokens, uncoordinated policy.
  Matched diagnostic runs completed 12 requests; one 52-request no-load anchor
  tested whether the requested long run could remain stationary.

The 2048 and 4096 points use separate same-context no-load baselines. Pure prefill
excludes retrieval and GPU queue time. Pipeline TTFT/queue-inclusive values remain
separate from `prefill_ms`.

## 3. Clean-run criteria

Immediately before and after each measurement the harness records `vm_stat`,
`vm.swapusage`, `memory_pressure`, model/worker RSS, experiment-process RSS,
reclaimable headroom, pagein/pageout, compressor and swap counters. It also records
30--60 s idle pageout rates. A strict clean run requires zero experiment-window
pageout, zero swap-used growth, no severe memory-pressure state, at least 4 GiB
headroom, no failure, and no material first/last latency drift. Suspect records are
preserved. A cumulative nonzero macOS pageout counter is never itself disqualifying.

No run increased swap usage. Headroom was normally about 6.2--7.3 GB. Direct
temperature and GPU-clock telemetry was unavailable, so `temperature_c` and
`gpu_frequency_mhz` remain null. Thermal classification therefore uses run order,
cooldown response, and within-run latency drift; this is a limitation, not a direct
frequency measurement.

## 4. Completed matrix, failures, and contamination

The primary matrix completed both contexts, four CPU conditions (none, HNSW-1,
HNSW-4, random), and isolated prefill/decode in attempt 1. Attempts 2 and 3 repeated
the matrix but were run with inadequate inter-attempt recovery and are retained as
thermal-protocol-suspect. Attempt 4 was interrupted after an orchestration stop;
its orphan workers were terminated and no incomplete point is used. Attempt 9
completed the long-cooldown HNSW-4/no-load block. Attempt 12 completed the short,
cooled random/no-load block.

Pipeline smoke passed. Matched 12-request no-load/HNSW-4/random runs completed at
both contexts. A 52-request context-2048 no-load run completed but became thermally
nonstationary (1.62x first/last prefill drift), demonstrating that the requested
50-request target cannot be treated as a clean baseline on this fanless device
under sustained execution. It is retained as contaminated rather than silently
discarded.

## 5. Isolated median and tail results

The primary full-matrix attempt showed a real HNSW-4 interference effect, but not a
special 4096 regime: prefill median/p95 rose +31.8%/+32.9% at 2048 and
+35.2%/+32.5% at 4096. Decode median/p95 rose +51.0%/+50.1% at 2048 and
+50.7%/+52.4% at 4096. The 4096 HNSW prefill point narrowly failed the drift rule
(1.106x) despite zero pageout and swap growth; the 2048 point was strict-clean.

The short cooled synthetic control restored the original prefill regime:
+4.1% median/+3.9% p95 at 2048 and +6.3%/+7.1% at 4096. The 4096 loaded point was
stationary (1.075x drift) but observed 59 pageout pages, so it remains explicitly
memory-suspect. It had no swap growth.

The decisive targeted HNSW sequence was nonstationary. At 4096, individual
prefills climbed from 5.159 s to 15.485 s within twenty measurements (2.849x
drift); its median/p95 slowdown was +117.5%/+163.0%. At 2048 in the same block,
HNSW prefill was 2.226--2.563 s and its full median slowdown was +13.9%. After
cooldown, the final 2048 no-load point was flat at 2.145 s median (1.005x drift).

| Context | Workload | Metric | Clean/stationary slowdown | Contaminated or suspect slowdown |
|---:|---|---|---:|---:|
| 2048 | HNSW-4 | Prefill median | +31.8% (attempt 1) | +115.3% (hot attempt 3) |
| 2048 | HNSW-4 | Prefill p95 | +32.9% | +102.0% (hot attempt 3) |
| 4096 | HNSW-4 | Prefill median | no strict-clean pair; +35.2% near-stationary | +117.5% (attempt 9) |
| 4096 | HNSW-4 | Prefill p95 | no strict-clean pair; +32.5% near-stationary | +163.0% (attempt 9) |
| 2048 | HNSW-4 | Decode p95 | +50.1% | n/a |
| 4096 | HNSW-4 | Decode p95 | +52.4% | n/a |
| 2048 | random | Prefill median | +4.1% | +100.4% (hot attempt 3) |
| 4096 | random | Prefill median | +6.3% stationary, pageout-observed | +48.8% (hot attempt 3) |

“Clean/stationary” does not erase the documented protocol caveat that attempt-1
baselines were warmer than the targeted short block. The raw and processed tables
retain both strict flags and fixed early-window metrics.

## 6. Isolated versus pipeline

Matched 12-request pipeline runs contradict a pipeline-only slowdown explanation.
HNSW-4 changed pure prefill median/p95 by only +2.9%/+7.3% at 2048 and
+2.5%/+5.6% at 4096. The 2048 HNSW run was strict-clean; both 4096 baseline and
loaded runs had small pageout deltas but were thermally stationary with no swap
growth. Queue-inclusive median changed by -34% at 2048 and -18% at 4096 because
retrieval pacing changed GPU queue occupancy; it is not GPU prefill interference.

Continuous random load gave +7.0%/+10.9% pure-prefill median/p95 at 2048 with
1.120x drift. At 4096 it gave +13.2%/+51.2%, but that run drifted 1.483x and is
thermal-contaminated. Decode p95 increased +23.1% and +20.0%, respectively. Thus
the pipeline instrumentation correctly exposes a long-context tail only when the
run itself becomes nonstationary.

## 7. HNSW versus synthetic random and memory/thermal diagnostics

The short synthetic run produces only 4--7% prefill slowdown across contexts,
while HNSW-4 can produce a sustained roughly 32--35% penalty even before collapse.
Therefore FAISS-HNSW and the one-process 96 MiB random gather are not equivalent
interference mechanisms or intensities. Context alone is insufficient to predict
the prefill effect.

Pageout delta is not the main explanation: the old slow HNSW points and several
new slow points had zero pageout, swap-used delta was always zero, and Spearman
association between pageout delta and prefill slowdown was negative (-0.27) in
this small heterogeneous set. Run order had a positive association (+0.45).
Neither is causal evidence. The strongest diagnostic is direct nonstationarity:
4096 HNSW rose 3x within one measurement, the 52-request no-load pipeline rose
1.62x, and latency recovered after cooldown. Compressor deltas did not track the
slowdown consistently; headroom never crossed the 4 GiB threshold.

## 8. Selected outcome: Category F — Thermal artifact

Category F is best supported for the **large apparent context-4096 slowdown**.
Longer individual prefills and sustained HNSW+Metal execution cross a fanless
device duration/power threshold sooner, causing large median and especially tail
inflation. The effect grows within runs and with accumulated execution, appears
even in a long no-load pipeline, and shrinks to 4--7% for short, stationary
synthetic controls. Stable matched pipeline HNSW points show nearly identical
small prefill effects at both contexts.

This does not prove temperature or GPU-frequency causality because those counters
were unavailable, and it does not claim zero real HNSW prefill interference. The
data support a roughly one-third sustained HNSW effect at both contexts in the
primary attempt, plus thermal amplification at 4096. Category A is not met because
there are no five repeatable, strict-clean isolated HNSW pairs showing a larger
4096 median effect under more than one workload. Category E is not met because
slowdown does not track pageout and swap never grew. Category D is not primary
because the original result was isolated, although pipeline metric boundaries are
an important secondary issue.

## 9. Revised claim and scheduler implication

Preserve the phase-asymmetry claim for stationary moderate-duration measurements.
Do not claim that context universally breaks it. HNSW workload intensity and
sustained duration can raise prefill latency, while long-context tails on this
fanless M4 are strongly confounded by thermal/power-state drift.

No scheduler redesign or learned predictor is justified by this validation. The
runtime should keep pure phase time separate from GPU queue time, and experiments
should enforce a stationarity/cooldown guard. If future direct telemetry confirms
a repeatable context threshold, a simple context/duration-aware prefill gate would
be sufficient before considering a learned policy.

Recommended paper wording:

> The apparent long-context prefill slowdown does not reproduce as a stable,
> context-specific effect: short stationary random-access measurements remain in
> the 4--7% regime, while sustained HNSW and long pipeline runs on the fanless M4
> exhibit strong within-run thermal/power-state drift. We therefore preserve the
> phase-asymmetry claim for stationary measurements and treat the earlier 4096
> tail as thermally contaminated.

## 10. Commands used

```bash
.venv/bin/python scripts/run_context_validation.py --attempt 1 --allow-battery
.venv/bin/python scripts/run_context_validation.py --attempt 9 \
  --workloads none,hnsw4 --phases prefill --prefill-warmups 5 \
  --prefill-iterations 20 --cooldown 120 --idle-seconds 60 --allow-battery
.venv/bin/python scripts/run_context_validation.py --attempt 12 \
  --workloads none,random --phases prefill --prefill-warmups 3 \
  --prefill-iterations 5 --cooldown 120 --idle-seconds 60 --allow-battery
.venv/bin/python scripts/run_context_pipeline.py --attempt 1 --context 2048 \
  --workload none --requests 50 --idle-seconds 30 --allow-battery
.venv/bin/python scripts/run_context_pipeline.py --attempt 2 --context 4096 \
  --workload none --requests 12 --idle-seconds 15 --allow-battery
# Attempts 3--7: matched HNSW-4/no-load/random 12-request pipeline points.
.venv/bin/python scripts/analyze_context_validation.py
```

Exact command lines and dependency metadata are saved per run under `logs/`.

## 11. Threats to validity and next experiments

- Direct die temperature, power, and GPU clock were unavailable; latency drift is
  a proxy. Repeat with supported telemetry or an external power/temperature logger.
- Five clean independent HNSW runs were infeasible because sustained 4096 runs
  repeatedly became nonstationary. A temperature-triggered rather than fixed-time
  cooldown is needed.
- The deterministic token tensor validates execution length but not prompt-content
  sensitivity.
- HNSW worker processes each load the index; sharing/mapping behavior may differ
  from another production retrieval runtime.
- Pipeline diagnostic runs use 12 requests. The one 52-request anchor established
  that long-run p95 is thermally contaminated, but 12 requests alone is a weak
  tail estimator.
- The small heterogeneous correlation sample is diagnostic only.

Next, repeat only no-load/HNSW-4 at both contexts on an actively cooled M4 or a
device with temperature/clock telemetry, trigger each point from a fixed thermal
state, and collect five independent stationary pairs. No broader scheduler matrix
is warranted before that control succeeds.
