# PhaseGuard results

## Final decision

**B. Measurement result with limited scheduler benefit.** Real HNSW retrieval
reproduced a strong phase asymmetry at context 2048: four workers increased decode
p95 by 61.3% but prefill p95 by only 5.7%. A fixed phase-only policy that permits
all CPU work during idle/prefill and zero workers during decode produced the best
repeatable policy point. The specified lookup rule (largest worker count predicted
to meet the TPOT SLO) selected one worker and did not improve materially over the
matching static-1 policy or uncoordinated execution. At context 4096, prefill also
slowed severely, so the clean asymmetry is not universal.

## 1. What was implemented

- Thread-safe `IDLE/PREFILL/DECODE` monitor with request, phase, first-token, and
  every-token timestamps.
- A fixed spawn-based CPU process pool. Every worker loads the persistent HNSW
  index once and checks a shared permit at query/chunk boundaries; workers are not
  recreated during measurement.
- Closed-loop clients following retrieval → queued MLX prefill → MLX decode, with
  one serialized GPU worker and application-owned CPU work overlapping it.
- Uncoordinated, serialized, static phase-only (decode permits 0/1/2/4), profile/SLO,
  and logical-throughput policies plus an evaluation-only best-observed oracle.
- Durable JSONL/CSV logging, resume keys, environment/memory/power metadata,
  aggregation, and Figures A-E in PNG/PDF.

## 2. Hardware and software

- MacBook Air `Mac16,13`, Apple M4, 10 CPU cores (4 performance, 6 efficiency),
  16 GB single-pool unified memory, fanless.
- macOS 26.5.1 (25F80), Darwin 25.5.0, Python 3.13.12.
- MLX 0.31.2, MLX-LM 0.31.3, NumPy 2.4.6, Matplotlib 3.11.0,
  FAISS CPU 1.14.3; git base `6b6089c53b8e66f8772b4397a611a2d28be7d741`.
- Qwen2.5-1.5B-Instruct 4-bit, 12 GB MLX memory limit, fixed token IDs/seeds.
- Persistent 100,000-vector, 384-dimensional FAISS HNSW index, M=32,
  `efConstruction=80`, `efSearch=128`, top-k=10; 180,820,834 bytes on disk.
- Runs reported AC power and charging. Rootless timing-spread telemetry was used;
  sudo-only temperature/GPU-clock/power sampling was unavailable and omitted.

## 3. Commands used

The exact primary build/profile/evaluation/analyze commands are in
`experiments/phaseguard/README.md`. Smoke used a 10k index, context 512, 8 decode
tokens, one request/client, and each policy. The primary trace command was repeated
in randomized order for policies and `--repeat 0/1/2`; static sweeps used
`--decode-workers 0/2/4`; SLO sweeps used `--slo 1.05/1.10/1.20`. Context
generalization used:

```bash
.venv/bin/python scripts/run_phaseguard_profile.py \
  --index experiments/phaseguard/index/hnsw_100k_d384.faiss \
  --model 1.5B --context 4096 --workers 0,4 --max-workers 4 \
  --ef-search 128 --decode-steps 128 --repetitions 3 \
  --queries-per-task 32 --cooldown 2 --tag ctx4096 --allow-battery
```

## 4. Completed experiments

- All code compiled; five policy smoke tests and all five plot paths completed.
- Stage B: 24 primary profile points (2 phases × 4 worker counts × 3 repeats).
- Primary policy comparison: 15 runs, 120 requests total, three independent runs
  per required policy at concurrency 4 and SLO 1.15.
- Three independent static-0 runs; one static-2 and one static-4 run.
- SLO 1.05, 1.10, 1.15, and 1.20 evaluations.
- One 6-request run per required policy at concurrency 2.
- Twelve context-4096 profile points (2 phases × 2 worker counts × 3 repeats).
- In total, raw files contain 40 profile records, 36 pipeline runs, and 244 request
  records including smoke/pilot runs. The processed evaluation selects 28 `eval`
  runs; pilots are retained but excluded by tag.

## 5. Failed or incomplete experiments

- The first in-sandbox GPU smoke failed loudly because Metal was unavailable;
  rerunning with host device permission succeeded. No partial measurement was used.
- `hnswlib` was absent. FAISS `IndexHNSWFlat` was used as the specified fallback;
  no compilation attempt delayed measurement.
- Context 4096 did not reproduce prefill immunity: four workers caused 2.18× prefill
  p95 and 1.56× decode p95, with high timing spread. It is a completed negative
  generalization result, not discarded data.
- No 3B/7B or second-device run was attempted after completing the staged second
  context. No open-loop trace or optional file-search workload was needed.
- `powermetrics` was not used because it requires sudo. No learned predictor was
  built because the go criterion for Stage E was not met.

## 6. Key numerical results

### Isolated phase profile, context 2048

| HNSW workers | Prefill p95 (ms) | Prefill slowdown | Decode p95 (ms) | Decode slowdown |
|---:|---:|---:|---:|---:|
| 0 | 2052.6 | 1.000× | 13.96 | 1.000× |
| 1 | 2108.9 | 1.027× | 14.53 | 1.040× |
| 2 | 2140.0 | 1.043× | 16.67 | 1.194× |
| 4 | 2169.2 | 1.057× | 22.52 | 1.613× |

### Policy medians, context 2048/concurrency 4/SLO 1.15

| Policy | Runs | requests/s | p95 TPOT (ms) | token SLO violation | p95 end-to-end (ms) |
|---|---:|---:|---:|---:|---:|
| Uncoordinated | 3 | 0.1637 | 15.27 | 1.87% | 26,860 |
| Serialized | 3 | 0.1590 | 14.19 | 0.79% | 25,470 |
| Static phase-only, decode=0 | 3 | **0.1740** | **14.45** | **0.59%** | **25,357** |
| Static phase-only, decode=1 | 3 | 0.1644 | 15.34 | 2.07% | 26,729 |
| Profile/SLO PhaseGuard | 3 | 0.1645 | 15.27 | 1.87% | 26,756 |
| Bandwidth-only | 3 | 0.1637 | 15.24 | 1.67% | 26,854 |

Static-0 improved median workflow throughput 6.3% and reduced p95 TPOT 5.4%
versus uncoordinated. Versus serialized, it improved throughput 9.4% while p95 TPOT
was 1.8% higher. At concurrency 2, static-0 achieved 0.1651 requests/s and 14.51 ms
p95 versus uncoordinated 0.1609/16.03 and serialized 0.1361/14.35 (single runs).
The evaluation-only best-observed point was a static-0 run at 0.2074 requests/s and
14.02 ms; this optimistic upper bound is not treated as a repeatable median.

## 7. RQ1: real-workload phase asymmetry

**Yes at context 2048.** HNSW crossed the 10% go threshold at two workers and the
decode effect was much larger than the prefill effect. **No clean generalization at
context 4096:** four workers changed prefill p95 from 5.99 to 13.07 s (+118%) and
decode p95 from 15.36 to 23.98 ms (+56%). Long prefill appears memory/thermal
sensitive on this fanless 16 GB device.

## 8. RQ2: static scheduling

**A useful but modest Pareto improvement exists.** Static-0 is not full
serialization: it allows four workers during GPU idle and prefill and pauses only at
decode chunk boundaries. It beat uncoordinated on both median workflow throughput
and p95 TPOT at concurrency 4, and was substantially faster than full serialization
at concurrency 2. The improvement is below 10% at the primary concurrency-4 median.

## 9. RQ3: profile lookup and bandwidth-only

The 1.15× lookup selected one decode worker, exactly matching static-1, and their
medians were indistinguishable (<0.2%). More importantly, maximizing the count of
SLO-safe workers was not the workflow optimum: static-0 completed GPU service faster
enough to offset less decode-time retrieval progress.

At SLO 1.05, the profile predicted one worker safe (14.53 ms ≤ 14.66 ms), but the
held-out run measured 15.31 ms and a 12.6% token violation rate. SLO 1.10 also
underpredicted; SLO 1.20 selected two workers and remained safe. This exposes the
cost of profile underprediction.

The legacy workload pair remains the clearest proxy failure: random gather generated
only 2.39 logical GB/s yet slowed decode 28.3%, more than scan at 47.6 GB/s and 22.3%
slowdown. In this pipeline, bandwidth-only was notably worse at concurrency 2
(16.49 ms p95, 9.58% violations) than phase-aware static-0 (14.51 ms, 0.39%), though
that comparison has only one run and should not be generalized.

## 10. RQ4: learned prediction

Not justified. One fixed phase-only policy was better than profile lookup in the
primary configuration, while the context-4096 profile changed the phase behavior
itself. A learned model from this small, thermally nonstationary dataset would add
variance and data leakage risk rather than establish a credible improvement.

## 11. Threats to validity

- Every long evaluation observed 19-160 pageouts of 16 KiB each (about 0.3-2.6 MB)
  and is explicitly marked contaminated, although reclaimable headroom remained
  7.3-7.6 GB and the primary profile saw only 0-3 pageouts. This looks like background
  VM activity rather than index/model capacity pressure, but it can add jitter.
- The fanless device had no sudo telemetry. Timing spread reached 1.27-2.03 in long
  runs; random order and three repetitions reduce but do not remove thermal/DVFS bias.
- Policy processes were randomized but long-run cooldown occurred through model
  reload/warmup rather than a fixed inter-run delay.
- Body-only prefill follows the established repository methodology; production TTFT
  also includes output projection and tokenization.
- Retrieval was intentionally long (4096 HNSW queries) to sustain closed-loop overlap.
  Lower-intensity pilots naturally staggered to one active worker and showed little
  scheduling opportunity; benefits depend on retrieval intensity and concurrency.
- Concurrency-2, static-2/static-4, and SLO-edge points have one evaluation run.
- Only one model, one device, deterministic prompts, and a 100k index were evaluated.
- The oracle is best observed after evaluation, not a deployable policy.

## 12. Recommended paper claims

Supported:

- Real HNSW retrieval can strongly interfere with decode on single-pool unified
  memory, while 2048-token prefill is comparatively insensitive.
- Application-level phase shaping can protect decode without stopping CPU work
  during prefill/idle; a simple static policy provided the best measured tradeoff.
- Logical workload throughput alone is not a safe interference signal.
- A lookup that only maximizes SLO-safe CPU workers need not optimize workflow
  throughput and can underpredict held-out TPOT.

Not supported:

- Universal prefill immunity, a learned-policy advantage, cross-model/device
  generality, arbitrary OS-process control, or truly parallel GPU request execution.

## 13. Recommended next experiments

1. Repeat static-0/uncoordinated/serialized at concurrency 2 and static-2 at
   concurrency 4 to three runs with fixed 10-15 s cooldown and optional
   `powermetrics` on a fan-cooled device.
2. Profile contexts between 2048 and 4096 to locate where HNSW begins to affect
   prefill and add context-dependent prefill permits.
3. Change lookup selection from “largest safe worker count” to an offline estimate of
   total request service time, incorporating both retrieval progress and GPU slowdown.
4. Repeat central points on Qwen2.5 3B and a second Apple device, then vary index size
   and `efSearch` with held-out configuration splits.
5. Test shorter realistic retrieval traces and an open-loop arrival process; report
   conditions where natural staggering makes scheduling unnecessary.

---

## Context-4096 validation (2026-07-28)

This section is a post-report validation and supersedes the interpretation—not the
raw measurements—of the context-4096 prefill profile. Full methods and evidence are
in `CONTEXT4096_VALIDATION.md`.

The original 4096 “p95” was the p95 of three single prefill calls with no local
warmup and only 2 s cooldown. New 20-iteration measurements exposed severe
nonstationarity: in the targeted HNSW-4 run, 4096 prefill rose from 5.159 s to
15.485 s within one point (2.849x drift). A 52-request no-load pipeline also drifted
1.62x, showing that sustained execution alone can contaminate long-run tails on
this fanless M4. Swap usage never increased, headroom remained above 6 GiB, and
pageout delta did not positively track slowdown.

Short stationary synthetic-random controls measured prefill median/p95 slowdown of
+4.1%/+3.9% at context 2048 and +6.3%/+7.1% at 4096. Matched 12-request HNSW
pipeline runs measured pure-prefill median/p95 changes of +2.9%/+7.3% at 2048 and
+2.5%/+5.6% at 4096. The primary full isolated matrix did find sustained HNSW-4
prefill interference of roughly 32--35% at both contexts, so workload intensity
matters, but it did not reveal a unique 4096 regime.

**Validation outcome: Category F — thermal artifact** for the large apparent
context-4096 effect. Preserve the phase-asymmetry claim for stationary measurements;
do not claim universal prefill immunity or that context universally breaks the
asymmetry. No scheduler redesign or learned predictor is warranted from these data.
Future work should repeat the two-context HNSW control from a fixed measured thermal
state with temperature/clock telemetry or active cooling.
