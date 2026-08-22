# Base-M4 Paper Update Recommendations

These are proposed replacements only. Do not edit the LaTeX automatically.

## Replacement abstract sentences

"On a fan-cooled base-M4 Mac mini, PhaseGate increased retrieval throughput by a median 100.88% over the best calibrated continuous Fixed policy under a frozen 1.25x joint TTFT/TPOT budget across seven paired randomized repeats."

"Against a calibration-matched phase-blind TimeGate schedule, PhaseGate improved retrieval throughput by 2.36% (95% paired bootstrap interval 1.77%--2.57%) while satisfying the joint latency budget in 7/7 blocks versus 0/7 for TimeGate."

"The PhaseGate-versus-Fixed throughput gain decreased from 130.39% at 64 output tokens to 43.29% at 512 as the measured PREFILL fraction fell from 69.08% to 20.65%."

## Replacement contribution bullets

- "We provide a phase-blind, duty-cycle-matched TimeGate control that replays calibration-derived high/low cap intervals without access to LLM phase state."
- "We report seven paired randomized base-M4 repeats using frozen Fixed-1, PhaseGate 4->1, and TimeGate 4/1 policies, with run-level paired bootstrap intervals and complete per-repeat results."
- "We quantify workload-shape dependence at 64, 128, and 512 output tokens using event-derived PREFILL/DECODE wall time rather than percentile-derived phase estimates."
- "We audit swap, memory pressure, and pageouts per block and retain pageout-flagged but otherwise valid blocks rather than selecting favorable repeats."

## Paper-ready primary-results paragraph

"At the frozen primary budget B=1.25, calibration selected Fixed-1 and PhaseGate 4->1. In seven held-out paired randomized repeats, PhaseGate achieved a median 1574.2 retrieval QPS versus 783.7 for Fixed-1, a median paired gain of 100.88% (95% run-level bootstrap interval 100.81%--101.07%). Both policies passed the joint p95 TTFT/TPOT budget in 7/7 blocks. PhaseGate's median normalized p95 TPOT and TTFT were 1.181 and 1.046, respectively."

## Paper-ready phase-alignment-control paragraph

"The matched phase-blind TimeGate control achieved 1537.6 median retrieval QPS but passed the joint latency budget in 0/7 blocks because its median normalized p95 TPOT was 1.603. PhaseGate exceeded TimeGate throughput by a median 2.36% (1.77%--2.57%) while passing in 7/7 blocks. Event reconstruction showed that TimeGate ran cap 4 during 23.11% of actual DECODE time, whereas PhaseGate did so during only 0.003%. This comparison supports a bounded phase-alignment benefit beyond merely alternating between high and low concurrency."

## Paper-ready workload-shape paragraph

"With the Fixed-1 and PhaseGate 4->1 policies frozen at the 128-token calibration, PhaseGate's median paired retrieval-QPS gain decreased monotonically from 130.39% at 64 tokens, to 101.37% at 128, and 43.29% at 512. The corresponding event-measured PhaseGate PREFILL fractions were 69.08%, 51.60%, and 20.65%. The trend supports dependence on available PREFILL overlap time over the measured range; it does not establish behavior outside these lengths."

## Paper-ready limitations paragraph

"The evaluation is device- and workload-specific: a base-M4 Mac mini, Qwen2.5-1.5B-Instruct 4-bit under MLX, 2,048-token context, and output lengths of 64--512. Only one of seven primary triplets was completely pageout-free, although all 21 blocks had zero swap growth and normal memory pressure; therefore we do not claim pageout neutrality. The 512-token isolated baseline showed run-level p95 TPOT tail variability (17.018, 13.124, 17.574 ms), so normalized tail metrics at that length require caution. Phase fraction and output length covary, and the observed association should not be extrapolated beyond the measured workloads or devices."

## Proposed primary table rows

| Policy | Retrieval QPS | normalized p95 TPOT | normalized p95 TTFT | joint passes |
|---|---:|---:|---:|---:|
| fixed1 | 783.7 | 1.165 | 1.006 | 7/7 |
| phasegate4to1 | 1574.2 | 1.181 | 1.046 | 7/7 |
| timegate4to1 | 1537.6 | 1.603 | 1.043 | 0/7 |

## Proposed output-shape table rows

| Output tokens | PREFILL fraction | Fixed QPS | PhaseGate QPS | paired gain [95% interval] |
|---:|---:|---:|---:|---:|
| 64 | 69.08% | 811.0 | 1869.5 | 130.39% [130.15, 131.19] |
| 128 | 51.60% | 782.1 | 1573.3 | 101.37% [101.10, 101.59] |
| 512 | 20.65% | 726.0 | 1041.0 | 43.29% [43.22, 43.69] |

## Proposed figure captions

- **Primary control figure:** "Per-repeat retrieval-QPS gains for frozen PhaseGate 4->1 relative to Fixed-1 and calibration-matched phase-blind TimeGate 4/1 on a fan-cooled base-M4 Mac mini. Points are seven paired randomized repeats; intervals use 10,000 run-level paired bootstrap resamples."
- **Output-length figure:** "Median paired PhaseGate-versus-Fixed retrieval-QPS gain at 64, 128, and 512 output tokens. Policies were frozen at 128 tokens; error bars are 95% run-level paired bootstrap intervals over five pairs per length."
- **PREFILL-fraction figure:** "Paired retrieval-QPS gain versus event-measured PREFILL wall-time fraction. The association is descriptive because output length changes both phase fraction and other runtime behavior."
- **Pageout figure:** "Paired PhaseGate-versus-Fixed retrieval-QPS gain versus the PhaseGate-minus-Fixed global pageout delta. The n=7 correlation is descriptive and does not establish absence of pageout confounding."

## Statements that must not be made

- Do not call this hardware M4 Pro or generalize the result to all Apple Silicon.
- Do not claim the seven policy blocks are independent replicates; they are seven paired randomized repeats.
- Do not claim TimeGate was phase-aware or tuned on held-out phase overlap.
- Do not claim PhaseGate eliminates contention or always improves latency.
- Do not claim the 100.88% 128-token gain is universal; the measured gain was length-dependent.
- Do not claim pageout-free or pageout-neutral primary execution; only 1/7 triplets were completely pageout-free.
- Do not interpret normalized 512-token TPOT below 1 as evidence that retrieval accelerates decode.
- Do not use individual requests as bootstrap units for the primary comparison.
- Do not merge raw QPS across this base-M4 campaign and older M2-Pro/MacBook-Air campaigns.
- Do not change `K_hi`, the selected policies, or `primary_B` after observing held-out results.
