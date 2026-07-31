# PhaseGate backlog-demand validation

This experiment fixes the original closed-loop workload's accidental natural
staggering. Request A enters decode first; every subsequent decode boundary then
injects a fixed number of future requests (1/2/4 for low/medium/high), so multiple
application-owned FAISS-HNSW tasks can execute or queue during decode. The rule is
identical across policies even though wall-clock arrival times reflect policy speed.

The primary matrix must run on a fan-cooled Mac. The runner refuses a MacBook Air
unless `--smoke --allow-fanless-smoke` is explicitly used. Before every block, a
no-load prefill sentinel must return to within ±5% of its stored reference.

## Measured demand, not configured workers

The 2 ms demand sampler records:

- workers actually inside an HNSW search chunk;
- decode/retrieval overlap fraction;
- permitted worker count and the fraction for which the permit binds;
- queued tasks, claimed-but-paused tasks, effective backlog, and outstanding tasks;
- pure/application TTFT, TPOT, application goodput, and retrieval goodput.

Requested `low`, `medium`, and `high` labels are accepted only if the
uncoordinated calibration measures the intended active-retrieval band. Adjust the
arrival trace and rerun calibration if `target_met` is false.

## Fan-cooled execution

```bash
# 1. Calibration: uncoordinated demand plus fixed k=1..4, randomized.
.venv/bin/python scripts/run_phasegate_backlog_matrix.py \
  --stage calibration --repeats 3

# 2. Verify demand bands and select fixed-k using calibration only.
.venv/bin/python scripts/analyze_phasegate_backlog.py

# 3. Held-out randomized evaluation using the saved fixed-k map.
.venv/bin/python scripts/run_phasegate_backlog_matrix.py \
  --stage evaluation --repeats 3

# 4. Final tables and plots.
.venv/bin/python scripts/analyze_phasegate_backlog.py
```

Each block is durable and resumes by run key. Exact commands, randomized order,
sentinel attempts, environment versions, and host model are stored in `logs/`.
Calibration and evaluation remain separate; the best fixed-k is never selected
from held-out evaluation runs.

## Smoke test on the current fanless development host

```bash
.venv/bin/python scripts/run_phasegate_backlog.py \
  --policy uncoordinated --demand high --repeat 99 --split calibration \
  --smoke --allow-fanless-smoke --allow-battery
```

Fanless smoke results validate mechanics only and must not be combined with the
fan-cooled primary results.
