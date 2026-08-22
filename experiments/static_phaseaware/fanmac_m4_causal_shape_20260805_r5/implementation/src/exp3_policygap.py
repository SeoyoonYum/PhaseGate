#!/usr/bin/env python3
"""
exp3_policygap.py - Exp 3: policy gap, current frameworks vs a measured-cost oracle, with
an online causal heuristic. [RQ3/H3]  Measurement-only SIMULATION (no model execution;
scores decisions with the Exp1 cost model). See EXPERIMENTS.md Exp 3 (+ causal spec).

Multi-sequence eviction model: `concurrency` conversations are resident at once and share a
total KV budget; their turns interleave (round-robin), so an idle conversation's KV sits
resident until evicted. When resident KV exceeds budget, the policy chooses victims.

Policies:
  rotating(W)       - mlx-lm default: cap each conv's cache at the last W tokens (cheap,
                      bounded, but SILENTLY DROPS older context -> coverage < 1).
  full_keep         - never evict (cheap resume) -> over budget = crash.
  always_recompute  - drop every conv's KV between turns -> 0 idle footprint, recompute
                      EVERY turn (high TTFT).
  lru               - evict least-recently-used (recency only; causal's ancestor).
  causal            - online heuristic: evict argmax  kv_bytes / (P_reuse * recompute_cost),
                      P_reuse from PAST distance (recency, step at K). = LRU + a recompute-cost
                      axis (Exp1's "recompute is superlinear in N", turned into a policy signal).
  oracle            - SAME score, P_reuse from FUTURE distance (true reuse). The ONLY
                      difference from causal is past-vs-future reuse info -> upper bound.

Key output: recovery = how much of the LRU->oracle gap (mean TTFT) causal closes with no
foresight. Cost model: prefill_ms(N)/decode_ms(N) from results/csv/exp1_<model>_*.csv;
KV bytes/token from the _meta.json. Pure arithmetic - no MLX, no model load.
"""
import argparse
import bisect
import csv
import glob
import json
import math
import os
import random
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import workloads  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(REPO, "results", "csv")
FIG_DIR = os.path.join(REPO, "results", "figures")
POLICIES = ("rotating", "full_keep", "always_recompute", "lru", "causal", "oracle")


def _find_cost_files(model):
    csvs = [c for c in sorted(glob.glob(os.path.join(CSV_DIR, f"exp1_{model}_*.csv")))
            if "smoke" not in c and "_meta" not in c]
    metas = sorted(glob.glob(os.path.join(CSV_DIR, f"exp1_{model}_*_meta.json")))
    if not csvs or not metas:
        sys.exit(f"[exp3] need Exp1 cost data for {model} in results/csv/ (exp1_{model}_*.csv)")
    return csvs[-1], metas[-1]


class CostModel:
    """prefill_ms(N) / decode_ms(N) from the Exp1 curves (log-log interp + power-law
    extrapolation), plus KV bytes/token. Shared by every policy (incl. oracle & causal)."""

    def __init__(self, model):
        csv_path, meta_path = _find_cost_files(model)
        Ns, pre, dec = [], [], []
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                Ns.append(int(r["N"])); pre.append(float(r["prefill_ms_med"]))
                dec.append(float(r["decode_ms_med"]))
        order = np.argsort(Ns)
        self.Ns = [Ns[i] for i in order]
        self.pre = [pre[i] for i in order]
        self.dec = [dec[i] for i in order]
        self.kv_bpt = json.load(open(meta_path))["kv_bytes_per_token"]
        self.src = os.path.basename(csv_path)

    def _loglog(self, N, ys):
        N = max(1.0, float(N))
        lx, lxs, lys = math.log(N), [math.log(x) for x in self.Ns], [math.log(y) for y in ys]
        if N <= self.Ns[0]:
            sl = (lys[1] - lys[0]) / (lxs[1] - lxs[0])
            return math.exp(lys[0] + sl * (lx - lxs[0]))
        if N >= self.Ns[-1]:
            sl = (lys[-1] - lys[-2]) / (lxs[-1] - lxs[-2])
            return math.exp(lys[-1] + sl * (lx - lxs[-1]))
        return math.exp(float(np.interp(lx, lxs, lys)))

    def prefill_ms(self, N):
        return self._loglog(N, self.pre) if N > 0 else 0.0

    def decode_ms(self, N):
        return self._loglog(N, self.dec)

    def incr_prefill_ms(self, a, b):
        if b <= a:
            return 0.0
        return self.prefill_ms(b) - self.prefill_ms(a) if a > 0 else self.prefill_ms(b)

    def decode_sum_ms(self, ctx0, n):
        return n * self.decode_ms(ctx0 + n / 2.0) if n > 0 else 0.0

    def kv_bytes(self, N):
        return self.kv_bpt * N


def build_schedule(convs, concurrency, think_mean=20.0, seed=0):
    """Realistic interleaving: conversations arrive staggered and each conversation's turns
    are separated by a VARIABLE think-time (user reply gap), so KV reuse distance is
    heterogeneous like real serving (unlike a uniform round-robin, where recency perfectly
    predicts reuse and LRU is trivially optimal). `concurrency` sets the arrival rate so
    roughly that many conversations are in flight; think-time is lognormal (some users
    reply fast, some slow). Returns the time-ordered [(conv_id, turn_idx)]."""
    rng = random.Random(seed)
    interarr = max(think_mean * 4.0 / max(concurrency, 1), 1e-6)
    timed, t0 = [], 0.0
    for cid, conv in enumerate(convs):
        t0 += rng.expovariate(1.0 / interarr)            # staggered arrivals
        at = t0
        for ti in range(len(conv)):
            timed.append((at, cid, ti))
            at += 1.0 + rng.lognormvariate(math.log(think_mean), 0.8)  # variable think-time
    timed.sort()
    return [(c, ti) for (_, c, ti) in timed]


def simulate(convs, events, cm, policy, budget, W, K):
    """Replay the interleaved schedule under `policy`. Returns aggregate metrics."""
    conv_events = {}
    for idx, (c, _) in enumerate(events):
        conv_events.setdefault(c, []).append(idx)

    def next_use(c, e):
        lst = conv_events[c]; j = bisect.bisect_right(lst, e)
        return lst[j] if j < len(lst) else math.inf

    ctx, resident, res_tok, last_used = {}, {}, {}, {}
    total, ttfts = 0.0, []
    peak, over_budget_events, recomputes = 0, 0, 0
    covered, ctx_total = 0, 0

    def memory():
        return sum(cm.kv_bytes(res_tok[c]) for c in resident if resident[c])

    def drop_score(x, p_reuse):
        # drop_gain = keep_cost − P_reuse·recompute_cost, both in ms. keep_cost = time to
        # read that KV once (~92 GB/s, Exp2); recompute_cost = Exp1 prefill(N) (superlinear).
        # Evict argmax: keep convs that are reused soon AND expensive to recompute.
        keep_ms = cm.kv_bytes(res_tok[x]) / 92e6
        return keep_ms - p_reuse * cm.prefill_ms(ctx[x])

    for e, (c, ti) in enumerate(events):
        p, r = convs[c][ti]
        prior = ctx.get(c, 0)
        was_res = resident.get(c, False)
        if policy == "rotating":
            ttft = cm.incr_prefill_ms(min(prior, W), min(prior + p, W))
        elif was_res:
            ttft = cm.incr_prefill_ms(prior, prior + p)
        else:
            ttft = cm.prefill_ms(prior + p)                 # recompute prior context
            if prior > 0:
                recomputes += 1
        total += ttft + cm.decode_sum_ms(prior + p, r)
        ttfts.append(ttft)

        ctx[c] = prior + p + r
        resident[c] = True
        res_tok[c] = min(ctx[c], W) if policy == "rotating" else ctx[c]
        last_used[c] = e
        seen = min(prior + p, W) if policy == "rotating" else prior + p
        covered += seen; ctx_total += (prior + p)

        if policy == "always_recompute":
            for x in resident:                              # keep nothing idle
                resident[x] = False; res_tok[x] = 0
        elif policy in ("full_keep", "rotating"):
            if memory() > budget:
                over_budget_events += 1                     # cannot free -> crash/over-budget
        else:                                               # lru / causal / oracle: evict idle
            while memory() > budget:
                cands = [x for x in resident if resident[x] and x != c]
                if not cands:
                    over_budget_events += 1; break
                if policy == "lru":
                    v = min(cands, key=lambda x: last_used[x])
                elif policy == "causal":
                    v = max(cands, key=lambda x: drop_score(
                        x, math.exp(-(e - last_used[x]) / K)))            # recency estimate
                else:  # oracle: same score, TRUE future reuse (will it EVER return?)
                    v = max(cands, key=lambda x: drop_score(
                        x, 0.9 if next_use(x, e) < math.inf else 0.1))
                resident[v] = False; res_tok[v] = 0
        peak = max(peak, memory())

    n = len(convs)
    return dict(total_ms=total, mean_ttft=float(np.mean(ttfts)),
                p95_ttft=float(np.percentile(ttfts, 95)), peak_mb=peak / 1e6,
                over_budget_rate=over_budget_events / max(len(events), 1),
                recomputes=recomputes, coverage=covered / ctx_total if ctx_total else 1.0)


def run(args):
    cm = CostModel(args.model)
    print(f"[exp3] cost model from {cm.src} (KV {cm.kv_bpt} B/tok)")
    if args.trace and os.path.exists(args.trace):
        convs = workloads.load_sharegpt(args.trace, max_convs=args.max_convs)
        src = os.path.basename(args.trace)
    else:
        convs = workloads.synthetic_conversations(n_convs=args.n_convs, seed=0)
        src = f"synthetic(n={args.n_convs})"
        if args.trace:
            print(f"[exp3] trace '{args.trace}' not found -> synthetic")
    print(f"[exp3] traces: {src}  {workloads.trace_stats(convs)}")
    events = build_schedule(convs, args.concurrency, args.think_mean)
    budget = args.kv_budget_gb * 1e9
    print(f"[exp3] {len(events)} turn-events | concurrency {args.concurrency} | "
          f"shared KV budget {args.kv_budget_gb} GB | rotating W={args.window} | K-sweep {args.ks}\n")

    Ks = [int(x) for x in str(args.ks).split(",")]
    res = {}
    for pol in POLICIES:
        if pol == "causal":
            for K in Ks:
                res[f"causal_K{K}"] = simulate(convs, events, cm, "causal", budget, args.window, K)
        else:
            res[pol] = simulate(convs, events, cm, pol, budget, args.window,
                                Ks[len(Ks) // 2] if pol == "oracle" else 0)
    for name, a in res.items():
        print(f"  {name:13s} | total {a['total_ms']/1e3:8.1f}s | TTFT mean {a['mean_ttft']:7.1f} "
              f"p95 {a['p95_ttft']:8.1f} ms | peak {a['peak_mb']:6.0f}MB | "
              f"over-budget {a['over_budget_rate']*100:4.0f}% | recomp {a['recomputes']:5d} | "
              f"ctx {a['coverage']*100:4.0f}%")
    analyze(res, Ks)
    date = datetime.now().strftime("%Y%m%d")
    os.makedirs(CSV_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(CSV_DIR, f"exp3_{args.model}_{date}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy"] + list(next(iter(res.values())).keys()))
        for name, a in res.items():
            w.writerow([name] + [round(v, 4) for v in a.values()])
    print(f"\n[out] wrote {path}")
    plot(res, Ks, os.path.join(FIG_DIR, f"exp3_policygap_{args.model}_{date}.png"), args.model)


def _recovery(res, K, metric="mean_ttft"):
    """% of the LRU->oracle gap that causal(K) closes (and the always_recompute->oracle span)."""
    o, lru, ar = res["oracle"][metric], res["lru"][metric], res["always_recompute"][metric]
    cz = res[f"causal_K{K}"][metric]
    rec_lru = 100 * (lru - cz) / (lru - o) if lru != o else float("nan")
    rec_ar = 100 * (ar - cz) / (ar - o) if ar != o else float("nan")
    return rec_lru, rec_ar


def analyze(res, Ks):
    print("\n================ RECOVERY (mean TTFT) ================")
    o = res["oracle"]["mean_ttft"]
    print(f"  oracle {o:.1f}ms  |  lru {res['lru']['mean_ttft']:.1f}  |  "
          f"always_recompute {res['always_recompute']['mean_ttft']:.1f}ms")
    for K in Ks:
        rl, ra = _recovery(res, K)
        print(f"  causal K={K}: {res[f'causal_K{K}']['mean_ttft']:.1f}ms  -> recovers "
              f"{rl:.0f}% of LRU->oracle gap  ({ra:.0f}% of always_recompute->oracle)")
    print("  (30-60% recovery of the LRU->oracle gap = strong: recency + a recompute-cost")
    print("   axis closes much of the foresight gap with NO future info.)")
    print("======================================================")


def plot(res, Ks, png, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["rotating", "full_keep", "always_recompute", "lru"] + \
            [f"causal_K{K}" for K in Ks] + ["oracle"]
    order = [p for p in order if p in res]
    col = {"rotating": "tab:gray", "full_keep": "tab:blue", "always_recompute": "tab:orange",
           "lru": "tab:purple", "oracle": "tab:green"}
    cs = [col.get(p, "tab:red") for p in order]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    panels = [("mean_ttft", "mean TTFT (ms)"), ("over_budget_rate", "over-budget / crash rate (%)"),
              ("coverage", "context coverage (%)"), ("total_ms", "total latency (s)")]
    for axi, (key, title) in zip(ax.ravel(), panels):
        sc = 100 if key in ("over_budget_rate", "coverage") else (1e-3 if key == "total_ms" else 1)
        axi.bar(range(len(order)), [res[p][key] * sc for p in order], color=cs)
        axi.set_xticks(range(len(order))); axi.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        axi.set_title(title); axi.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"Exp3 policy gap + causal heuristic (cost-model sim) - {model}", fontsize=13)
    fig.tight_layout(); fig.savefig(png, dpi=150)
    print(f"[out] wrote {png}")


def main():
    ap = argparse.ArgumentParser(description="Exp3 policy gap + online causal heuristic (sim).")
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--trace", default="", help="ShareGPT/LMSys JSON path; empty -> synthetic")
    ap.add_argument("--max-convs", type=int, default=None)
    ap.add_argument("--n-convs", type=int, default=400, help="synthetic conversation count")
    ap.add_argument("--concurrency", type=int, default=16, help="conversations in flight at once")
    ap.add_argument("--think-mean", type=float, default=20.0,
                    help="mean think-time between a conversation's turns (turn-events)")
    ap.add_argument("--kv-budget-gb", type=float, default=2.0, help="shared resident KV budget")
    ap.add_argument("--window", type=int, default=4096, help="rotating cache window W (tokens)")
    ap.add_argument("--ks", default="2,4,8", help="causal recency K sweep")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
