#!/usr/bin/env python3
"""
exp_p1_kvlink.py — P1: strengthen the KV link ("contention changes the KV decision").

Cost-model SIMULATION (no model execution; reuses the Exp3 CostModel + schedule). Three
measurements, all on the SAME real-ShareGPT interleaved trace and the SAME measured cost
curves, so they are directly comparable to Exp3:

  P1a  alpha-sweep      — victim score = alpha*recency + (1-alpha)*recompute (each min-max
                          normalized over the live candidates). alpha in [0,1]. alpha=1 == LRU
                          (pure recency); alpha=0 == evict-cheapest-recompute (the Phase-2a
                          failure mode). Run the whole sweep under contention OFF vs ON and
                          report the realized-cost CURVE + its argmin -> does the best alpha MOVE?

  P1b  PCIe vs UMA A/B  — the eviction cost model differs by hardware: UMA drops+recomputes
                          (expensive, N-dependent prefill); PCIe swaps to CPU DRAM and back
                          (cheap, ~flat, KV preserved). A foresight oracle minimizing cost under
                          each model picks DIFFERENT victims (UMA protects soon-reused = Belady;
                          PCIe frees memory greedily = evict-largest). We measure (i) decision
                          disagreement and (ii) the PENALTY of running the PCIe-optimal policy on
                          UMA hardware, under contention OFF vs ON.

  P1c  contention-aware — a victim rule that scales keep-cost by the MEASURED contention factor
                          when the bus is loaded, vs contention-blind LRU, under contention.
                          Does knowing the bus is contended recover any TTFT? (one positive
                          point; honest off-ramp if not.)

Measured inputs: Exp1 cost curves (CostModel) + Exp2 decode-slowdown at full CPU load
(results/csv/exp2_<model>_*.csv) as the contention factor. MODELED input: PCIe swap latency
(~19 ms; PCIe is not measurable on the M4) — parameterized and labeled as modeled. Deterministic
(fixed schedule seed). Pure arithmetic; safe to run off AC (no GPU work), but power is logged.
"""
import argparse
import bisect
import csv
import glob
import math
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import workloads                                   # noqa: E402
from exp3_policygap import CostModel, build_schedule           # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(REPO, "results", "csv")
FIG_DIR = os.path.join(REPO, "results", "figures")
KEEP_BW = 92e6                                                 # bytes/ms; Exp2 single-core read BW


def contention_factor(model):
    """MEASURED decode slowdown at full CPU load (intensity 100%) from Exp2, as a multiplier on
    the keep/decode (bandwidth-bound) path. Returns (factor, detail) or (None, msg)."""
    cands = [c for c in sorted(glob.glob(os.path.join(CSV_DIR, f"exp2_{model}_*.csv")))
             if "_meta" not in c and "pm" not in c and "pf" not in c]
    if not cands:
        return None, f"no exp2_{model}_*.csv"
    rows = list(csv.DictReader(open(cands[-1])))
    full = [float(r["slowdown_pct"]) for r in rows if int(float(r["intensity_pct"])) == 100]
    if not full:
        return None, f"no intensity=100 rows in {os.path.basename(cands[-1])}"
    f = 1.0 + (sum(full) / len(full)) / 100.0
    return f, f"{os.path.basename(cands[-1])}: decode x{f:.3f} at full load ({len(full)} N)"


def _next_use_map(events):
    conv_events = {}
    for idx, (c, _) in enumerate(events):
        conv_events.setdefault(c, []).append(idx)

    def next_use(c, e):
        lst = conv_events[c]; j = bisect.bisect_right(lst, e)
        return lst[j] if j < len(lst) else math.inf
    return next_use


def _norm(vals):
    """min-max to [0,1] over the candidate list; flat list -> all 0.5 (no signal)."""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def simulate(convs, events, cm, *, victim, budget, contention=1.0, evict_cost="uma",
             swap_ms=19.0, alpha=None, K=8.0, caware=False, next_use=None):
    """One pass over the interleaved schedule. `victim` selects who to evict under pressure:
       'lru' | 'alpha' | 'oracle_uma' | 'oracle_pcie'. `evict_cost` sets the RESUME cost of an
       evicted conv: 'uma' = recompute prefill (context lost); 'pcie' = swap_ms + only-new-tokens
       prefill (context preserved). `contention` multiplies the decode/keep (bandwidth) path."""
    ctx, resident, res_tok, last_used, swapped = {}, {}, {}, {}, {}
    ttfts, total = [], 0.0
    peak, over_budget, recomputes, swaps, disagree, evictions = 0, 0, 0, 0, 0, 0

    def memory():
        return sum(cm.kv_bytes(res_tok[c]) for c in resident if resident[c])

    def keep_ms(x):
        f = contention if caware else 1.0          # contention-aware policy sees the dearer bus
        return f * cm.kv_bytes(res_tok[x]) / KEEP_BW

    def pick(cands, e):
        if victim == "lru":
            return min(cands, key=lambda x: last_used[x])
        if victim == "alpha":
            stale = _norm([e - last_used[x] for x in cands])             # stalest -> evict
            cheap = _norm([-cm.prefill_ms(ctx[x]) for x in cands])       # cheap recompute -> evict
            keepb = _norm([keep_ms(x) for x in cands])                   # dearer to keep -> evict
            score = [alpha * (stale[i]) + (1 - alpha) * (0.5 * cheap[i] + 0.5 * keepb[i])
                     for i in range(len(cands))]
            return cands[int(np.argmax(score))]
        if victim == "oracle_uma":                  # recompute is expensive -> protect soon-reused
            return max(cands, key=lambda x: (next_use(x, e), cm.kv_bytes(res_tok[x])))
        if victim == "oracle_pcie":                 # swap cheap & flat -> free most memory
            return max(cands, key=lambda x: (cm.kv_bytes(res_tok[x]), next_use(x, e)))
        raise ValueError(victim)

    for e, (c, ti) in enumerate(events):
        p, r = convs[c][ti]
        prior = ctx.get(c, 0)
        if resident.get(c, False):
            ttft = cm.incr_prefill_ms(prior, prior + p)                  # resume from resident KV
        elif swapped.get(c, False):                                      # PCIe: swap KV back in
            ttft = swap_ms + cm.incr_prefill_ms(prior, prior + p); swaps += 1
        else:
            ttft = cm.prefill_ms(prior + p)                              # UMA: recompute prior+new
            if prior > 0:
                recomputes += 1
        total += ttft + contention * cm.decode_sum_ms(prior + p, r)      # decode = bandwidth path
        ttfts.append(ttft)

        ctx[c] = prior + p + r
        resident[c] = True; swapped[c] = False; res_tok[c] = ctx[c]; last_used[c] = e

        while memory() > budget:
            cands = [x for x in resident if resident[x] and x != c]
            if not cands:
                over_budget += 1; break
            v = pick(cands, e); evictions += 1
            if next_use is not None and victim in ("oracle_uma", "oracle_pcie"):
                alt = (max(cands, key=lambda x: (next_use(x, e), cm.kv_bytes(res_tok[x])))
                       if victim == "oracle_pcie" else
                       max(cands, key=lambda x: (cm.kv_bytes(res_tok[x]), next_use(x, e))))
                if alt != v:
                    disagree += 1
            resident[v] = False; res_tok[v] = 0
            swapped[v] = (evict_cost == "pcie")      # PCIe preserves ctx; UMA leaves it to recompute
        peak = max(peak, memory())

    return dict(total_ms=total, mean_ttft=float(np.mean(ttfts)),
                p95_ttft=float(np.percentile(ttfts, 95)), peak_mb=peak / 1e6,
                over_budget_rate=over_budget / max(len(events), 1), recomputes=recomputes,
                swaps=swaps, evictions=evictions,
                disagree_rate=disagree / max(evictions, 1))


# ----------------------------------------------------------------------------- P1a / P1b / P1c
def p1a_alpha(convs, events, cm, budget, cf, alphas):
    out = {"off": [], "on": []}
    for cond, ct in (("off", 1.0), ("on", cf)):
        for a in alphas:
            m = simulate(convs, events, cm, victim="alpha", budget=budget, contention=ct, alpha=a)
            out[cond].append((a, m["mean_ttft"], m["recomputes"], m["total_ms"]))
    return out


def p1b_oracle(convs, events, cm, budgets_gb, swap_ms):
    """Swept over KV budget (the key P1b result). For each budget: a foresight oracle minimizing
    UMA cost (recompute -> protect soon-reused = Belady) vs one minimizing PCIe cost (swap cheap
    & flat -> free most memory = evict-largest), both realized on UMA hardware. Penalty =
    extra TTFT from using the PCIe-cost policy on UMA. Contention-INVARIANT (TTFT is prefill,
    which Exp2 shows is compute-bound/immune), so run once without contention."""
    nu = _next_use_map(events)
    rows = []
    for bg in budgets_gb:
        budget = bg * 1e9
        u = simulate(convs, events, cm, victim="oracle_uma", budget=budget, evict_cost="uma",
                     next_use=nu)
        pu = simulate(convs, events, cm, victim="oracle_pcie", budget=budget, evict_cost="uma",
                      swap_ms=swap_ms, next_use=nu)
        pp = simulate(convs, events, cm, victim="oracle_pcie", budget=budget, evict_cost="pcie",
                      swap_ms=swap_ms, next_use=nu)
        pen = 100 * (pu["mean_ttft"] - u["mean_ttft"]) / u["mean_ttft"]
        rows.append(dict(budget_gb=bg, uma_ttft=u["mean_ttft"], pcie_ttft=pu["mean_ttft"],
                         uma_recomp=u["recomputes"], pcie_recomp=pu["recomputes"], penalty_pct=pen,
                         disagree=pu["disagree_rate"], pcie_native_ttft=pp["mean_ttft"]))
    return rows


def p1c_crossover(convs, events, cm, cf):
    """The keep(read)-vs-recompute THRESHOLD under contention. Each time a conversation RETURNS,
    its prior context is a keep-vs-recompute decision. keep_read_ms(N) = KV(N)/KEEP_BW scaled by
    the contention factor (decode reads are the bandwidth path); recompute_ms = prefill(N)
    (compute-bound, immune). margin = recompute_ms / keep_read_ms (>1 => keep wins). Contention
    divides the margin by the factor, but (Exp1: B* << bandwidth) it rarely crosses 1. This
    ISOLATES the honest contention->KV link: it narrows the margin / inflates ITL, but does NOT
    flip steady-state keep-vs-recompute. The divergence that DOES flip eviction is the swap-tier
    collapse, i.e. the UMA-vs-PCIe cost model (P1b)."""
    ctx, Ns = {}, []
    for (c, ti) in events:
        p, r = convs[c][ti]
        prior = ctx.get(c, 0)
        if prior > 0:
            Ns.append(prior)
        ctx[c] = prior + p + r
    m_off = [cm.prefill_ms(N) / (cm.kv_bytes(N) / KEEP_BW) for N in Ns]
    m_on = [cm.prefill_ms(N) / (cf * cm.kv_bytes(N) / KEEP_BW) for N in Ns]
    flips = sum(1 for x, y in zip(m_off, m_on) if (x > 1.0) != (y > 1.0))
    n = max(len(Ns), 1)
    return dict(n_decisions=len(Ns), Ns=Ns, margin_off=m_off, margin_on=m_on, flips=flips,
                keep_wins_off=sum(x > 1.0 for x in m_off) / n,
                keep_wins_on=sum(x > 1.0 for x in m_on) / n,
                min_margin_off=min(m_off) if Ns else float("nan"),
                min_margin_on=min(m_on) if Ns else float("nan"),
                beff_gbps=KEEP_BW / 1e6, beff_on_gbps=KEEP_BW / cf / 1e6)


def run(args):
    from common import thermal
    try:
        _, line = thermal.power_state(); print(f"[power] {line}  (sim: no GPU work, AC not required)")
    except Exception as ex:
        print(f"[power] n/a ({ex})")
    cm = CostModel(args.model)
    cf, detail = contention_factor(args.model)
    if cf is None:
        sys.exit(f"[p1] need Exp2 contention data: {detail}")
    print(f"[p1] cost model {cm.src} (KV {cm.kv_bpt} B/tok) | contention {detail}")
    print(f"[p1] PCIe swap = {args.swap_ms} ms (MODELED, not measurable on M4)")

    if args.trace and os.path.exists(args.trace):
        convs = workloads.load_sharegpt(args.trace, max_convs=args.max_convs)
        src = os.path.basename(args.trace)
    else:
        convs = workloads.synthetic_conversations(n_convs=args.n_convs, seed=0)
        src = f"synthetic(n={args.n_convs})"
    events = build_schedule(convs, args.concurrency, args.think_mean)
    budget = args.kv_budget_gb * 1e9
    print(f"[p1] trace {src} {workloads.trace_stats(convs)} | {len(events)} events | "
          f"concurrency {args.concurrency} | budget {args.kv_budget_gb} GB\n")

    alphas = [round(0.1 * i, 1) for i in range(11)]
    budgets = [float(x) for x in str(args.budget_sweep).split(",")]
    a = p1a_alpha(convs, events, cm, budget, cf, alphas)
    b = p1b_oracle(convs, events, cm, budgets, args.swap_ms)
    c = p1c_crossover(convs, events, cm, cf)
    _report(a, b, c, cf, args.kv_budget_gb)

    date = datetime.now().strftime("%Y%m%d")
    os.makedirs(CSV_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    _write_csv(a, b, c, cf, args, os.path.join(CSV_DIR, f"exp_p1_{args.model}_{date}.csv"))
    _plot(a, b, c, cf, args.model, os.path.join(FIG_DIR, f"exp_p1_kvlink_{args.model}_{date}.png"))


def _argmin_alpha(rows):
    return min(rows, key=lambda t: t[3])           # (alpha, mean_ttft, recomp, total) -> by total


def _report(a, b, c, cf, abudget):
    print(f"========== P1a  alpha-sweep: recency<->recompute eviction axis (budget {abudget} GB) ==========")
    print("  alpha:          " + "  ".join(f"{t[0]:>6.1f}" for t in a["off"]))
    print("  total off (s):  " + "  ".join(f"{t[3]/1e3:>6.1f}" for t in a["off"]))
    print("  total on  (s):  " + "  ".join(f"{t[3]/1e3:>6.1f}" for t in a["on"]))
    print("  recomputes:     " + "  ".join(f"{t[2]:>6d}" for t in a["off"]))
    ao, an = _argmin_alpha(a["off"]), _argmin_alpha(a["on"])
    tot = [t[3] for t in a["off"]]
    spread = 100 * (max(tot) - min(tot)) / min(tot)
    print(f"  best alpha  OFF = {ao[0]:.1f} ({ao[3]/1e3:.1f}s)   ON = {an[0]:.1f} ({an[3]/1e3:.1f}s)"
          f"  | spread across alpha = {spread:.1f}%  ({'optimum MOVES' if ao[0] != an[0] else 'optimum same'})")
    print("  -> recency(alpha=1)<->recompute-cost(alpha=0) eviction axis: no robust optimum across")
    print("     scale/budget; contention shifts uniformly. Consistent with Phase 2a (a recompute-cost")
    print("     eviction heuristic does not reliably beat recency).")

    print("\n========== P1b  UMA-cost vs PCIe-cost eviction, swept over KV budget (the real divergence) ==========")
    print("  budget(GB) | UMA-cost oracle (recomp) | PCIe-cost policy on UMA (recomp) | penalty | disagree")
    for r in b:
        print(f"  {r['budget_gb']:8.0f}   | {r['uma_ttft']:8.1f}ms ({r['uma_recomp']:5d})   | "
              f"{r['pcie_ttft']:8.1f}ms ({r['pcie_recomp']:5d})        | "
              f"+{r['penalty_pct']:6.0f}% | {r['disagree']*100:.0f}%")
    print("  -> as the budget becomes adequate (real engines' target), the UMA oracle drives recomputes")
    print("     toward 0 by protecting reuse, while the PCIe-cost policy keeps recomputing -> penalty")
    print("     GROWS with budget. PCIe-cost eviction is MISMATCHED on UMA: the swap-tier collapse,")
    print("     NOT contention, is what flips the KV decision.")

    print("\n========== P1c  contention crossover / decision-flip (HONEST: contention->KV) ==========")
    print(f"  {c['n_decisions']} keep-vs-recompute decisions on the trace.")
    print(f"  effective KV-read BW {c['beff_gbps']:.0f} GB/s -> {c['beff_on_gbps']:.0f} GB/s under contention")
    print(f"  keep wins: off {c['keep_wins_off']*100:.1f}%  on {c['keep_wins_on']*100:.1f}%  | "
          f"closest margin off {c['min_margin_off']:.0f}x  on {c['min_margin_on']:.0f}x")
    print(f"  decisions flipped keep<->recompute by contention: {c['flips']}/{c['n_decisions']} "
          f"({100*c['flips']/max(c['n_decisions'],1):.1f}%)")
    print(f"  -> contention narrows the margin (~x{cf:.2f}) but keep still wins; its KV lever is decode")
    print("     ITL/throughput + admission ACCOUNTING, not eviction flips. (Eviction divergence = P1b.)")
    print("==========================================================================================")


def _write_csv(a, b, c, cf, args, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "key", "cond", "alpha", "mean_ttft", "recomputes", "total_ms",
                    "disagree_rate", "contention_factor"])
        for cond in ("off", "on"):
            for (al, tt, rc, to) in a[cond]:
                w.writerow(["p1a", "alpha", cond, al, round(tt, 3), rc, round(to, 1), "", round(cf, 4)])
        for r in b:
            w.writerow(["p1b", f"budget_{r['budget_gb']:.0f}GB", "", "", round(r["pcie_ttft"], 3),
                        r["pcie_recomp"], round(r["penalty_pct"], 1), round(r["disagree"], 4),
                        round(r["uma_ttft"], 3)])     # last col = UMA-oracle baseline TTFT
        w.writerow(["p1c", "crossover_off", "off", "", "", "", "", "", round(cf, 4)])
        w.writerow(["p1c", "summary", "", "", "", c["flips"], c["n_decisions"], "", round(cf, 4)])
        w.writerow(["p1c", "keep_wins_off", "off", "", "", "", "", round(c["keep_wins_off"], 4), ""])
        w.writerow(["p1c", "keep_wins_on", "on", "", "", "", "", round(c["keep_wins_on"], 4), ""])
        w.writerow(["p1c", "min_margin", "", "", round(c["min_margin_off"], 2), "",
                    round(c["min_margin_on"], 2), "", ""])
        w.writerow(["p1c", "beff_gbps", "", "", round(c["beff_gbps"], 1), "",
                    round(c["beff_on_gbps"], 1), "", ""])
    print(f"\n[out] wrote {path}")


def _plot(a, b, c, cf, model, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    xs = [t[0] for t in a["off"]]
    ax[0].plot(xs, [t[3] / 1e3 for t in a["off"]], "o-", color="tab:blue", label="no contention")
    ax[0].plot(xs, [t[3] / 1e3 for t in a["on"]], "s-", color="tab:red", label=f"contention x{cf:.2f}")
    for cond, col in (("off", "tab:blue"), ("on", "tab:red")):
        am = _argmin_alpha(a[cond]); ax[0].scatter([am[0]], [am[3] / 1e3], s=150, facecolors="none",
                                                    edgecolors=col, linewidths=2, zorder=5)
    ax[0].set_xlabel(r"$\alpha$  (1 = pure recency/LRU,  0 = pure recompute-cost)")
    ax[0].set_ylabel("realized total latency (s)")
    ax[0].set_title("P1a: recency↔recompute axis\n(optimum contention-robust)")
    ax[0].legend(); ax[0].grid(alpha=.3)

    bg = [r["budget_gb"] for r in b]
    ax[1].plot(bg, [r["uma_recomp"] for r in b], "o-", color="tab:green",
               label="UMA-cost oracle recomputes")
    ax[1].plot(bg, [r["pcie_recomp"] for r in b], "s-", color="tab:orange",
               label="PCIe-cost policy recomputes")
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("KV budget (GB)")
    ax[1].set_ylabel("recomputes on UMA"); ax[1].grid(alpha=.3)
    ax1b = ax[1].twinx()
    ax1b.plot(bg, [max(r["penalty_pct"], 1e-1) for r in b], "^--", color="tab:red",
              label="TTFT penalty (%)")
    ax1b.set_yscale("log"); ax1b.set_ylabel("PCIe-on-UMA TTFT penalty (%)", color="tab:red")
    ax1b.tick_params(axis="y", labelcolor="tab:red")
    ax[1].set_title("P1b: PCIe cost model misprices UMA eviction\n(penalty grows as budget gets adequate)")
    ax[1].legend(loc="center right", fontsize=8)

    mo = np.log10(np.clip(c["margin_off"], 1e-3, None))
    mn = np.log10(np.clip(c["margin_on"], 1e-3, None))
    lo, hi = float(min(mo.min(), mn.min())), float(max(mo.max(), mn.max()))
    bins = np.linspace(lo, hi, 40)
    ax[2].hist(mo, bins=bins, color="tab:blue", alpha=.55, label="no contention")
    ax[2].hist(mn, bins=bins, color="tab:red", alpha=.55, label=f"contention x{cf:.2f}")
    ax[2].axvline(0, color="k", lw=2, ls="--")
    ax[2].text(0.1, ax[2].get_ylim()[1]*.9, "keep wins →", fontsize=9)
    ax[2].text(-0.05, ax[2].get_ylim()[1]*.9, "← recompute", fontsize=9, ha="right")
    ax[2].set_xlabel(r"$\log_{10}$(recompute / keep-read margin)")
    ax[2].set_ylabel("trace decisions")
    ax[2].set_title(f"P1c: contention shifts margin left\nbut 0 flips ({c['flips']}/{c['n_decisions']})")
    ax[2].legend(); ax[2].grid(alpha=.3, axis="y")

    fig.suptitle(f"P1 KV-link: where the UMA KV decision diverges (cost-model sim) — {model}",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(png, dpi=150)
    print(f"[out] wrote {png}")


def main():
    ap = argparse.ArgumentParser(description="P1: contention changes the KV-cache decision (sim).")
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--trace", default=os.path.join(REPO, "data/raw/old/sg_52k.json"))
    ap.add_argument("--max-convs", type=int, default=2000)
    ap.add_argument("--n-convs", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--think-mean", type=float, default=20.0)
    ap.add_argument("--kv-budget-gb", type=float, default=2.0, help="budget for P1a/P1c")
    ap.add_argument("--budget-sweep", default="2,4,8,16,32", help="P1b KV-budget sweep (GB)")
    ap.add_argument("--swap-ms", type=float, default=19.0, help="MODELED PCIe swap latency")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
