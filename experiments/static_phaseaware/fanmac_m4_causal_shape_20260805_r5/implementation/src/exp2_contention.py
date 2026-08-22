#!/usr/bin/env python3
"""
exp2_contention.py - Exp 2: CPU/GPU shared-bus bandwidth contention during decode.
[RQ2/H2] - the core novelty.  See EXPERIMENTS.md Exp 2.

Question: how much does concurrent CPU memory traffic slow GPU decode on Apple Silicon's
shared unified-memory bus, and how far does it shift the keep-vs-recompute crossover?

Method: for each cached context length N, measure GPU decode throughput (tok/s) in a
fixed time window while a controlled CPU memory-bandwidth load (native STREAM triad,
common/cpuload.py, P-core biased) runs concurrently at increasing intensity
(threads 0 = solo .. 4 = all P-cores). The CPU load reports the aggregate GB/s it
actually achieved -> decode tok/s is plotted against real CPU bandwidth consumed.

Controls (RESEARCH.md §6): mx.eval-forced decode (trap #1, via common.measure); thermal
cooldown between points + throttle_ratio + optional powermetrics (trap #2); mem limit
(trap #3); AC power (trap #4); CPU load QoS-biased to P-cores.

Confound to keep honest: the net decode slowdown bundles (i) shared DRAM-bandwidth
contention, (ii) SoC power/thermal budget sharing, and (iii) P-core scheduling pressure
on the GPU-dispatch thread. v1 measures the NET effect with these controls; isolating
(i) from (ii)/(iii) is a documented follow-up.
"""
import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import measure, thermal, models, cpuload  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(REPO, "results", "csv")
FIG_DIR = os.path.join(REPO, "results", "figures")


def build_cache(model, N):
    """Build + FULLY eval an N-token KV cache (trap #1). Done BEFORE the CPU load starts
    so the load covers the entire decode window (the prior version built the cache inside
    the timed section, letting the load expire mid-window for large N -> under-measured)."""
    cache = measure.make_prompt_cache(model)
    x = measure.make_tokens(N)
    mx.eval(x)
    built = measure.forward_body(model, x, cache)
    mx.eval(built)
    for c in cache:                              # force K/V arrays themselves
        s = c.state
        if s and s[0] is not None:
            mx.eval(s[0], s[1])
    return cache


def decode_throughput(model, cache, window_s=3.0, warmup=3):
    """Sustained autoregressive decode tok/s over a fixed window on a PREBUILT cache.
    Every step is mx.eval-forced (real sequential decode)."""
    kvb, _ = measure.kv_cache_bytes(cache)
    one = mx.array([[7]], dtype=mx.int32)
    mx.eval(one)
    y = None
    for _ in range(warmup):                      # decode-kernel warmup (excluded)
        y = model(one, cache=cache)
        mx.eval(y)

    times = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < window_s:
        s0 = time.perf_counter()
        y = model(one, cache=cache)
        mx.eval(y)                               # per-step barrier (true sequential decode)
        times.append((time.perf_counter() - s0) * 1e3)
    total = time.perf_counter() - t0

    steps = len(times)
    return dict(tok_s=steps / total, steps=steps,
                ms_per_step=float(np.median(times)),
                throttle=thermal.throttle_ratio(times),
                kv_mb=kvb / 1024**2)


def run(args):
    thermal.assert_power(args.allow_battery)
    procs = thermal.top_mem_procs()
    if procs:
        print("[mem] top: " + ", ".join(f"{c.split('/')[-1]}={mb:.0f}MB" for c, mb in procs))
    measure.set_mem_limit_gb(args.mem_limit_gb)
    print(f"[mem] set_memory_limit = {args.mem_limit_gb} GB")

    print(f"[load] {models.resolve_model_id(args.model)} ...")
    t0 = time.perf_counter()
    model, _ = models.load_model(args.model)
    mx.eval(model.parameters())
    print(f"[load] done in {time.perf_counter()-t0:.1f}s")
    cfg = models.model_config(model)
    print(f"[config] {cfg}")
    cpuload.ensure_built()
    print("[cpuload] stream_load.c compiled")
    print("[warmup] global kernel + DVFS clock warmup ...")
    measure.global_warmup(model)

    Ns = [int(x) for x in str(args.ns).split(",") if x.strip()]
    intensities = [int(x) for x in str(args.intensity).split(",") if x.strip()]
    load_seconds = round(0.5 + args.window_s + 0.8, 2)   # ramp + window + margin
    rows = []
    for N in Ns:
        base = None
        for inten in intensities:
            measure.reset_peak()
            cache = build_cache(model, N)                # build BEFORE the load (bugfix:
            load = cpuload.CpuBandwidthLoad(inten, mb=args.mb, seconds=load_seconds).start()
            if inten > 0:                                # so the load covers the whole window)
                time.sleep(0.5)                          # let the CPU load ramp first
            r = decode_throughput(model, cache, window_s=args.window_s)
            gbps = load.wait()
            peak = measure.peak_mb()
            del cache
            measure.free_buffers()
            if inten == 0:
                base = r["tok_s"]
            slow = (1 - r["tok_s"] / base) * 100 if base else 0.0
            rows.append(dict(
                N=N, intensity_pct=inten, cpu_gbps=round(gbps, 2),
                decode_tok_s=round(r["tok_s"], 2), ms_per_step=round(r["ms_per_step"], 3),
                slowdown_pct=round(slow, 1), decode_throttle=round(r["throttle"], 3),
                steps=r["steps"], peak_mb=round(peak, 0), kv_mb=round(r["kv_mb"], 1)))
            thr = "" if r["throttle"] < 1.10 else f"  <throttle {r['throttle']:.2f}>"
            print(f"  N={N:5d} load={inten:3d}% | cpu={gbps:6.1f} GB/s | "
                  f"decode={r['tok_s']:6.1f} tok/s ({r['ms_per_step']:.2f} ms/step) | "
                  f"slowdown={slow:5.1f}% | peak={peak:.0f}MB{thr}")
            time.sleep(args.cooldown)

    tag = args.model if args.model in models.REGISTRY else models.resolve_model_id(args.model).split("/")[-1]
    date = datetime.now().strftime("%Y%m%d")
    suffix = "_smoke" if args.smoke else ""
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    csv_path = os.path.join(CSV_DIR, f"exp2_{tag}_{date}{suffix}.csv")
    cols = ["N", "intensity_pct", "cpu_gbps", "decode_tok_s", "ms_per_step", "slowdown_pct",
            "decode_throttle", "steps", "peak_mb", "kv_mb"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[out] wrote {csv_path}")

    with open(os.path.join(CSV_DIR, f"exp2_{tag}_{date}{suffix}_meta.json"), "w") as f:
        json.dump(dict(timestamp=datetime.now().isoformat(timespec="seconds"),
                       model=models.resolve_model_id(args.model), config=cfg,
                       mlx="0.31.2", mlx_lm="0.31.3", python=platform.python_version(),
                       params=dict(ns=Ns, intensity=intensities, window_s=args.window_s,
                                   mb=args.mb, cooldown=args.cooldown,
                                   load_seconds=load_seconds)), f, indent=2)

    png = os.path.join(FIG_DIR, f"exp2_contention_{tag}_{date}{suffix}.png")
    plot(rows, png, tag)
    analyze(rows)
    return rows


def plot(rows, png, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ns = sorted({r["N"] for r in rows})
    colors = {Ns[i]: c for i, c in enumerate(
        ["tab:green", "tab:blue", "tab:orange", "tab:red", "tab:purple"][:len(Ns)])}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    for N in Ns:
        rs = sorted([r for r in rows if r["N"] == N], key=lambda r: r["intensity_pct"])
        x = [r["cpu_gbps"] for r in rs]
        a.plot(x, [r["decode_tok_s"] for r in rs], "o-", color=colors[N], label=f"N={N}")
        b.plot(x, [r["slowdown_pct"] for r in rs], "o-", color=colors[N], label=f"N={N}")
        for r in rs:                              # annotate load intensity
            a.annotate(f"{r['intensity_pct']}%", (r["cpu_gbps"], r["decode_tok_s"]),
                       textcoords="offset points", xytext=(4, 4), fontsize=7)
    a.set_xlabel("concurrent CPU load (GB/s, STREAM triad)")
    a.set_ylabel("GPU decode throughput (tok/s)")
    a.set_title("decode throughput vs CPU bandwidth load")
    a.grid(True, alpha=0.25); a.legend(fontsize=9)
    b.set_xlabel("concurrent CPU load (GB/s)")
    b.set_ylabel("decode slowdown vs solo (%)")
    b.set_title("contention slowdown")
    b.grid(True, alpha=0.25); b.legend(fontsize=9)
    fig.suptitle(f"Exp2 shared-bus contention - {tag} (M4 MBA, unified mem)", fontsize=12)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"[out] wrote {png}")


def analyze(rows):
    print("\n================ CONTENTION FINDINGS ================")
    for N in sorted({r["N"] for r in rows}):
        rs = sorted([r for r in rows if r["N"] == N], key=lambda r: r["intensity_pct"])
        base, top = rs[0], rs[-1]
        print(f"N={N:5d}: solo {base['decode_tok_s']:.1f} tok/s  ->  "
              f"{top['decode_tok_s']:.1f} tok/s @ {top['cpu_gbps']:.0f} GB/s CPU load "
              f"({top['intensity_pct']}%)  =  {top['slowdown_pct']:.1f}% slower")
    print("=> decode slowing as CPU bandwidth load rises = shared-bus contention (H2).")
    print("   A bigger slowdown means decode (keep-and-read) gets more expensive under")
    print("   contention -> shifts the keep-vs-recompute crossover N* (recompute relatively")
    print("   cheaper when the bus is busy). Quantify N* shift with the contended decode_ms.")
    print("====================================================")


def main():
    ap = argparse.ArgumentParser(description="Exp2 CPU/GPU shared-bus bandwidth contention.")
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--ns", default="2048,8192")
    ap.add_argument("--intensity", default="0,25,50,75,100",
                    help="CPU bandwidth load duty cycle %% (0=solo baseline)")
    ap.add_argument("--window-s", type=float, default=3.0, help="decode measurement window")
    ap.add_argument("--mb", type=int, default=96, help="STREAM array MB per thread (>> LLC)")
    ap.add_argument("--cooldown", type=float, default=6.0)
    ap.add_argument("--mem-limit-gb", type=float, default=12.0)
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="fast: ns=2048 intensity=0,100 window=1 cd=2")
    args = ap.parse_args()
    if args.smoke:
        args.ns, args.intensity, args.window_s, args.cooldown = "2048", "0,100", 1.0, 2.0
    run(args)


if __name__ == "__main__":
    main()
