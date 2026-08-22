#!/usr/bin/env python3
"""
exp2_prefill_contention.py - closes Exp 2's open question: does shared-bus contention
SHIFT the keep-vs-recompute crossover N*, or just scale both sides together?

Exp2 showed CPU bandwidth load slows DECODE (keep-and-read) ~36-38%. But recompute
(prefill) uses the same bus, so it may slow too. N* only moves if the two have DIFFERENT
contention sensitivity. Hypothesis: prefill is more COMPUTE-bound (weights reused across
N tokens) -> LESS bandwidth-sensitive than decode -> under contention keep gets relatively
costlier -> N* shifts toward recompute.

For each N: measure prefill_ms (drop-and-recompute, body-only) and decode tok/s
(keep-and-read), each solo vs under a 100% CPU bandwidth load, + powermetrics on the
loaded prefill to confirm it runs full-clock (so any smaller slowdown is bandwidth-slack,
not throttling). Compares prefill_slowdown% vs decode_slowdown%.
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import measure, models, cpuload, thermal      # noqa: E402
from exp2_contention import build_cache, decode_throughput  # noqa: E402
from exp2_pm_probe import parse_pm                           # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(REPO, "results", "csv")
LOG_DIR = os.path.join(REPO, "results", "logs")


def prefill_med(model, N, reps, warmup=1):
    return float(np.median(measure.time_prefill(model, N, reps=reps, warmup=warmup)))


def decode_toks(model, N, window_s, loaded):
    """Decode tok/s on a freshly built N-cache, optionally under a 100% CPU load."""
    cache = build_cache(model, N)
    load = cpuload.CpuBandwidthLoad(100 if loaded else 0,
                                    seconds=0.5 + window_s + 0.8).start()
    if loaded:
        time.sleep(0.5)
    r = decode_throughput(model, cache, window_s=window_s)
    gbps = load.wait()
    del cache
    measure.free_buffers()
    return r["tok_s"], gbps


def run(args):
    thermal.assert_power(args.allow_battery)
    measure.set_mem_limit_gb(args.mem_limit_gb)
    print(f"[load] {models.resolve_model_id(args.model)} ...")
    model, _ = models.load_model(args.model)
    mx.eval(model.parameters())
    cpuload.ensure_built()
    print("[warmup] global ...")
    measure.global_warmup(model)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    Ns = [int(x) for x in str(args.ns).split(",") if x.strip()]
    rows = []
    for N in Ns:
        # ---- prefill: solo ----
        pf_solo = prefill_med(model, N, args.reps)
        time.sleep(args.cooldown)
        # ---- prefill: under 100% CPU load (+ powermetrics) ----
        load_s = round(0.5 + (1 + args.reps) * (pf_solo / 1000.0) * 1.4 + 1.0, 1)
        pmfile = os.path.join(LOG_DIR, f"pm_prefill_{args.model}_{N}.txt")
        load = cpuload.CpuBandwidthLoad(100, seconds=load_s).start()
        time.sleep(0.6)
        pm = thermal.PowerMetricsLogger(pmfile, interval_ms=250, samplers="gpu_power")
        pm_ok = pm.start()
        pf_load = prefill_med(model, N, args.reps)
        pm.stop()
        gbps_pf = load.wait()
        st = parse_pm(pmfile) if pm_ok else {"gpu_mhz": float("nan")}
        time.sleep(args.cooldown)

        # ---- decode: solo vs loaded ----
        dec_solo, _ = decode_toks(model, N, args.window_s, loaded=False)
        time.sleep(args.cooldown)
        dec_load, gbps_dec = decode_toks(model, N, args.window_s, loaded=True)
        time.sleep(args.cooldown)

        pf_slow = (pf_load / pf_solo - 1) * 100        # prefill ms goes UP
        dec_slow = (1 - dec_load / dec_solo) * 100     # decode tok/s goes DOWN
        rows.append(dict(
            N=N, prefill_solo_ms=round(pf_solo, 1), prefill_load_ms=round(pf_load, 1),
            prefill_slow_pct=round(pf_slow, 1), prefill_gpu_mhz=round(st["gpu_mhz"], 0),
            decode_solo_toks=round(dec_solo, 1), decode_load_toks=round(dec_load, 1),
            decode_slow_pct=round(dec_slow, 1),
            cpu_gbps_pf=round(gbps_pf, 1), cpu_gbps_dec=round(gbps_dec, 1)))
        print(f"  N={N:5d} | prefill {pf_solo:7.0f}->{pf_load:7.0f}ms ({pf_slow:+5.1f}%, "
              f"GPU {st['gpu_mhz']:.0f}MHz @ {gbps_pf:.0f}GB/s) | "
              f"decode {dec_solo:5.1f}->{dec_load:5.1f}tok/s ({dec_slow:5.1f}% slower @ {gbps_dec:.0f}GB/s)")
        verdict(N, pf_slow, dec_slow)

    date = datetime.now().strftime("%Y%m%d")
    path = os.path.join(CSV_DIR, f"exp2pf_{args.model}_{date}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[out] wrote {path}")


def verdict(N, pf_slow, dec_slow):
    print(f"    --- N={N}: prefill {pf_slow:+.1f}% vs decode {dec_slow:.1f}% contention slowdown ---")
    if dec_slow > pf_slow + 5:
        print("    => DECODE more bandwidth-sensitive than prefill: under contention keep-and-read")
        print("       gets relatively costlier -> N* SHIFTS TOWARD RECOMPUTE (recompute relatively")
        print(f"       cheaper when the bus is busy). Magnitude factor ~{(1+dec_slow/100)/(1+pf_slow/100):.2f}x.")
    elif pf_slow > dec_slow + 5:
        print("    => PREFILL more sensitive -> N* shifts toward keep (recompute costlier under load).")
    else:
        print("    => prefill and decode ~equally contention-sensitive -> N* ~UNCHANGED under contention")
        print("       (contention scales both sides together; it is not an N*-moving lever).")


def main():
    ap = argparse.ArgumentParser(description="Exp2 follow-up: prefill vs decode contention sensitivity.")
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--ns", default="2048,4096")
    ap.add_argument("--reps", type=int, default=3, help="prefill reps (median)")
    ap.add_argument("--window-s", type=float, default=3.0, help="decode window")
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--mem-limit-gb", type=float, default=12.0)
    ap.add_argument("--allow-battery", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
