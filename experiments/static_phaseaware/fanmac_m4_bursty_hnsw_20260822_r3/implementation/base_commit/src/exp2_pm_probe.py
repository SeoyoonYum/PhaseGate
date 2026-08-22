#!/usr/bin/env python3
"""
exp2_pm_probe.py - Exp 2 mechanism probe: is the decode-under-CPU-load slowdown
POWER/THERMAL throttling or BANDWIDTH/arbitration contention?

For each context length N, run sustained GPU decode for a fixed window twice -- (a) solo
and (b) with a 100% CPU memory-bandwidth load -- while sampling GPU HW active frequency,
active residency and power via powermetrics (needs the scoped passwordless sudo entry for
/usr/bin/powermetrics; see thermal.PowerMetricsLogger).

Discriminator (the whole point):
  - GPU clock DROPS under load  -> power/thermal throttling is a factor.
  - GPU clock ~unchanged but tok/s falls -> the GPU runs full-clock but STALLS on memory
    = shared-bus bandwidth / arbitration contention (not power throttling).
"""
import argparse
import csv
import os
import re
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import measure, models, cpuload, thermal  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(REPO, "results", "csv")
LOG_DIR = os.path.join(REPO, "results", "logs")


def build_cache(model, N):
    cache = measure.make_prompt_cache(model)
    x = measure.make_tokens(N)
    mx.eval(x)
    built = measure.forward_body(model, x, cache)
    mx.eval(built)
    for c in cache:
        s = c.state
        if s and s[0] is not None:
            mx.eval(s[0], s[1])
    return cache


def decode_for(model, cache, window_s, warmup=3):
    one = mx.array([[7]], dtype=mx.int32)
    mx.eval(one)
    for _ in range(warmup):
        y = model(one, cache=cache)
        mx.eval(y)
    times = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < window_s:
        s0 = time.perf_counter()
        y = model(one, cache=cache)
        mx.eval(y)
        times.append((time.perf_counter() - s0) * 1e3)
    dt = time.perf_counter() - t0
    return len(times) / dt, float(np.median(times)), thermal.throttle_ratio(times)


def parse_pm(path):
    """Mean GPU active frequency (MHz), active residency (%), power (mW) over a log."""
    try:
        txt = open(path, errors="ignore").read()
    except Exception:
        return dict(gpu_mhz=float("nan"), gpu_active_pct=float("nan"),
                    gpu_mw=float("nan"), n=0)
    fr = [int(m) for m in re.findall(r"GPU HW active frequency:\s*(\d+)\s*MHz", txt)]
    rs = [float(m) for m in re.findall(r"GPU HW active residency:\s*([\d.]+)%", txt)]
    pw = [int(m) for m in re.findall(r"GPU Power:\s*(\d+)\s*mW", txt)]
    mean = lambda a: (float(np.mean(a)) if a else float("nan"))
    return dict(gpu_mhz=mean(fr), gpu_active_pct=mean(rs), gpu_mw=mean(pw), n=len(fr))


def run(args):
    thermal.assert_power(args.allow_battery)
    measure.set_mem_limit_gb(args.mem_limit_gb)
    print(f"[load] {models.resolve_model_id(args.model)} ...")
    model, _ = models.load_model(args.model)
    mx.eval(model.parameters())
    cpuload.ensure_built()
    print("[warmup] global ...")
    measure.global_warmup(model)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    Ns = [int(x) for x in str(args.ns).split(",") if x.strip()]
    rows = []
    for N in Ns:
        per_n = {}
        for label, inten in (("solo", 0), ("loaded", 100)):
            cache = build_cache(model, N)
            pmfile = os.path.join(LOG_DIR, f"pm_{args.model}_{N}_{label}.txt")
            load = cpuload.CpuBandwidthLoad(inten, seconds=args.window_s + 2.0).start()
            if inten > 0:
                time.sleep(0.7)                       # let the CPU load ramp first
            pm = thermal.PowerMetricsLogger(pmfile, interval_ms=250, samplers="gpu_power")
            pm_ok = pm.start()
            tok, med, thr = decode_for(model, cache, args.window_s)
            pm.stop()
            gbps = load.wait()
            st = parse_pm(pmfile) if pm_ok else dict(gpu_mhz=float("nan"),
                                                     gpu_active_pct=float("nan"),
                                                     gpu_mw=float("nan"), n=0)
            row = dict(N=N, phase=label, cpu_gbps=round(gbps, 1), decode_tok_s=round(tok, 1),
                       med_ms=round(med, 2), throttle=round(thr, 2),
                       gpu_mhz=round(st["gpu_mhz"], 0), gpu_active_pct=round(st["gpu_active_pct"], 1),
                       gpu_mw=round(st["gpu_mw"], 0), pm_samples=st["n"])
            rows.append(row)
            per_n[label] = row
            print(f"  N={N:5d} [{label:6}] cpu={gbps:5.1f} GB/s | decode={tok:5.1f} tok/s "
                  f"(med {med:.2f}ms, thr {thr:.2f}) | GPU {st['gpu_mhz']:.0f}MHz "
                  f"act {st['gpu_active_pct']:.0f}% {st['gpu_mw']:.0f}mW (n={st['n']})")
            del cache
            measure.free_buffers()
            time.sleep(args.cooldown)
        verdict(N, per_n["solo"], per_n["loaded"])

    date = datetime.now().strftime("%Y%m%d")
    path = os.path.join(CSV_DIR, f"exp2pm_{args.model}_{date}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[out] wrote {path}")


def verdict(N, solo, load):
    df = ((load["gpu_mhz"] - solo["gpu_mhz"]) / solo["gpu_mhz"] * 100) if solo["gpu_mhz"] else 0
    dt = ((load["decode_tok_s"] - solo["decode_tok_s"]) / solo["decode_tok_s"] * 100)
    print(f"  --- N={N} verdict ---")
    print(f"    decode : {solo['decode_tok_s']:.1f} -> {load['decode_tok_s']:.1f} tok/s ({dt:+.1f}%)")
    print(f"    GPU MHz: {solo['gpu_mhz']:.0f} -> {load['gpu_mhz']:.0f} ({df:+.1f}%)")
    print(f"    GPU act: {solo['gpu_active_pct']:.0f}% -> {load['gpu_active_pct']:.0f}%   "
          f"power {solo['gpu_mw']:.0f} -> {load['gpu_mw']:.0f} mW")
    if df < -5:
        print(f"    => GPU CLOCK DROPS {df:.0f}% under load -> POWER/THERMAL throttling is a factor.")
    elif dt < -3:
        print("    => GPU clock ~flat but throughput down -> BANDWIDTH/arbitration contention")
        print("       (GPU runs full-clock, stalls on memory; not power throttling).")
    else:
        print("    => little change either way (decode robust to CPU load at this N).")


def main():
    ap = argparse.ArgumentParser(description="Exp2 mechanism probe (powermetrics A/B).")
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--ns", default="2048,8192")
    ap.add_argument("--window-s", type=float, default=6.0, help="decode window per phase")
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--mem-limit-gb", type=float, default=12.0)
    ap.add_argument("--allow-battery", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
