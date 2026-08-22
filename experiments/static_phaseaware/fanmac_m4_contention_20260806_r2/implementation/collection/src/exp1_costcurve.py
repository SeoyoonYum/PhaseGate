#!/usr/bin/env python3
"""
exp1_costcurve.py - Exp 1: KV-cache primitive cost curves (see EXPERIMENTS.md Exp 1).

  prefill_ms(N)     : recompute N-token KV from scratch   (drop-and-recompute)
  decode_step_ms(N) : one decode step with N tokens cached (keep-and-read)

All measurement controls (RESEARCH.md §6) live in src/common:
  - mx.eval sync + decode-isolation + body-only prefill + DVFS warmup -> common.measure
  - power guard + throttle detection (+ optional powermetrics)        -> common.thermal
  - 12GB memory cap                                                   -> common.measure.set_mem_limit_gb

Usage:
  python src/exp1_costcurve.py --model 1.5B
  python src/exp1_costcurve.py --model 7B --ns 128,256,512,1024,2048,4096 --cooldown 12
  python src/exp1_costcurve.py --model 1.5B --smoke
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make `common` importable
from common import measure, thermal, models

RESULTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def run(args):
    on_ac = thermal.assert_power(args.allow_battery)
    procs = thermal.top_mem_procs()
    if procs:
        print("[mem] top: " + ", ".join(f"{c}={mb:.0f}MB" for c, mb in procs))
    measure.set_mem_limit_gb(args.mem_limit_gb)
    print(f"[mem] set_memory_limit = {args.mem_limit_gb} GB")

    import mlx.core as mx
    model_id = models.resolve_model_id(args.model)
    print(f"[load] {model_id} ...")
    t0 = time.perf_counter()
    model, _tok = models.load_model(args.model)
    mx.eval(model.parameters())
    print(f"[load] done in {time.perf_counter()-t0:.1f}s")
    cfg = models.model_config(model)
    print(f"[config] {cfg}")
    print(f"[prefill] body-only KV build: {getattr(model, 'model', None) is not None}")

    print("[warmup] global kernel + DVFS clock warmup ...")
    measure.global_warmup(model)

    Ns = [int(x) for x in str(args.ns).split(",") if x.strip()]
    rows = []
    kv_per_token = models.kv_bytes_per_token(cfg)
    for i, N in enumerate(Ns):
        measure.reset_peak()
        pref = measure.time_prefill(model, N, args.reps, args.warmup)
        dec, kvb, _kvmeta, ctx0 = measure.time_decode(model, N, args.reps, args.warmup)
        ps, ds = measure.summarize(pref), measure.summarize(dec)
        tr = thermal.throttle_ratio(dec)
        sl = thermal.cpu_speed_limit()
        row = dict(N=N,
                   prefill_ms_med=ps["med"], decode_ms_med=ds["med"],
                   prefill_min=ps["min"], decode_min=ds["min"], decode_max=ds["max"],
                   peak_mb=measure.peak_mb(), prefill_max=ps["max"],
                   decode_throttle_ratio=tr,
                   cpu_speed_limit=(sl if sl is not None else ""),
                   decode_ctx_tokens=ctx0, kv_mb=kvb / 1024**2)
        rows.append(row)
        warn = "  <THROTTLE?>" if tr and tr > 1.10 else ""
        print(f"  N={N:5d} | prefill med={row['prefill_ms_med']:8.2f} "
              f"(min {row['prefill_min']:8.2f}) | decode med={row['decode_ms_med']:6.3f} "
              f"(min {row['decode_min']:6.3f}, max {row['decode_max']:6.3f}, x{tr:.2f}) | "
              f"kv={row['kv_mb']:6.1f}MB peak={row['peak_mb']:6.0f}MB{warn}")
        if i < len(Ns) - 1:
            time.sleep(args.cooldown)  # cooldown between N (fanless thermal control)

    write_outputs(args, model_id, cfg, rows, kv_per_token, on_ac)


def write_outputs(args, model_id, cfg, rows, kv_per_token, on_ac):
    tag = args.model if args.model in models.REGISTRY else model_id.split("/")[-1]
    date = datetime.now().strftime("%Y%m%d")
    stem = f"exp1_{tag}_{date}" + ("_smoke" if args.smoke else "")
    csv_dir = os.path.join(RESULTS, "csv")
    fig_dir = os.path.join(RESULTS, "figures")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, stem + ".csv")
    cols = ["N", "prefill_ms_med", "decode_ms_med", "prefill_min", "decode_min",
            "decode_max", "peak_mb", "prefill_max", "decode_throttle_ratio",
            "cpu_speed_limit", "decode_ctx_tokens", "kv_mb"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[out] {csv_path}")

    meta = dict(timestamp=datetime.now().isoformat(timespec="seconds"),
                experiment="exp1", model=model_id, config=cfg,
                mlx="0.31.2", mlx_lm="0.31.3", python=platform.python_version(),
                machine=platform.platform(),
                params=dict(ns=args.ns, reps=args.reps, warmup=args.warmup,
                            cooldown=args.cooldown, mem_limit_gb=args.mem_limit_gb,
                            bw_gbps=args.bw_gbps, on_ac=on_ac, smoke=args.smoke),
                kv_bytes_per_token=kv_per_token)
    with open(os.path.join(csv_dir, stem + "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    png = os.path.join(fig_dir, stem + ".png")
    plot(rows, png, args.bw_gbps, kv_per_token, model_id)
    analyze(rows, args.bw_gbps, kv_per_token)


def plot(rows, png, bw_gbps, kv_per_token, model_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = [r["N"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    ax1.set_xscale("log", base=2)
    l1, = ax1.plot(N, [r["prefill_ms_med"] for r in rows], "o-", color="tab:blue",
                   label="prefill_ms (drop-and-recompute)")
    ax1.set_xlabel("context length N (tokens)")
    ax1.set_ylabel("prefill_ms", color="tab:blue")
    ax1.set_xticks(N); ax1.set_xticklabels([str(n) for n in N])
    ax1.grid(True, which="both", alpha=0.25)
    ax2 = ax1.twinx()
    l2, = ax2.plot(N, [r["decode_ms_med"] for r in rows], "s-", color="tab:red",
                   label="decode_step_ms (keep-and-read)")
    ax2.set_ylabel("decode_step_ms", color="tab:red")
    lines = [l1, l2]
    if kv_per_token:
        pred = [kv_per_token * n / (bw_gbps * 1e9) * 1e3 for n in N]
        l3, = ax2.plot(N, pred, "--", color="gray", label=f"KV-read model @{bw_gbps:.0f}GB/s")
        lines.append(l3)
    ax1.set_title(f"Exp1 cost curves - {model_id.split('/')[-1]}")
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(png, dpi=150)
    print(f"[out] {png}")


def analyze(rows, bw_gbps, kv_per_token):
    N = np.array([r["N"] for r in rows], float)
    dec = np.array([r["decode_ms_med"] for r in rows], float)
    print("\n==== analysis ====")
    print(f"decode_ms {dec[0]:.3f} -> {dec[-1]:.3f}  (x{dec[-1]/dec[0]:.2f})")
    if len(N) >= 2:
        a, b = np.polyfit(N, dec, 1)
        print(f"decode fit: {a*1e3:.4f} us/token * N + {b:.3f} ms floor")
        if kv_per_token:
            pred = kv_per_token / (bw_gbps * 1e9) * 1e3
            print(f"KV-read model {pred*1e3:.4f} us/token -> measured/predicted = {a/pred:.2f}")


def main():
    ap = argparse.ArgumentParser(description="Exp1 cost curves (Apple Silicon UMA).")
    ap.add_argument("--model", default="1.5B", help="registry key (0.5B/1.5B/3B/7B) or HF id")
    ap.add_argument("--ns", default="128,256,512,1024,2048,4096,8192")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=8.0)
    ap.add_argument("--mem-limit-gb", type=float, default=12.0)
    ap.add_argument("--bw-gbps", type=float, default=120.0)
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="fast: ns=128,256 reps=5 cooldown=2")
    args = ap.parse_args()
    if args.smoke:
        args.ns, args.reps, args.cooldown = "128,256", 5, 2.0
    run(args)


if __name__ == "__main__":
    main()
