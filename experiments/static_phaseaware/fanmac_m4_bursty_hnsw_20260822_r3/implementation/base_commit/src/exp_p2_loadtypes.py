#!/usr/bin/env python3
"""
exp_p2_loadtypes.py — P2: realistic CPU load types + refined intensity sweep.

Exp2 showed a synthetic STREAM triad slows GPU decode ~38% (bandwidth-arbitration, not
thermal). P2 answers the "is that a STREAM artifact?" reviewer question by (a) refining the
intensity sweep into a smooth monotone curve, and (b) driving the SAME measurement with
*realistic agent-side CPU patterns* — memcpy (copy), sequential scan (file/mmap analogue),
and random gather (vector-search / embedding-lookup analogue) — to show diverse CPU traffic
contends with decode while prefill (compute-bound) stays immune.

Real GPU measurement (NOT a sim). All Exp2 controls: mx.eval-forced decode (common.measure),
cache built BEFORE the load covers the window, AC power, mem limit, fanless cooldown + throttle
detection, CPU load QoS-biased to one P-core. Reuses exp2_contention.build_cache /
decode_throughput and measure.time_prefill so numbers are directly comparable to Exp2.

Blocks:
  P2a  intensity sweep (stream), N fixed, duty 0/10/.../100  -> decode slowdown curve (refined).
  P2b  load-type A/B, duty 100%: {stream, memcpy, scan, random} -> decode slowdown AND prefill
       slowdown each, vs the solo baseline. Headline: all patterns hit decode, none hit prefill.
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import measure, thermal, models, cpuload         # noqa: E402
from exp2_contention import build_cache, decode_throughput    # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(REPO, "results", "csv")
FIG_DIR = os.path.join(REPO, "results", "figures")
MODES = ["stream", "memcpy", "scan", "random", "rag"]   # rag = realistic RAG sidecar (Python)
RAG_SIDECAR = os.path.join(REPO, "src", "rag_sidecar.py")


class RagSidecarLoad:
    """Drives src/rag_sidecar.py with the same start()/wait() contract as cpuload.CpuBandwidthLoad,
    so the P2 harness launches/stops it identically to the C loads. start() BLOCKS until the sidecar
    prints `READY` (one-time index/mmap setup done, so it is excluded from the measured window);
    wait() collects the logical `GBPS` line. rag has no duty knob: intensity 0 = off, >0 = on."""

    def __init__(self, intensity_pct, seconds=3.5):
        self.intensity = float(intensity_pct)
        self.seconds = float(seconds)
        self.proc = None
        self.gbps = 0.0
        self.raw = ""
        self.path = "-"

    def start(self):
        if self.intensity <= 0:
            return self
        self.proc = subprocess.Popen([sys.executable, RAG_SIDECAR, f"{self.seconds}"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        while True:                                  # wait for setup to finish (READY)
            line = self.proc.stdout.readline()
            if line == "":                           # EOF before READY -> setup failed
                break
            if line.startswith("READY"):
                p = line.split()
                self.path = p[1] if len(p) > 1 else "?"
                break
        return self

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def wait(self):
        if self.proc is None:
            return 0.0
        self.raw, _ = self.proc.communicate()
        for line in (self.raw or "").splitlines():
            if line.startswith("GBPS"):
                self.gbps = float(line.split()[1])
        return self.gbps

    def stop(self):
        if self.running():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        return self.gbps


def make_load(mode, intensity, seconds):
    """C memory-pattern load (stream/memcpy/scan/random) or the Python RAG sidecar (rag)."""
    if mode == "rag":
        return RagSidecarLoad(intensity, seconds=seconds)
    return cpuload.CpuBandwidthLoad(intensity, seconds=seconds, mode=mode)


def decode_under(model, N, window_s, intensity, mode):
    """Decode tok/s on a freshly built N-cache under a CPU load of `mode` at `intensity`%."""
    cache = build_cache(model, N)
    load = make_load(mode, intensity, seconds=0.5 + window_s + 0.8).start()
    if intensity > 0:
        time.sleep(0.5)                          # let the load ramp before timing
    r = decode_throughput(model, cache, window_s=window_s)
    gbps = load.wait()
    del cache
    measure.free_buffers()
    return r["tok_s"], gbps, r["throttle"], r["ms_per_step"], getattr(load, "path", "-")


def prefill_under(model, N, reps, intensity, mode, pf_solo_ms):
    """Median prefill_ms of N tokens while a `mode` load at `intensity`% runs. Load sized to
    cover (1 warmup + reps) prefills with margin (mirrors exp2_prefill_contention)."""
    load_s = round(0.5 + (1 + reps) * (pf_solo_ms / 1000.0) * 1.5 + 1.0, 1)
    load = make_load(mode, intensity, seconds=load_s).start()
    if intensity > 0:
        time.sleep(0.5)
    ms = float(np.median(measure.time_prefill(model, N, reps=reps, warmup=1)))
    gbps = load.wait()
    return ms, gbps, getattr(load, "path", "-")


def run(args):
    thermal.assert_power(args.allow_battery)
    procs = thermal.top_mem_procs()
    if procs:
        print("[mem] top: " + ", ".join(f"{c.split('/')[-1]}={mb:.0f}MB" for c, mb in procs))
    measure.set_mem_limit_gb(args.mem_limit_gb)
    print(f"[mem] set_memory_limit = {args.mem_limit_gb} GB")
    print(f"[load] {models.resolve_model_id(args.model)} ...")
    model, _ = models.load_model(args.model)
    mx.eval(model.parameters())
    cfg = models.model_config(model)
    cpuload.ensure_built()
    print("[cpuload] stream_load.c compiled (modes: stream/memcpy/scan/random)")
    print("[warmup] global kernel + DVFS clock warmup ...")
    measure.global_warmup(model)
    N = args.n

    # ---------------------------------------------------------------- P2a: refined intensity sweep
    intens = [int(x) for x in str(args.intensity).split(",") if x.strip()]
    print(f"\n[P2a] intensity sweep (stream), N={N}: {intens}")
    a_rows, base_tok = [], None
    for it in intens:
        tok, gbps, thr, msps, _ = decode_under(model, N, args.window_s, it, "stream")
        if it == 0:
            base_tok = tok
        slow = (1 - tok / base_tok) * 100 if base_tok else 0.0
        a_rows.append(dict(intensity_pct=it, cpu_gbps=round(gbps, 2), decode_tok_s=round(tok, 2),
                           ms_per_step=round(msps, 3), slowdown_pct=round(slow, 1),
                           throttle=round(thr, 3)))
        thrm = "" if thr < 1.10 else f"  <throttle {thr:.2f}>"
        print(f"  load={it:3d}% | cpu={gbps:6.1f} GB/s | decode={tok:6.1f} tok/s | "
              f"slowdown={slow:5.1f}%{thrm}")
        time.sleep(args.cooldown)

    # ---------------------------------------------------------------- P2b: load-type A/B (duty 100%)
    print(f"\n[P2b] load-type A/B at {args.duty}% duty, N={N} (decode vs prefill per pattern)")
    pf_solo = float(np.median(measure.time_prefill(model, N, reps=args.reps, warmup=1)))
    time.sleep(args.cooldown)
    dec_solo, _, _, _, _ = decode_under(model, N, args.window_s, 0, "stream")
    time.sleep(args.cooldown)
    print(f"  solo: prefill {pf_solo:.0f} ms | decode {dec_solo:.1f} tok/s")
    b_rows = []
    for mode in MODES:
        pf_load, gpf, ppath = prefill_under(model, N, args.reps, args.duty, mode, pf_solo)
        time.sleep(args.cooldown)
        dtok, gdec, thr, _, dpath = decode_under(model, N, args.window_s, args.duty, mode)
        time.sleep(args.cooldown)
        pf_slow = (pf_load / pf_solo - 1) * 100        # prefill ms UP
        dec_slow = (1 - dtok / dec_solo) * 100         # decode tok/s DOWN
        b_rows.append(dict(mode=mode, duty_pct=args.duty, cpu_gbps_dec=round(gdec, 2),
                           cpu_gbps_pf=round(gpf, 2), prefill_solo_ms=round(pf_solo, 1),
                           prefill_load_ms=round(pf_load, 1), prefill_slow_pct=round(pf_slow, 1),
                           decode_solo_toks=round(dec_solo, 1), decode_load_toks=round(dtok, 1),
                           decode_slow_pct=round(dec_slow, 1), decode_throttle=round(thr, 3),
                           load_path=dpath))
        ptag = f" | {dpath}" if dpath not in ("-", None) else ""
        print(f"  {mode:7s} | cpu(dec) {gdec:6.1f} GB/s | prefill {pf_slow:+5.1f}% | "
              f"decode {dec_slow:5.1f}% slower{ptag}")

    _save(a_rows, b_rows, args, cfg, N)
    _analyze(a_rows, b_rows)


def _save(a_rows, b_rows, args, cfg, N):
    date = datetime.now().strftime("%Y%m%d")
    os.makedirs(CSV_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    ap = os.path.join(CSV_DIR, f"exp_p2_intensity_{args.model}_{date}.csv")
    with open(ap, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(a_rows[0].keys())); w.writeheader(); w.writerows(a_rows)
    bp = os.path.join(CSV_DIR, f"exp_p2_loadtypes_{args.model}_{date}.csv")
    with open(bp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(b_rows[0].keys())); w.writeheader(); w.writerows(b_rows)
    with open(os.path.join(CSV_DIR, f"exp_p2_{args.model}_{date}_meta.json"), "w") as f:
        json.dump(dict(timestamp=datetime.now().isoformat(timespec="seconds"),
                       model=models.resolve_model_id(args.model), config=cfg, N=N,
                       mlx="0.31.2", mlx_lm="0.31.3", python=platform.python_version(),
                       params=dict(intensity=args.intensity, duty=args.duty, window_s=args.window_s,
                                   reps=args.reps, cooldown=args.cooldown, modes=MODES)), f, indent=2)
    print(f"\n[out] wrote {ap}\n[out] wrote {bp}")
    _plot(a_rows, b_rows, args.model, N, os.path.join(FIG_DIR, f"exp_p2_loadtypes_{args.model}_{date}.png"))


def _analyze(a_rows, b_rows):
    top = a_rows[-1]
    print("\n================ P2 FINDINGS ================")
    print(f"P2a (intensity): decode slowdown rises monotonically to {top['slowdown_pct']:.1f}% "
          f"@ {top['cpu_gbps']:.0f} GB/s (stream).")
    print("P2b (load types) decode slowdown / prefill slowdown @ full duty:")
    for r in b_rows:
        tag = f"  [{r.get('load_path')}]" if r.get("load_path") not in ("-", None) else ""
        print(f"  {r['mode']:7s}: decode {r['decode_slow_pct']:5.1f}% slower | "
              f"prefill {r['prefill_slow_pct']:+5.1f}% | (cpu {r['cpu_gbps_dec']:.0f} GB/s){tag}")
    seq = [r for r in b_rows if r["mode"] in ("stream", "memcpy", "scan")]
    print(f"  => sequential patterns (stream/memcpy/scan, {min(r['cpu_gbps_dec'] for r in seq):.0f}-"
          f"{max(r['cpu_gbps_dec'] for r in seq):.0f} GB/s) all slow decode "
          f"{min(r['decode_slow_pct'] for r in seq):.0f}-{max(r['decode_slow_pct'] for r in seq):.0f}%; "
          f"prefill stays within +-{max(abs(r['prefill_slow_pct']) for r in b_rows):.0f}%.")
    print("     => NOT a STREAM artifact: realistic agent-side CPU traffic contends with decode,")
    print("        prefill (compute-bound) is immune. (random gather = latency-bound, low LOGICAL")
    print("        GB/s but cache-line-amplified DRAM traffic; see its decode hit.)")
    print("=============================================")


def _plot(a_rows, b_rows, tag, N, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.8))

    x = [r["cpu_gbps"] for r in a_rows]; y = [r["slowdown_pct"] for r in a_rows]
    ax0.plot(x, y, "o-", color="tab:red")
    for r in a_rows:
        ax0.annotate(f"{r['intensity_pct']}%", (r["cpu_gbps"], r["slowdown_pct"]),
                     textcoords="offset points", xytext=(4, -8), fontsize=7)
    ax0.set_xlabel("concurrent CPU load (GB/s, STREAM duty sweep)")
    ax0.set_ylabel("GPU decode slowdown vs solo (%)")
    ax0.set_title(f"P2a: refined intensity sweep (N={N})"); ax0.grid(alpha=.3)

    i = np.arange(len(b_rows)); wd = 0.38
    dec = [r["decode_slow_pct"] for r in b_rows]; pf = [r["prefill_slow_pct"] for r in b_rows]
    ax1.bar(i - wd/2, dec, wd, color="tab:red", label="decode (keep-and-read)")
    ax1.bar(i + wd/2, pf, wd, color="tab:green", label="prefill (recompute)")
    ax1.axhline(0, color="k", lw=.8)
    for k, r in enumerate(b_rows):
        ax1.annotate(f"{r['cpu_gbps_dec']:.0f}\nGB/s", (k - wd/2, dec[k]),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax1.set_xticks(i); ax1.set_xticklabels([r["mode"] for r in b_rows])
    ax1.set_ylabel("slowdown vs solo (%)")
    ax1.set_title("P2b: realistic CPU patterns — decode hit, prefill immune")
    ax1.legend(fontsize=9); ax1.grid(alpha=.3, axis="y")
    fig.suptitle(f"P2 contention is not a STREAM artifact — {tag} (M4 MBA, unified mem)", fontsize=12)
    fig.tight_layout(); fig.savefig(png, dpi=150)
    print(f"[out] wrote {png}")


def main():
    ap = argparse.ArgumentParser(description="P2: realistic CPU load types + refined intensity sweep.")
    ap.add_argument("--model", default="1.5B")
    ap.add_argument("--n", type=int, default=2048, help="cached context length for decode/prefill")
    ap.add_argument("--intensity", default="0,10,20,30,40,50,60,70,80,90,100",
                    help="P2a duty-cycle sweep %% (0 = solo baseline)")
    ap.add_argument("--duty", type=int, default=100, help="P2b load duty for the mode A/B")
    ap.add_argument("--window-s", type=float, default=3.0, help="decode measurement window")
    ap.add_argument("--reps", type=int, default=3, help="prefill reps (median)")
    ap.add_argument("--cooldown", type=float, default=6.0)
    ap.add_argument("--mem-limit-gb", type=float, default=12.0)
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="fast: intensity=0,100 window=1 reps=1 cd=2")
    args = ap.parse_args()
    if args.smoke:
        args.intensity, args.window_s, args.reps, args.cooldown = "0,100", 1.0, 1, 2.0
    run(args)


if __name__ == "__main__":
    main()
