#!/usr/bin/env python3
"""
plot_bw_vs_slowdown.py — P2 visualization: decode slowdown vs CPU logical bandwidth.

Pure plotting (no measurement). Reads the canonical M4 P2b CSV and makes one scatter:
x = logical CPU bandwidth (GB/s), y = GPU decode slowdown (%), one point per load type.
Sequential kernels (stream/memcpy/scan) trace a rough bandwidth->slowdown trend (fitted line
through those three only); the latency-bound patterns (random gather, RAG sidecar) sit far
ABOVE that trend — low logical bandwidth, large decode hit — because of cache-line amplification.
Two families are separated by marker SHAPE (grayscale-safe) and colour.

Column mapping (printed for verification): x=cpu_gbps_dec, y=decode_slow_pct, name=mode.
"""
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results", "csv", "exp_p2_loadtypes_1.5B_20260626.csv")
FIG_DIR = os.path.join(REPO, "results", "figures")
OUT = os.path.join(FIG_DIR, "exp_p2_bw_vs_slowdown_20260626")

SEQUENTIAL = ("stream", "memcpy", "scan")     # bandwidth-bound kernels (trace the trend)
LATENCY = ("random", "rag")                   # latency-bound (gather / RAG) — break the trend


def load_rows():
    with open(CSV) as f:
        reader = csv.DictReader(f)
        print("[csv] header:", ",".join(reader.fieldnames))
        print("[map]  x = cpu_gbps_dec   y = decode_slow_pct   name = mode")
        rows = {}
        for r in reader:
            rows[r["mode"]] = (float(r["cpu_gbps_dec"]), float(r["decode_slow_pct"]))
    return rows


def main():
    rows = load_rows()
    print("\n[plotted] name      bw(GB/s)   decode_slowdown(%)")
    for name in SEQUENTIAL + LATENCY:
        if name not in rows:
            sys.exit(f"[err] '{name}' not in {os.path.basename(CSV)}")
        bw, sl = rows[name]
        fam = "sequential" if name in SEQUENTIAL else "latency-bound"
        print(f"          {name:8s} {bw:7.2f}   {sl:6.1f}   ({fam})")

    sx = np.array([rows[n][0] for n in SEQUENTIAL])
    sy = np.array([rows[n][1] for n in SEQUENTIAL])
    slope, intercept = np.polyfit(sx, sy, 1)               # trend through the 3 sequential only
    xline = np.linspace(0, max(rows[n][0] for n in rows) * 1.05, 100)
    yline = slope * xline + intercept

    above = []
    for n in LATENCY:
        bw, sl = rows[n]
        pred = slope * bw + intercept
        above.append((n, sl, pred))
    print("\n[check] latency-bound points vs the sequential trend at their bandwidth:")
    for n, sl, pred in above:
        verdict = "ABOVE" if sl > pred else "below"
        ratio = f"  (~{sl / pred:.1f}x)" if pred > 1 else "  (trend ~0% there)"
        print(f"          {n:8s} actual {sl:5.1f}%  vs trend {pred:5.1f}%  -> {verdict}{ratio}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    # trend line through the sequential kernels (extended; below-zero part clipped by ylim)
    ax.plot(xline, yline, ls="--", color="0.55", lw=1.6, zorder=1,
            label="trend (sequential-kernel fit)")
    # sequential family — filled circles
    ax.scatter(sx, sy, s=120, marker="o", facecolor="steelblue", edgecolor="black",
               linewidth=0.8, zorder=3, label="sequential kernels (stream / memcpy / scan)")
    # latency-bound family — emphasized red stars
    lx = [rows[n][0] for n in LATENCY]; ly = [rows[n][1] for n in LATENCY]
    ax.scatter(lx, ly, s=320, marker="*", facecolor="crimson", edgecolor="black",
               linewidth=0.8, zorder=4, label="latency-bound (random gather, RAG sidecar)")

    off = {"stream": (6, 6), "memcpy": (6, 6), "scan": (8, -4),
           "random": (10, 2), "rag": (10, 4)}
    for n in SEQUENTIAL + LATENCY:
        bw, sl = rows[n]
        ax.annotate(n, (bw, sl), textcoords="offset points", xytext=off[n],
                    fontsize=11, fontweight="bold" if n in LATENCY else "normal",
                    color="crimson" if n in LATENCY else "black")
    # dotted connectors: latency-bound point down to the trend at its bandwidth (the "break")
    for n in LATENCY:
        bw, sl = rows[n]
        ax.plot([bw, bw], [max(slope * bw + intercept, 0), sl], ls=":", color="crimson",
                lw=1.0, alpha=0.6, zorder=2)

    ax.set_xlim(0, max(rows[n][0] for n in rows) * 1.08)
    ax.set_ylim(0, max(rows[n][1] for n in rows) * 1.15)
    ax.set_xlabel("CPU logical bandwidth (GB/s)")
    ax.set_ylabel("GPU decode slowdown vs solo (%)")
    ax.set_title("Decode slowdown vs CPU logical bandwidth (M4, 1.5B)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(OUT + ".png", dpi=200)
    fig.savefig(OUT + ".pdf")                              # vector version for publication
    print(f"\n[out] {OUT}.png")
    print(f"[out] {OUT}.pdf")
    print("[caption] Sequential kernels (stream/memcpy/scan) trace a rough bandwidth->slowdown "
          "trend; random gather and the RAG sidecar sit far above it (low logical bandwidth, "
          "large decode hit) due to cache-line amplification.")


if __name__ == "__main__":
    main()
