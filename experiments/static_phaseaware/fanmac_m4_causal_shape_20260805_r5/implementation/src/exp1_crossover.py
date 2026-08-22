#!/usr/bin/env python3
"""
exp1_crossover.py - keep-and-read vs drop-and-recompute, from the Exp1 cost curves.

Reads exp1.csv + exp1_meta.json and works out, per context length N, the decision a
KV-cache policy faces under memory pressure once a sequence's KV must leave the active
working set:

  drop-and-recompute : pay prefill_ms(N) to rebuild the KV from scratch when resumed.
  keep / move        : keep the KV addressable and read it back. On PCIe this is a
                       swap over the bus (KV_bytes / B_pcie); on unified memory the KV
                       never physically moves, so "read-back" ~ KV_bytes / B_uma ~ free.

Two quantities make the contrast precise:

  recompute tax (decode-steps) = prefill_ms(N) / decode_step_ms(N)
      How much *added latency*, in units of decode steps, you eat by dropping and later
      recomputing instead of keeping. (Keep always wins on pure latency; this is the price.)

  B*(N) = KV_bytes(N) / prefill_s(N)            [the crossover bandwidth]
      Recompute beats read-back iff the read-back bandwidth is BELOW B*(N):
          prefill_ms(N) < KV_bytes(N) / B   <=>   B < B*(N).
      So B*(N) is the bandwidth at which the swap-vs-recompute decision flips. Comparing
      B*(N) to PCIe (~16-32 GB/s) vs unified memory (~60-120 GB/s) tells us whether the
      FlexGen/vLLM crossover even exists on each platform.

This is analysis only (no model, no MLX) - pure arithmetic on the measured curves.
"""
import argparse
import csv
import json


def load(csv_path, meta_path):
    with open(meta_path) as f:
        kv_bpt = json.load(f)["kv_bytes_per_token"]
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                N=int(r["N"]),
                prefill_ms=float(r["prefill_ms_med"]),
                decode_ms=float(r["decode_ms_med"]),
                kv_bytes=int(round(float(r["kv_mb"]) * 1024 * 1024)),
            ))
    return rows, kv_bpt


def analyze(rows, kv_bpt, pcie_gbps, uma_peak_gbps, uma_eff_gbps):
    out = []
    for r in rows:
        N = r["N"]
        kv_b = r["kv_bytes"]
        prefill_s = r["prefill_ms"] / 1e3
        tax = r["prefill_ms"] / r["decode_ms"]                  # in decode-steps
        b_star = kv_b / prefill_s / 1e9                         # GB/s
        pcie_in = kv_b / (pcie_gbps * 1e9) * 1e3               # one-way swap-in, ms
        uma_peak = kv_b / (uma_peak_gbps * 1e9) * 1e3
        uma_eff = kv_b / (uma_eff_gbps * 1e9) * 1e3
        out.append(dict(
            N=N, prefill_ms=r["prefill_ms"], decode_ms=r["decode_ms"],
            kv_mb=kv_b / 1024 / 1024,
            recompute_tax_steps=tax,
            b_star_gbps=b_star,
            pcie_swapin_ms=pcie_in, pcie_roundtrip_ms=2 * pcie_in,
            uma_read_ms=uma_peak, uma_read_eff_ms=uma_eff,
            recompute_beats_pcie=(r["prefill_ms"] < 2 * pcie_in),
            recompute_beats_uma=(r["prefill_ms"] < uma_eff),
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="exp1.csv")
    ap.add_argument("--meta", default="exp1_meta.json")
    ap.add_argument("--pcie-gbps", type=float, default=25.0,  # PCIe4 x16 effective
                    help="PCIe swap bandwidth (vLLM/FlexGen offload path)")
    ap.add_argument("--uma-peak-gbps", type=float, default=120.0)
    ap.add_argument("--uma-eff-gbps", type=float, default=62.0)  # measured KV-read effective
    ap.add_argument("--out-prefix", default="exp1_crossover")
    args = ap.parse_args()

    rows, kv_bpt = load(args.csv, args.meta)
    res = analyze(rows, kv_bpt, args.pcie_gbps, args.uma_peak_gbps, args.uma_eff_gbps)

    print(f"KV bytes/token = {kv_bpt}  ({kv_bpt/1024:.1f} KiB)")
    print(f"PCIe swap BW = {args.pcie_gbps} GB/s | UMA peak = {args.uma_peak_gbps} | "
          f"UMA effective (measured KV-read) = {args.uma_eff_gbps} GB/s\n")
    hdr = (f"{'N':>5} {'prefill_ms':>10} {'decode_ms':>9} {'KV_MB':>7} "
           f"{'tax(steps)':>10} {'B*(GB/s)':>9} {'PCIe_in_ms':>10} {'UMA_in_ms':>9} "
           f"{'recomp<PCIe?':>12} {'recomp<UMA?':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in res:
        print(f"{r['N']:>5} {r['prefill_ms']:>10.1f} {r['decode_ms']:>9.2f} "
              f"{r['kv_mb']:>7.1f} {r['recompute_tax_steps']:>10.1f} "
              f"{r['b_star_gbps']:>9.3f} {r['pcie_swapin_ms']:>10.3f} "
              f"{r['uma_read_ms']:>9.3f} {str(r['recompute_beats_pcie']):>12} "
              f"{str(r['recompute_beats_uma']):>11}")

    # ---- write CSV --------------------------------------------------------------
    cols = ["N", "prefill_ms", "decode_ms", "kv_mb", "recompute_tax_steps",
            "b_star_gbps", "pcie_swapin_ms", "pcie_roundtrip_ms", "uma_read_ms",
            "uma_read_eff_ms", "recompute_beats_pcie", "recompute_beats_uma"]
    with open(f"{args.out_prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in res:
            w.writerow(r)
    print(f"\n[out] wrote {args.out_prefix}.csv")

    # ---- key findings -----------------------------------------------------------
    bmin = min(r["b_star_gbps"] for r in res)
    bmax = max(r["b_star_gbps"] for r in res)
    print("\n================ CROSSOVER FINDINGS ================")
    print(f"crossover bandwidth B*(N) = {bmin:.3f}-{bmax:.3f} GB/s "
          f"(~{bmin*1000:.0f}-{bmax*1000:.0f} MB/s)")
    print(f"  -> PCIe ({args.pcie_gbps:.0f} GB/s) is {args.pcie_gbps/bmax:.0f}-"
          f"{args.pcie_gbps/bmin:.0f}x above B*  => swap beats recompute on PCIe too")
    print(f"  -> UMA  ({args.uma_eff_gbps:.0f} GB/s) is {args.uma_eff_gbps/bmax:.0f}-"
          f"{args.uma_eff_gbps/bmin:.0f}x above B*  => keep/read dominates on UMA")
    print(f"recompute tax grows {res[0]['recompute_tax_steps']:.0f} -> "
          f"{res[-1]['recompute_tax_steps']:.0f} decode-steps "
          f"(N={res[0]['N']} -> {res[-1]['N']}), super-linear")
    any_pcie = any(r["recompute_beats_pcie"] for r in res)
    any_uma = any(r["recompute_beats_uma"] for r in res)
    print(f"recompute ever beats PCIe swap? {any_pcie}   ever beats UMA read? {any_uma}")
    print("===================================================")

    plot(res, f"{args.out_prefix}.png", args)


def plot(res, png, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = [r["N"] for r in res]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # left: cost to re-materialize an N-token KV cache (log-y)
    axL.set_xscale("log", base=2)
    axL.set_yscale("log")
    axL.plot(N, [r["prefill_ms"] for r in res], "o-", color="tab:blue",
             label="drop-and-recompute  (= prefill_ms)")
    axL.plot(N, [r["pcie_roundtrip_ms"] for r in res], "^--", color="tab:orange",
             label=f"PCIe swap round-trip ({args.pcie_gbps:.0f} GB/s)")
    axL.plot(N, [r["uma_read_ms"] for r in res], "s-", color="tab:green",
             label=f"UMA read ({args.uma_peak_gbps:.0f} GB/s peak)")
    axL.plot(N, [r["uma_read_eff_ms"] for r in res], "s:", color="tab:red",
             label=f"UMA read ({args.uma_eff_gbps:.0f} GB/s effective)")
    axL.set_xticks(N); axL.set_xticklabels([str(n) for n in N])
    axL.set_xlabel("context length N (tokens)")
    axL.set_ylabel("cost to recover N-token KV (ms, log)")
    axL.set_title("Recover an evicted KV cache: recompute vs read-back")
    axL.grid(True, which="both", alpha=0.25)
    axL.legend(fontsize=8)

    # right: recompute tax in decode-steps
    axR.set_xscale("log", base=2)
    axR.plot(N, [r["recompute_tax_steps"] for r in res], "o-", color="tab:purple")
    axR.set_xticks(N); axR.set_xticklabels([str(n) for n in N])
    axR.set_xlabel("context length N (tokens)")
    axR.set_ylabel("recompute tax = prefill_ms / decode_step_ms  (decode-steps)")
    axR.set_title("Latency price of dropping (vs keeping) an N-token KV")
    axR.grid(True, which="both", alpha=0.25)
    for r in res:
        axR.annotate(f"{r['recompute_tax_steps']:.0f}", (r["N"], r["recompute_tax_steps"]),
                     textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")

    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"[out] wrote {png}")


if __name__ == "__main__":
    main()
