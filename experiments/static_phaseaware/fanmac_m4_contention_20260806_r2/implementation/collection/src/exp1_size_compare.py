#!/usr/bin/env python3
"""
exp1_size_compare.py - overlay the Exp1 cost curves across model sizes.

Reads every per-size CSV produced by exp1_costcurve.py that exists on disk and asks:
as the model grows, what happens to
  - prefill_ms(N) and decode_step_ms(N),
  - the recompute tax  = prefill_ms / decode_step_ms      (added latency, in decode-steps),
  - the crossover bandwidth  B*(N) = KV_bytes(N) / prefill_s(N),
  - the UMA-vs-PCIe recovery gap = prefill_ms / (PCIe swap round-trip)?

The hypothesis-sharpening question: does the gap (UMA forced-recompute vs the PCIe swap it
replaces) WIDEN with model size? Pure arithmetic on the measured curves - no MLX, no model.
"""
import argparse
import csv
import os

# (label, out-prefix) ordered by parameter count; only those present are used.
SIZES = [("0.5B", "exp1_0p5b"), ("1.5B", "exp1"), ("3B", "exp1_3b"), ("7B", "exp1_7b")]
COLORS = {"0.5B": "tab:green", "1.5B": "tab:blue", "3B": "tab:orange", "7B": "tab:red"}


def load_one(prefix):
    path = f"{prefix}.csv"
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                N=int(r["N"]),
                prefill_ms=float(r["prefill_ms_med"]),
                decode_ms=float(r["decode_ms_med"]),
                kv_bytes=float(r["kv_mb"]) * 1024 * 1024,
            ))
    return rows


def derive(rows, pcie_gbps):
    for r in rows:
        r["tax"] = r["prefill_ms"] / r["decode_ms"]
        r["b_star_gbps"] = r["kv_bytes"] / (r["prefill_ms"] / 1e3) / 1e9
        swap_roundtrip_ms = 2 * r["kv_bytes"] / (pcie_gbps * 1e9) * 1e3
        r["gap_vs_pcie"] = r["prefill_ms"] / swap_roundtrip_ms
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcie-gbps", type=float, default=25.0)
    ap.add_argument("--out-prefix", default="exp1_size_compare")
    args = ap.parse_args()

    data = []
    for label, prefix in SIZES:
        rows = load_one(prefix)
        if rows is None:
            print(f"[skip] {label}: {prefix}.csv not found")
            continue
        derive(rows, args.pcie_gbps)
        data.append((label, rows))
        print(f"[ok]   {label}: {len(rows)} points from {prefix}.csv")

    if not data:
        print("no size CSVs found - run exp1_costcurve.py first")
        return

    # ---- comparison tables at representative N --------------------------------------
    for probe in (2048, 4096, 8192):
        print(f"\n===== cross-size @ N={probe} =====")
        print(f"{'size':>6} {'prefill_ms':>10} {'decode_ms':>9} {'tax(steps)':>10} "
              f"{'B*(GB/s)':>9} {'gap_vs_PCIe':>11}")
        for label, rows in data:
            r = next((x for x in rows if x["N"] == probe), None)
            if r is None:
                print(f"{label:>6} {'(no point)':>10}")
                continue
            print(f"{label:>6} {r['prefill_ms']:>10.1f} {r['decode_ms']:>9.2f} "
                  f"{r['tax']:>10.1f} {r['b_star_gbps']:>9.3f} {r['gap_vs_pcie']:>11.0f}")

    # ---- long-form CSV ---------------------------------------------------------------
    with open(f"{args.out_prefix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "N", "prefill_ms", "decode_ms", "tax_steps",
                    "b_star_gbps", "gap_vs_pcie"])
        for label, rows in data:
            for r in rows:
                w.writerow([label, r["N"], f"{r['prefill_ms']:.3f}", f"{r['decode_ms']:.4f}",
                            f"{r['tax']:.2f}", f"{r['b_star_gbps']:.4f}", f"{r['gap_vs_pcie']:.2f}"])
    print(f"\n[out] wrote {args.out_prefix}.csv")

    # ---- key finding -----------------------------------------------------------------
    print("\n================ SIZE-SCALING FINDING ================")
    for probe in (4096, 8192, 2048):
        pts = [(label, next((x for x in rows if x["N"] == probe), None)) for label, rows in data]
        pts = [(l, r) for l, r in pts if r]
        if len(pts) >= 2:
            taxes = ", ".join(f"{l}={r['tax']:.0f}" for l, r in pts)
            gaps = ", ".join(f"{l}={r['gap_vs_pcie']:.0f}x" for l, r in pts)
            print(f"@N={probe}: recompute tax (steps): {taxes}")
            print(f"@N={probe}: UMA-vs-PCIe gap:        {gaps}")
    print("=> if tax and gap rise with size, UMA's forced-recompute penalty worsens for")
    print("   bigger models, i.e. the PCIe cost model transfers even worse at scale.")
    print("=====================================================")

    plot(data, f"{args.out_prefix}.png", args.pcie_gbps)


def plot(data, png, pcie_gbps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (a, b), (c, d) = axes

    def style(ax):
        ax.set_xscale("log", base=2)
        Ns = sorted({r["N"] for _, rows in data for r in rows})
        ax.set_xticks(Ns)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel("context length N (tokens)")
        ax.grid(True, which="both", alpha=0.25)

    for label, rows in data:
        N = [r["N"] for r in rows]
        col = COLORS.get(label, "gray")
        a.plot(N, [r["prefill_ms"] for r in rows], "o-", color=col, label=label)
        b.plot(N, [r["decode_ms"] for r in rows], "o-", color=col, label=label)
        c.plot(N, [r["tax"] for r in rows], "o-", color=col, label=label)
        d.plot(N, [r["gap_vs_pcie"] for r in rows], "o-", color=col, label=label)

    a.set_yscale("log"); a.set_ylabel("prefill_ms (log)")
    a.set_title("prefill: drop-and-recompute cost")
    b.set_ylabel("decode_step_ms"); b.set_title("decode: keep-and-read cost")
    c.set_yscale("log"); c.set_ylabel("recompute tax (decode-steps, log)")
    c.set_title("recompute tax = prefill / decode_step")
    d.set_yscale("log"); d.set_ylabel(f"prefill_ms / PCIe swap ({pcie_gbps:.0f} GB/s), log")
    d.set_title("UMA forced-recompute vs PCIe swap: the gap")
    for ax in (a, b, c, d):
        style(ax); ax.legend(fontsize=9, title="model")

    fig.suptitle("Exp1 size sweep - cost curves & UMA-vs-PCIe gap (M4 MBA, 4bit)", fontsize=13)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"[out] wrote {png}")


if __name__ == "__main__":
    main()
