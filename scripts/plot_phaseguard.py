#!/usr/bin/env python3
"""Generate publication-ready PhaseGuard Figures A-E as PNG and PDF."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

REPO = Path(__file__).resolve().parents[1]
COLORS = {"uncoordinated": "#d62728", "serialized": "#7f7f7f", "static-0": "#9467bd",
          "static-1": "#ff7f0e", "static-2": "#bcbd22", "static-4": "#8c564b",
          "phaseguard": "#1f77b4", "bandwidth_only": "#2ca02c",
          "oracle-best-observed": "black"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f: return list(csv.DictReader(f))


def save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig); print(f"[out] {out}.png/.pdf")


def figure_a(profile: list[dict[str, str]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for phase, color, marker in (("PREFILL", "#2ca02c", "s"), ("DECODE", "#d62728", "o")):
        rows = sorted((r for r in profile if r["phase"] == phase), key=lambda r: int(r["workers"]))
        ax.plot([int(r["workers"]) for r in rows], [float(r["slowdown"]) for r in rows],
                marker=marker, lw=2.2, color=color, label=phase.lower())
    ax.axhline(1, color="black", lw=.8); ax.axhline(1.10, color="0.5", ls="--", lw=1)
    ax.set(xlabel="permitted HNSW worker processes", ylabel="normalized p95 phase time",
           title="A. Real HNSW retrieval: phase asymmetry")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    save(fig, out)


def figure_b(summary: list[dict[str, str]], out: Path) -> None:
    rows = [r for r in summary if float(r["slo_multiplier"]) == 1.15]
    concurrencies = sorted({int(r["concurrency"]) for r in rows})
    fig, axes = plt.subplots(1, len(concurrencies), figsize=(11.5, 4.8), sharey=True)
    if len(concurrencies) == 1: axes = [axes]
    handles, labels = [], []
    for ax, concurrency in zip(axes, concurrencies):
        for r in (x for x in rows if int(x["concurrency"]) == concurrency):
            p = r["policy"]; marker = "*" if p == "oracle-best-observed" else "o"
            artist = ax.scatter(float(r["request_throughput_s"]), float(r["p95_tpot_ms"]),
                                s=170 if marker == "*" else 75, marker=marker,
                                color=COLORS.get(p, "0.4"), edgecolor="black", linewidth=.5, label=p)
            if p not in labels: handles.append(artist); labels.append(p)
        ax.set_xlabel("workflow throughput (requests/s)"); ax.set_title(f"concurrency {concurrency}")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("p95 TPOT (ms)")
    fig.suptitle("B. Scheduling Pareto frontier (lower-right is better)", y=1.03)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .97), ncol=5,
               fontsize=7, frameon=False)
    save(fig, out)


def figure_c(summary: list[dict[str, str]], out: Path) -> None:
    rows = [r for r in summary if float(r["slo_multiplier"]) == 1.15 and r["policy"] != "oracle-best-observed"]
    policies = list(dict.fromkeys(r["policy"] for r in rows)); conc = sorted({int(r["concurrency"]) for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6)); width = .8 / max(1, len(policies)); x = np.arange(len(conc))
    specs = (("p95_tpot_ms", "p95 TPOT (ms)"), ("token_slo_violation_rate", "token SLO violation rate"),
             ("p95_request_ms", "p95 end-to-end latency (ms)"))
    for ax, (metric, ylabel) in zip(axes, specs):
        for i, policy in enumerate(policies):
            vals = [next((float(r[metric]) for r in rows if r["policy"] == policy and int(r["concurrency"]) == c), np.nan) for c in conc]
            ax.bar(x + (i - (len(policies)-1)/2) * width, vals, width, label=policy,
                   color=COLORS.get(policy, "0.5"))
        ax.set_xticks(x, [str(c) for c in conc]); ax.set_xlabel("closed-loop concurrency")
        ax.set_ylabel(ylabel); ax.grid(alpha=.2, axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .97), ncol=5,
               fontsize=7, frameon=False)
    fig.suptitle("C. Policy comparison", y=1.03)
    save(fig, out)


def figure_d(out: Path) -> None:
    source = REPO / "results/csv/exp_p2_loadtypes_1.5B_20260626.csv"
    rows = read_csv(source); fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for r in rows:
        mode = r["mode"]; latency = mode in ("random", "rag")
        ax.scatter(float(r["cpu_gbps_dec"]), float(r["decode_slow_pct"]),
                   marker="*" if latency else "o", s=210 if latency else 85,
                   color="#d62728" if latency else "#1f77b4", edgecolor="black")
        ax.annotate(mode, (float(r["cpu_gbps_dec"]), float(r["decode_slow_pct"])),
                    xytext=(5, 4), textcoords="offset points")
    ax.annotate("2.4 GB/s random > 47.6 GB/s scan\n(28.3% vs 22.3% slowdown)",
                xy=(2.39, 28.3), xytext=(18, 35), arrowprops={"arrowstyle": "->"}, fontsize=9)
    ax.set(xlabel="logical CPU throughput (GB/s)", ylabel="decode slowdown (%)",
           title="D. Logical bandwidth is an unsafe interference proxy")
    ax.grid(alpha=.25); save(fig, out)


def timeline_panel(ax: plt.Axes, data: dict[str, object], title: str) -> None:
    events = data["events"]; retrieval = data["retrieval"]
    times = [float(e["timestamp"]) for e in events] + [float(r["retrieval_start"]) for r in retrieval]
    origin = min(times); max_worker = max((int(r["retrieval_worker"]) for r in retrieval), default=0)
    for r in retrieval:
        start = float(r["retrieval_start"]) - origin; width = float(r["retrieval_end"]) - float(r["retrieval_start"])
        ax.broken_barh([(start, width)], (int(r["retrieval_worker"]), .72), facecolors="#4c78a8", alpha=.75)
    starts: dict[tuple[str, str | None], float] = {}
    for e in events:
        key = (str(e["phase"]), e.get("request_id"))
        if e["event"] == "start": starts[key] = float(e["timestamp"])
        elif key in starts and key[0] != "IDLE":
            color = "#f2cf5b" if key[0] == "PREFILL" else "#e45756"
            start = starts.pop(key)
            ax.broken_barh([(start - origin, float(e["timestamp"]) - start)],
                           (max_worker + 1, .72), facecolors=color)
    phase_starts = [e for e in events if e["event"] == "start" and e.get("permitted_workers") is not None]
    if phase_starts:
        tx = [float(e["timestamp"]) - origin for e in phase_starts]
        py = [float(e["permitted_workers"]) for e in phase_starts]
        twin = ax.twinx(); twin.step(tx, py, where="post", color="black", lw=1.2, alpha=.7)
        twin.set_ylim(-.2, max_worker + .5); twin.set_ylabel("CPU permits", fontsize=8)
    ax.set_yticks(list(range(max_worker + 1)) + [max_worker + 1],
                  [f"CPU {i}" for i in range(max_worker + 1)] + ["GPU"])
    ax.set_xlabel("time since run start (s)"); ax.set_title(title); ax.grid(alpha=.15, axis="x")


def figure_e(out: Path) -> None:
    directory = REPO / "experiments/phaseguard/raw/timelines"
    paths = list(directory.glob("*.json"))
    chosen = []
    for policy in ("uncoordinated", "phaseguard"):
        matches = [p for p in paths if p.name.startswith("eval_") and f"_{policy}_" in p.name
                   and "_c4_ctx2048_slo1.15_r0" in p.name]
        if matches: chosen.append((policy, json.loads(sorted(matches)[0].read_text())))
    if len(chosen) != 2: return
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), sharex=False)
    for ax, (policy, data) in zip(axes, chosen): timeline_panel(ax, data, policy)
    fig.suptitle("E. Representative application-level execution timelines")
    fig.legend(handles=[Patch(color="#4c78a8", label="CPU retrieval"),
                        Patch(color="#f2cf5b", label="GPU prefill"),
                        Patch(color="#e45756", label="GPU decode")],
               loc="upper center", bbox_to_anchor=(.5, .96), ncol=3, frameon=False)
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--profile", default="experiments/phaseguard/processed/profile_primary.csv")
    ap.add_argument("--summary", default="experiments/phaseguard/processed/summary.csv")
    args = ap.parse_args(); out = REPO / "experiments/phaseguard/plots"
    profile, summary = read_csv(REPO / args.profile), read_csv(REPO / args.summary)
    figure_a(profile, out / "figure_a_hnsw_phase_asymmetry")
    figure_b(summary, out / "figure_b_scheduling_pareto")
    figure_c(summary, out / "figure_c_policy_comparison")
    figure_d(out / "figure_d_bandwidth_proxy_failure")
    figure_e(out / "figure_e_execution_timeline")


if __name__ == "__main__": main()
