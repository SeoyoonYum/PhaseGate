#!/usr/bin/env python3
"""Select held-out fixed-k and summarize PhaseGate backlog experiments."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/phaseguard/backlog_validation"


def load() -> list[dict[str, Any]]:
    path = ROOT / "raw/runs.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("status") == "ok"
            and bool(row.get("fan_capable")) and not bool(row.get("smoke"))]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def med(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def select_fixed(rows: list[dict[str, Any]]) -> dict[str, int]:
    chosen: dict[str, int] = {}
    diagnostics: list[dict[str, Any]] = []
    for demand in ("low", "medium", "high"):
        candidates = [row for row in rows if row["split"] == "calibration"
                      and row["requested_demand"] == demand and row["policy_arg"] == "fixed"]
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            groups[int(row["fixed_workers"])].append(row)
        if not groups:
            continue
        scored = []
        for workers, subset in groups.items():
            score = {"demand": demand, "fixed_workers": workers, "runs": len(subset),
                     "tpot_p95_ms": med(subset, "tpot_p95_ms"),
                     "steady_tpot_p95_ms": med(subset, "steady_tpot_p95_ms"),
                     "slo_ms": med(subset, "slo_ms"),
                     "application_goodput_rps": med(subset, "application_goodput_rps"),
                     "retrieval_goodput_qps": med(subset, "retrieval_goodput_qps")}
            score["slo_safe"] = score["steady_tpot_p95_ms"] <= score["slo_ms"]
            diagnostics.append(score)
            scored.append(score)
        safe = [score for score in scored if score["slo_safe"]]
        winner = max(safe or scored, key=lambda score: (score["application_goodput_rps"],
                                                        -score["steady_tpot_p95_ms"]))
        chosen[demand] = int(winner["fixed_workers"])
    write_csv(ROOT / "processed/fixed_k_calibration.csv", diagnostics)
    if chosen:
        (ROOT / "processed").mkdir(parents=True, exist_ok=True)
        (ROOT / "processed/best_fixed_k.json").write_text(json.dumps(chosen, indent=2) + "\n")
    return chosen


def select_tpot_budget_cap(rows: list[dict[str, Any]], target_tpot_ms: float) -> int | None:
    """Choose the largest decode cap meeting the target on calibration blocks."""
    candidates = [row for row in rows if row["split"] == "calibration"
                  and row["requested_demand"] == "high"
                  and row["policy_arg"] == "static"]
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[int(row["decode_workers"])].append(row)
    diagnostics: list[dict[str, Any]] = []
    for cap, subset in sorted(groups.items()):
        measured = med(subset, "steady_tpot_p95_ms")
        diagnostics.append({"decode_cap": cap, "runs": len(subset),
                            "target_tpot_ms": target_tpot_ms,
                            "steady_tpot_p95_ms": measured,
                            "decode_retrieval_progress_qps": med(
                                subset, "steady_decode_retrieval_progress_qps"),
                            "target_met": measured <= target_tpot_ms})
    write_csv(ROOT / "processed/tpot_budget_calibration.csv", diagnostics)
    safe = [row for row in diagnostics if row["target_met"]]
    if not safe:
        return None
    chosen = max(int(row["decode_cap"]) for row in safe)
    payload = {"target_tpot_ms": target_tpot_ms, "decode_cap": chosen,
               "selection": "largest calibration cap with median steady p95 TPOT <= target"}
    (ROOT / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "processed/tpot_budget_cap.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    return chosen


def demand_check(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for demand in ("low", "medium", "high"):
        subset = [row for row in rows if row["split"] == "calibration"
                  and row["requested_demand"] == demand and row["policy_arg"] == "uncoordinated"]
        if not subset:
            continue
        active = med(subset, "steady_decode_active_retrieval_mean")
        queue = med(subset, "steady_decode_queue_depth_mean")
        if active <= 1.0:
            observed = "low"
        elif active < 2.5:
            observed = "medium"
        else:
            observed = "high"
        output.append({"requested_demand": demand, "observed_band": observed,
                       "runs": len(subset), "decode_active_retrieval_mean": active,
                       "decode_active_retrieval_p95": med(subset, "steady_decode_active_retrieval_p95"),
                       "decode_overlap_fraction": med(subset, "steady_decode_retrieval_overlap_fraction"),
                       "decode_queue_depth_mean": queue,
                       "target_met": observed == demand or (demand == "high" and queue >= 1.0)})
    return output


def evaluation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] != "evaluation":
            continue
        label = f"fixed-{row['fixed_workers']}" if row["policy_arg"] == "fixed" else row["policy_arg"]
        groups[(row["requested_demand"], label)].append(row)
    keys = ("application_goodput_rps", "retrieval_goodput_qps", "tpot_p50_ms", "tpot_p95_ms",
            "token_slo_violation_rate", "ttft_application_p95_ms", "end_to_end_p95_ms",
            "decode_active_retrieval_mean", "decode_retrieval_overlap_fraction",
            "decode_scheduler_binding_fraction", "decode_queue_depth_mean",
            "steady_decode_active_retrieval_mean", "steady_decode_retrieval_overlap_fraction",
            "steady_decode_scheduler_binding_fraction", "steady_decode_queue_depth_mean",
            "steady_decode_paused_inflight_mean", "steady_decode_effective_backlog_mean",
            "steady_tpot_p95_ms", "steady_token_slo_violation_rate")
    output = []
    for (demand, policy), subset in sorted(groups.items()):
        row: dict[str, Any] = {"requested_demand": demand, "policy": policy,
                               "runs": len(subset),
                               "pageout_observed_runs": sum(int(r["pageouts_delta"]) > 0 for r in subset)}
        for key in keys:
            row[key] = med(subset, key)
            row[key + "_min"] = min(float(r[key]) for r in subset)
            row[key + "_max"] = max(float(r[key]) for r in subset)
        output.append(row)
    return output


def tpot_budget_smoke(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize fanless functional blocks that include decode CPU progress."""
    selected = [row for row in all_rows if row.get("smoke")
                and row.get("split") == "evaluation"
                and row.get("requested_demand") == "high"
                and "steady_decode_retrieval_progress_qps" in row
                and (row.get("policy_arg") == "uncoordinated"
                     or row.get("target_tpot_ms") in (16.0558, 19.0))]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        if row["policy_arg"] == "uncoordinated":
            label = "uncoordinated"
        else:
            label = (f"phaseguard_target_{float(row['target_tpot_ms']):g}ms_"
                     f"cap_{int(row['profiled_decode_worker_cap'])}")
        groups[label].append(row)
    metrics = ("steady_tpot_p95_ms", "steady_decode_retrieval_progress_qps",
               "retrieval_goodput_qps", "application_goodput_rps",
               "steady_decode_active_retrieval_mean",
               "steady_decode_scheduler_binding_fraction",
               "steady_decode_retrieval_overlap_fraction")
    output: list[dict[str, Any]] = []
    for label, subset in sorted(groups.items()):
        row: dict[str, Any] = {"policy": label, "runs": len(subset),
                               "tpot_19ms_pass_runs": sum(
                                   float(item["steady_tpot_p95_ms"]) <= 19.0
                                   for item in subset)}
        for metric in metrics:
            row[metric] = med(subset, metric)
            row[metric + "_min"] = min(float(item[metric]) for item in subset)
            row[metric + "_max"] = max(float(item[metric]) for item in subset)
        output.append(row)
    return output


def plots(demand: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    (ROOT / "plots").mkdir(parents=True, exist_ok=True)
    if demand:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        x = np.arange(len(demand))
        ax.bar(x, [row["decode_active_retrieval_mean"] for row in demand], color="#4c78a8")
        ax.set_xticks(x, [row["requested_demand"] for row in demand])
        ax.set(ylabel="measured active retrievals during decode",
               title="Demand calibration under uncoordinated execution")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(ROOT / "plots/demand_calibration.png", dpi=220)
        fig.savefig(ROOT / "plots/demand_calibration.pdf")
        plt.close(fig)
    if summary:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
        for ax, level in zip(axes, ("low", "medium", "high")):
            subset = [row for row in summary if row["requested_demand"] == level]
            for row in subset:
                ax.scatter(row["application_goodput_rps"], row["tpot_p95_ms"], s=65,
                           label=row["policy"])
            ax.set(title=level, xlabel="application goodput (requests/s)")
            ax.grid(alpha=.25)
        axes[0].set_ylabel("p95 TPOT (ms)")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .88))
        fig.savefig(ROOT / "plots/policy_pareto_by_demand.png", dpi=220)
        fig.savefig(ROOT / "plots/policy_pareto_by_demand.pdf")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tpot-ms", type=float, default=19.0)
    args = ap.parse_args()
    if args.target_tpot_ms <= 0:
        raise SystemExit("target TPOT must be positive")
    raw_path = ROOT / "raw/runs.jsonl"
    all_rows = ([json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
                if raw_path.exists() else [])
    smoke_keys = ("run_key", "requested_demand", "policy_arg", "fixed_workers",
                  "target_tpot_ms", "profiled_decode_worker_cap",
                  "application_goodput_rps", "retrieval_goodput_qps", "tpot_p95_ms",
                  "steady_tpot_p95_ms", "decode_active_retrieval_mean",
                  "steady_decode_active_retrieval_mean", "steady_decode_retrieval_overlap_fraction",
                  "steady_decode_retrieval_progress_qps",
                  "steady_decode_scheduler_binding_fraction", "steady_decode_queue_depth_mean",
                  "steady_decode_paused_inflight_mean", "steady_decode_effective_backlog_mean")
    smoke_rows = [{key: row.get(key) for key in smoke_keys} for row in all_rows
                  if not bool(row.get("fan_capable")) and "steady_tpot_p95_ms" in row]
    write_csv(ROOT / "processed/fanless_smoke_summary.csv", smoke_rows)
    budget_smoke = tpot_budget_smoke(all_rows)
    write_csv(ROOT / "processed/tpot_budget_smoke_summary.csv", budget_smoke)
    rows = load()
    chosen = select_fixed(rows)
    target_cap = select_tpot_budget_cap(rows, args.target_tpot_ms)
    demand = demand_check(rows)
    summary = evaluation_summary(rows)
    write_csv(ROOT / "processed/demand_calibration.csv", demand)
    write_csv(ROOT / "processed/evaluation_summary.csv", summary)
    plots(demand, summary)
    print(json.dumps({"runs": len(rows), "best_fixed_k": chosen,
                      "tpot_budget_decode_cap": target_cap,
                      "demand_targets": demand, "evaluation_groups": len(summary),
                      "tpot_budget_smoke_groups": len(budget_smoke)}, indent=2))


if __name__ == "__main__":
    main()
