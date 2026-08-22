#!/usr/bin/env python3
"""Analyze clean/suspect context validation and generate Figures 1-6."""
from __future__ import annotations

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
ROOT = REPO / "experiments/phaseguard/context_validation"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text(""); return
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def rank(values: list[float]) -> np.ndarray:
    array = np.asarray(values, float); order = np.argsort(array); ranks = np.empty(len(array), float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]: end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2: return None
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def enrich_isolated(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [r for r in rows if int(r.get("attempt", 0)) >= 1 and r.get("status") == "ok"]
    baselines = {(int(r["attempt"]), int(r["context"]), r["phase"]): r for r in rows if r["workload"] == "none"}
    output = []
    for row in rows:
        baseline = baselines.get((int(row["attempt"]), int(row["context"]), row["phase"]))
        if baseline is None: continue
        item = dict(row)
        item["median_slowdown"] = float(row["median_ms"]) / float(baseline["median_ms"])
        item["p95_slowdown"] = float(row["p95_ms"]) / float(baseline["p95_ms"])
        item["p99_slowdown"] = float(row["p99_ms"]) / float(baseline["p99_ms"])
        item["baseline_clean"] = bool(baseline["clean"])
        item["pair_clean"] = bool(row["clean"] and baseline["clean"])
        # Whole-run medians can conceal the rapid power/thermal transition seen on
        # the fanless M4. Preserve every sample, but also expose an early window
        # fixed a priori and a strict stationary-pair flag.
        timings = [float(value) for value in row.get("timings_ms", [])]
        baseline_timings = [float(value) for value in baseline.get("timings_ms", [])]
        window = min(5, len(timings), len(baseline_timings))
        if window:
            early = timings[:window]
            early_base = baseline_timings[:window]
            item["early_window_iterations"] = window
            item["early_median_ms"] = float(np.median(early))
            item["early_p95_ms"] = float(np.percentile(early, 95))
            item["early_median_slowdown"] = item["early_median_ms"] / float(np.median(early_base))
            item["early_p95_slowdown"] = item["early_p95_ms"] / float(np.percentile(early_base, 95))
            item["late_median_ms"] = float(np.median(timings[-window:]))
            item["within_run_early_late_ratio"] = item["late_median_ms"] / item["early_median_ms"]
        item["stationary_pair"] = bool(
            item["pair_clean"]
            and float(row.get("thermal_drift_ratio", 99)) <= 1.10
            and float(baseline.get("thermal_drift_ratio", 99)) <= 1.10
        )
        output.append(item)
    return output


def summarize(rows: list[dict[str, Any]], predicate=lambda row: True) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if predicate(row): groups[(int(row["context"]), row["workload"], row["phase"])].append(row)
    output = []
    for (context, workload, phase), subset in sorted(groups.items()):
        def med(key: str) -> float: return float(np.median([float(r[key]) for r in subset]))
        output.append({"context": context, "workload": workload, "phase": phase,
                       "runs": len(subset), "median_latency_ms": med("median_ms"),
                       "p95_latency_ms": med("p95_ms"), "p99_latency_ms": med("p99_ms"),
                       "median_slowdown": med("median_slowdown"),
                       "p95_slowdown": med("p95_slowdown"), "p99_slowdown": med("p99_slowdown"),
                       "early_median_slowdown": med("early_median_slowdown"),
                       "early_p95_slowdown": med("early_p95_slowdown"),
                       "within_run_early_late_ratio": med("within_run_early_late_ratio"),
                       "median_cv": med("cv"), "median_iqr_ms": med("iqr_ms"),
                       "pageout_observed_runs": sum(bool(r["pageout_observed"]) for r in subset),
                       "thermal_suspect_runs": sum(bool(r["thermal_suspect"]) for r in subset)})
    return output


def memory_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("attempt", "order", "context", "workload", "phase", "median_slowdown", "p95_slowdown",
            "early_median_slowdown", "early_p95_slowdown", "within_run_early_late_ratio",
            "pair_clean", "stationary_pair", "pageouts_delta", "swap_used_delta_bytes", "compressions_delta",
            "pages_occupied_by_compressor_delta", "headroom_before_bytes", "headroom_after_bytes",
            "memory_free_percent_before", "memory_free_percent_after", "thermal_drift_ratio",
            "thermal_suspect", "workload_throughput")
    return [{key: row.get(key) for key in keys} for row in rows]


def pipeline_comparison(isolated: list[dict[str, Any]], pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Use the predeclared full-matrix attempt for HNSW and the explicitly short,
    # cooled attempt for random. Do not let a hot but internally flat later run
    # become the isolated representative merely because its within-run drift is low.
    preferred_attempt = {"hnsw4": 1, "random": 12, "none": 12}
    iso_map: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in isolated:
        if row["phase"] != "prefill":
            continue
        if int(row["attempt"]) == preferred_attempt.get(row["workload"], 1):
            iso_map[(int(row["context"]), row["workload"], row["phase"])] = row
    base: dict[int, dict[str, Any]] = {}
    for row in pipeline:
        if row.get("status") == "ok" and row.get("workload") == "none" and int(row.get("requests", 0)) == 12:
            base[int(row["context"])] = row
    result = []
    for row in pipeline:
        context, workload = int(row["context"]), row["workload"]
        if context not in base or int(row.get("requests", 0)) != int(base[context]["requests"]): continue
        baseline = base[context]
        iso_row = iso_map.get((context, workload, "prefill"))
        result.append({"context": context, "workload": workload,
                       "isolated_source_attempt": iso_row["attempt"] if iso_row else None,
                       "isolated_prefill_slowdown": iso_row["median_slowdown"] if iso_row else None,
                       "isolated_prefill_p95_slowdown": iso_row["p95_slowdown"] if iso_row else None,
                       "isolated_pair_clean": iso_row["pair_clean"] if iso_row else None,
                       "pipeline_pure_prefill_slowdown": float(row["prefill_median_ms"]) / float(baseline["prefill_median_ms"]),
                       "pipeline_queue_inclusive_slowdown": float(row["queue_inclusive_median_ms"]) / float(baseline["queue_inclusive_median_ms"]),
                       "pipeline_pure_prefill_p95_slowdown": float(row["prefill_p95_ms"]) / float(baseline["prefill_p95_ms"]),
                       "pipeline_tpot_p95_slowdown": float(row["tpot_p95_ms"]) / float(baseline["tpot_p95_ms"]),
                       "gpu_queue_median_ms": row["gpu_queue_median_ms"], "pipeline_clean": row["clean"],
                       "pipeline_thermal_drift_ratio": row["thermal_drift_ratio"],
                       "pipeline_pageouts_delta": row["pageouts_delta"]})
    workload_order = {"none": 0, "hnsw4": 1, "random": 2}
    return sorted(result, key=lambda row: (int(row["context"]), workload_order.get(row["workload"], 9)))


def save(fig: plt.Figure, stem: str) -> None:
    path = ROOT / "plots" / stem; path.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def plots(rows: list[dict[str, Any]], clean_summary: list[dict[str, Any]], comparison: list[dict[str, Any]]) -> None:
    # Figure 1
    fig, ax = plt.subplots(figsize=(7.2, 4.8)); contexts = [2048, 4096]
    for phase, metric, style in (("prefill", "median_slowdown", "o-"), ("prefill", "p95_slowdown", "o--"),
                                 ("decode", "median_slowdown", "s-"), ("decode", "p95_slowdown", "s--")):
        values = []
        for context in contexts:
            match = next((r for r in rows if int(r["attempt"]) == 1 and int(r["context"]) == context
                          and r["workload"] == "hnsw4" and r["phase"] == phase), None)
            values.append(match[metric] if match else np.nan)
        ax.plot(contexts, values, style, lw=2, label=f"{phase} {metric.split('_')[0]}")
    ax.axhline(1, color="black", lw=.8); ax.set(xticks=contexts, xlabel="context length (tokens)",
        ylabel="normalized slowdown", title="1. Context comparison — HNSW-4 primary full matrix")
    prefill_4096 = next(r["median_slowdown"] for r in rows if int(r["attempt"]) == 1
                        and int(r["context"]) == 4096 and r["workload"] == "hnsw4" and r["phase"] == "prefill")
    ax.annotate("4096 prefill drift-suspect", xy=(4096, prefill_4096), xytext=(3300, 1.23),
                arrowprops={"arrowstyle": "->", "color": "#666"}, fontsize=8, color="#666")
    ax.set_ylim(.95, 1.58); ax.grid(alpha=.25); ax.legend(frameon=False, loc="upper left")
    save(fig, "figure_1_context_comparison")

    # Figure 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    workloads = ["none", "hnsw1", "hnsw4", "random"]; width=.18; x=np.arange(2)
    for ax, phase in zip(axes, ("prefill", "decode")):
        for i, workload in enumerate(workloads):
            vals=[]
            for context in contexts:
                attempt = 12 if workload == "random" and phase == "prefill" else 1
                match=next((r for r in rows if int(r["attempt"])==attempt and int(r["context"])==context
                            and r["workload"]==workload and r["phase"]==phase),None)
                vals.append(match["median_slowdown"] if match else (1.0 if workload == "none" else np.nan))
            ax.bar(x+(i-1.5)*width,vals,width,label=workload)
        ax.set_xticks(x,contexts); ax.set_xlabel("context"); ax.set_title(phase); ax.grid(alpha=.2,axis="y")
    axes[0].set_ylabel("median slowdown"); axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("2. Workload comparison — primary HNSW and short stationary random control"); save(fig,"figure_2_workload_comparison")

    # Figure 3
    prefill=[r for r in rows if r["phase"]=="prefill" and r["workload"] in ("none","hnsw4","random")]
    groups=[]; labels=[]; colors=[]
    for context in contexts:
        for workload in ("none","hnsw4","random"):
            for clean in (True,False):
                vals=[v for r in prefill if int(r["context"])==context and r["workload"]==workload and bool(r["pair_clean"])==clean for v in r["timings_ms"]]
                if vals: groups.append(vals); labels.append(f"{context}\n{workload}\n{'clean' if clean else 'suspect'}"); colors.append("#4c78a8" if clean else "#e45756")
    fig,ax=plt.subplots(figsize=(12,5)); boxes=ax.boxplot(groups,patch_artist=True,showfliers=True)
    for patch,color in zip(boxes["boxes"],colors): patch.set_facecolor(color); patch.set_alpha(.65)
    ax.set_xticks(range(1,len(labels)+1),labels,rotation=35,ha="right",fontsize=7)
    ax.set_ylabel("prefill latency (ms)"); ax.set_title("3. Run-level prefill distributions"); ax.grid(alpha=.2,axis="y")
    save(fig,"figure_3_prefill_distribution")

    # Figure 4
    fig,ax=plt.subplots(figsize=(8,4.8))
    if comparison:
        labels=[f"{r['context']}\n{r['workload']}" for r in comparison]; xx=np.arange(len(labels)); w=.26
        for i,(key,name) in enumerate((("isolated_prefill_slowdown","isolated pure"),("pipeline_pure_prefill_slowdown","pipeline pure"),("pipeline_queue_inclusive_slowdown","queue-inclusive"))):
            ax.bar(xx+(i-1)*w,[r[key] if r[key] is not None else np.nan for r in comparison],w,label=name)
        ax.set_xticks(xx,labels)
    ax.axhline(1,color="black",lw=.8); ax.set_ylabel("median slowdown"); ax.set_title("4. Isolated versus full pipeline")
    ax.legend(frameon=False); ax.grid(alpha=.2,axis="y"); save(fig,"figure_4_isolated_vs_pipeline")

    # Figure 5
    fig,ax=plt.subplots(figsize=(7,4.8))
    for clean,color,marker in ((True,"#4c78a8","o"),(False,"#e45756","x")):
        subset=[r for r in rows if r["phase"]=="prefill" and bool(r["pair_clean"])==clean]
        ax.scatter([r["pageouts_delta"] for r in subset],[r["median_slowdown"] for r in subset],
                   color=color,marker=marker,label="clean pair" if clean else "suspect pair")
    ax.set(xlabel="experiment-window pageout delta (16 KiB pages)",ylabel="prefill median slowdown",
           title="5. Memory-state diagnostic"); ax.grid(alpha=.25); ax.legend(frameon=False)
    save(fig,"figure_5_memory_diagnostic")

    # Optional Figure 6
    fig,ax=plt.subplots(figsize=(7,4.8)); subset=sorted(
        [r for r in rows if r["phase"]=="prefill" and r["workload"]!="none"], key=lambda r: float(r["start_time"]))
    global_order={id(row): index + 1 for index, row in enumerate(subset)}
    for context,color in ((2048,"#4c78a8"),(4096,"#f58518")):
        ss=[r for r in subset if int(r["context"])==context]
        ax.scatter([global_order[id(r)] for r in ss],[r["median_slowdown"] for r in ss],label=str(context),color=color)
    ax.set(xlabel="chronological loaded-prefill point",ylabel="prefill median slowdown",
           title="6. Run-order diagnostic"); ax.grid(alpha=.25); ax.legend(title="context",frameon=False)
    save(fig,"figure_6_run_order")


def main() -> None:
    rows=enrich_isolated(load_jsonl(ROOT/"raw/isolated_runs.jsonl"))
    all_summary=summarize(rows); clean_summary=summarize(rows,lambda r:bool(r["pair_clean"]))
    stationary_summary=summarize(rows,lambda r:bool(r["stationary_pair"]))
    suspect_summary=summarize(rows,lambda r:not bool(r["pair_clean"]))
    write_csv(ROOT/"processed/summary.csv",all_summary); write_csv(ROOT/"processed/clean_only.csv",clean_summary)
    write_csv(ROOT/"processed/stationary_clean_only.csv",stationary_summary)
    write_csv(ROOT/"processed/contaminated_only.csv",suspect_summary)
    diagnostics=memory_rows(rows); write_csv(ROOT/"processed/memory_state_diagnostic.csv",diagnostics)
    pipeline=load_jsonl(ROOT/"raw/pipeline_runs.jsonl"); comparison=pipeline_comparison(rows,pipeline)
    write_csv(ROOT/"processed/isolated_vs_pipeline.csv",comparison)
    prefill=[r for r in rows if r["phase"]=="prefill" and r["workload"]!="none"]
    correlations={key:spearman([float(r[key]) for r in prefill],[float(r["median_slowdown"]) for r in prefill])
                  for key in ("order","pageouts_delta","swap_used_delta_bytes","compressions_delta",
                              "headroom_before_bytes","thermal_drift_ratio")}
    (ROOT/"processed/correlations.json").write_text(json.dumps(correlations,indent=2,sort_keys=True)+"\n")
    plots(rows,stationary_summary,comparison)
    print(f"[out] {len(rows)} normalized isolated rows; {len(stationary_summary)} stationary clean groups; correlations={correlations}")


if __name__=="__main__": main()
