#!/usr/bin/env python3
"""Analyze output-length dependence using event-derived phase fractions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def phase_times(timeline: Path) -> dict[str, float]:
    events = json.loads(timeline.read_text())["events"]
    by_request: dict[str, dict[str, float]] = {}
    for event in events:
        rid = event.get("request_id")
        if rid is None: continue
        item = by_request.setdefault(str(rid), {})
        kind = event["event_type"]
        if kind == "request_start": item["start"] = float(event["timestamp"])
        elif kind == "prefill_complete": item["prefill_end"] = float(event["timestamp"])
        elif kind == "request_complete": item["end"] = float(event["timestamp"])
    fractions = []; prefill = decode = 0.0
    for item in by_request.values():
        if not all(key in item for key in ("start", "prefill_end", "end")): continue
        p = max(0.0, item["prefill_end"] - item["start"])
        d = max(0.0, item["end"] - item["prefill_end"]); prefill += p; decode += d
        if p + d: fractions.append(p / (p + d))
    return {"aggregate_prefill_wall_s": prefill, "aggregate_decode_wall_s": decode,
            "aggregate_prefill_fraction": prefill / (prefill + decode),
            "median_request_prefill_fraction": median(fractions)}


def bootstrap(values: list[float], seed: int) -> tuple[float, float, float]:
    a=np.asarray(values); rng=np.random.default_rng(seed)
    b=np.median(rng.choice(a,size=(10000,len(a)),replace=True),axis=1)
    return float(np.median(a)),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("campaign",type=Path)
    args=parser.parse_args(); root=args.campaign.resolve(); run_rows=[]; fraction_rows=[]; pair_rows=[]
    for length in (64,128,512):
        stage=f"m4_output_{length}"; rows=[r for r in read_jsonl(root/f"{stage}/raw/runs.jsonl") if r["status"]=="valid"]
        if len(rows)!=10: raise RuntimeError(f"{stage}: expected 10 valid rows, got {len(rows)}")
        for row in rows:
            times=phase_times(root/stage/"raw/timelines"/f"{row['run_key']}.json")
            output={"output_tokens":length,"policy":row["policy"],"repeat":row["repeat"],
                "run_key":row["run_key"],"retrieval_qps":row["total_retrieval_goodput_qps"],
                "normalized_p95_tpot":row["normalized_p95_tpot"],"normalized_p95_ttft":row["normalized_p95_ttft"],
                "joint_slo_1_25":float(row["normalized_p95_tpot"])<=1.25 and float(row["normalized_p95_ttft"])<=1.25,
                "completed_queries_prefill":row["completed_queries_prefill"],
                "completed_queries_decode":row["completed_queries_decode"],"pageouts_delta":row["pageouts_delta"],
                "swap_used_delta_bytes":row["swap_used_delta_bytes"],**times}
            run_rows.append(output); fraction_rows.append({k:output[k] for k in
                ("output_tokens","policy","repeat","aggregate_prefill_wall_s","aggregate_decode_wall_s",
                 "aggregate_prefill_fraction","median_request_prefill_fraction",
                 "completed_queries_prefill","completed_queries_decode")})
        by={(int(r["repeat"]),r["policy"]):r for r in run_rows if r["output_tokens"]==length}
        for repeat in range(5):
            fixed,phase=by[(repeat,"fixed1")],by[(repeat,"phasegate4to1")]
            pair_rows.append({"output_tokens":length,"repeat":repeat,"fixed_qps":fixed["retrieval_qps"],
                "phasegate_qps":phase["retrieval_qps"],"paired_qps_gain":float(phase["retrieval_qps"])/float(fixed["retrieval_qps"])-1,
                "fixed_prefill_fraction":fixed["aggregate_prefill_fraction"],
                "phasegate_prefill_fraction":phase["aggregate_prefill_fraction"],
                "fixed_joint_slo_1_25":fixed["joint_slo_1_25"],"phasegate_joint_slo_1_25":phase["joint_slo_1_25"]})
    write_csv(root/"m4_output_length_runs.csv",run_rows); write_csv(root/"m4_phase_fraction_summary.csv",fraction_rows)
    summaries=[]
    for length in (64,128,512):
        pairs=[r for r in pair_rows if r["output_tokens"]==length]; ci=bootstrap([r["paired_qps_gain"] for r in pairs],2026080600+length)
        for policy in ("fixed1","phasegate4to1"):
            group=[r for r in run_rows if r["output_tokens"]==length and r["policy"]==policy]
            summaries.append({"output_tokens":length,"policy":policy,"valid_repeats":len(group),
                "median_retrieval_qps":median(r["retrieval_qps"] for r in group),
                "median_normalized_p95_tpot":median(r["normalized_p95_tpot"] for r in group),
                "median_normalized_p95_ttft":median(r["normalized_p95_ttft"] for r in group),
                "joint_slo_pass_count":sum(r["joint_slo_1_25"] for r in group),
                "median_aggregate_prefill_fraction":median(r["aggregate_prefill_fraction"] for r in group),
                "paired_phasegate_gain_median":ci[0],"paired_gain_ci_low":ci[1],"paired_gain_ci_high":ci[2]})
    write_csv(root/"m4_output_length_summary.csv",summaries); write_csv(root/"m4_output_length_pairs.csv",pair_rows)
    lengths=[64,128,512]; gains=[next(r for r in summaries if r["output_tokens"]==x)["paired_phasegate_gain_median"]*100 for x in lengths]
    fig,ax=plt.subplots(figsize=(5.2,3.7));ax.plot(lengths,gains,"o-");ax.set(xlabel="Output tokens",ylabel="Median paired QPS gain (%)")
    fig.tight_layout();fig.savefig(root/"figure_m4_gain_vs_output_length.pdf");fig.savefig(root/"figure_m4_gain_vs_output_length.png",dpi=180);plt.close(fig)
    fractions=[median(r["phasegate_prefill_fraction"] for r in pair_rows if r["output_tokens"]==x) for x in lengths]
    fig,ax=plt.subplots(figsize=(5.2,3.7));ax.plot(np.asarray(fractions)*100,gains,"o-");ax.set(xlabel="Measured PREFILL wall-time fraction (%)",ylabel="Median paired QPS gain (%)")
    fig.tight_layout();fig.savefig(root/"figure_m4_gain_vs_prefill_fraction.pdf");fig.savefig(root/"figure_m4_gain_vs_prefill_fraction.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.4,3.8))
    for policy,marker in (("fixed1","o"),("phasegate4to1","s")):
        ax.plot(lengths,[next(r for r in summaries if r["output_tokens"]==x and r["policy"]==policy)["median_normalized_p95_tpot"] for x in lengths],marker+"-",label=policy)
    ax.axhline(1.25,color="black",linestyle="--");ax.set(xlabel="Output tokens",ylabel="Normalized p95 TPOT");ax.legend()
    fig.tight_layout();fig.savefig(root/"figure_m4_slo_vs_output_length.pdf");fig.savefig(root/"figure_m4_slo_vs_output_length.png",dpi=180);plt.close(fig)
    monotonic=all(gains[i]>=gains[i+1] for i in range(len(gains)-1))
    report="# Base-M4 Output-Shape Report\n\n"+"\n".join(
        f"- {length} tokens: median paired QPS gain {gain:.2f}%, measured PREFILL fraction {fraction*100:.2f}%"
        for length,gain,fraction in zip(lengths,gains,fractions))+f"\n\nMonotonic gain reduction: {monotonic}. No extrapolation beyond measured lengths is made.\n"
    (root/"M4_OUTPUT_SHAPE_REPORT.md").write_text(report);print(report)


if __name__=="__main__":main()
