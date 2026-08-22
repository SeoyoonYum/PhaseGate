#!/usr/bin/env python3
"""Aggregate PhaseGuard JSONL while retaining contamination and failures."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def label(row: dict[str, object]) -> str:
    if row["policy"] == "static_phase": return f"static-{row['decode_workers']}"
    return str(row["policy"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="experiments/phaseguard/raw/runs.jsonl")
    ap.add_argument("--output", default="experiments/phaseguard/processed/summary.csv")
    ap.add_argument("--tag", default=None, help="only aggregate run keys beginning with this tag")
    args = ap.parse_args()
    source, output = REPO / args.runs, REPO / args.output
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    good = [r for r in rows if r.get("status") == "ok" and
            (args.tag is None or str(r.get("run_key", "")).startswith(args.tag + "_"))]
    groups: dict[tuple[str, int, int, float], list[dict[str, object]]] = defaultdict(list)
    for row in good:
        groups[(label(row), int(row["concurrency"]), int(row["context"]),
                float(row["slo_multiplier"]))].append(row)
    metrics = ["request_throughput_s", "retrieval_qps", "p95_request_ms", "p95_tpot_ms",
               "token_slo_violation_rate", "request_slo_violation_rate", "scheduler_overhead_ms"]
    result: list[dict[str, object]] = []
    for (policy, concurrency, context, slo), subset in sorted(groups.items()):
        out: dict[str, object] = {"policy": policy, "concurrency": concurrency,
                                  "context": context, "slo_multiplier": slo,
                                  "runs": len(subset), "contaminated_runs": sum(bool(r["contaminated"]) for r in subset)}
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in subset])
            out[metric] = float(np.median(values))
            out[metric + "_min"] = float(values.min())
            out[metric + "_max"] = float(values.max())
        result.append(out)
    # Evaluation-only oracle: best observed throughput among policies satisfying the run SLO.
    for key in sorted({(int(r["concurrency"]), int(r["context"]), float(r["slo_multiplier"])) for r in good}):
        c, ctx, slo = key
        candidates = [r for r in good if int(r["concurrency"]) == c and int(r["context"]) == ctx and
                      float(r["slo_multiplier"]) == slo]
        safe = [r for r in candidates if float(r["p95_tpot_ms"]) <= float(r["slo_ms"])]
        chosen = max(safe or candidates, key=lambda r: float(r["request_throughput_s"]))
        out = {"policy": "oracle-best-observed", "concurrency": c, "context": ctx,
               "slo_multiplier": slo, "runs": len(candidates),
               "contaminated_runs": sum(bool(r["contaminated"]) for r in candidates)}
        for metric in metrics:
            out[metric] = out[metric + "_min"] = out[metric + "_max"] = float(chosen[metric])
        result.append(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(result[0])
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(result)
    print(f"[out] {output} ({len(result)} aggregate rows; {len(good)}/{len(rows)} successful raw runs)")


if __name__ == "__main__": main()
