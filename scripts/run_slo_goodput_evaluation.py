#!/usr/bin/env python3
"""Revalidate the baseline and run frozen held-out SLO-goodput evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from fanmac_main_common import REPO, ensure_block
from slo_goodput_common import (add_args, append_command, assert_environment, configure_root,
                                CAMPAIGN_SEEDS, item_by_name, namespace, resolve)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def revalidate(pilot, user, campaign: Path, baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text())
    args = namespace(pilot, user, "baseline_revalidation",
                     CAMPAIGN_SEEDS["revalidation_prompt_trace"], 200)
    rows = [ensure_block(pilot, args, ("llm-only", "llm-only", 0, 0), repeat)
            for repeat in range(5)]
    tpot = float(np.median([float(row["p95_tpot_ms"]) for row in rows]))
    ttft = float(np.median([float(row["p95_ttft_ms"]) for row in rows]))
    p99_itg = float(np.median([float(row["p99_inter_token_gap_ms"]) for row in rows]))
    transition = float(np.median([float(row["p95_transition_gap_ms"]) for row in rows]))
    audit = {"calibration_baseline_tpot_ms": baseline["p95_tpot_ms"],
             "pre_evaluation_baseline_tpot_ms": tpot,
             "tpot_drift": tpot / float(baseline["p95_tpot_ms"]) - 1,
             "calibration_baseline_ttft_ms": baseline["p95_ttft_ms"],
             "pre_evaluation_baseline_ttft_ms": ttft,
             "ttft_drift": ttft / float(baseline["p95_ttft_ms"]) - 1,
             "p99_itg_drift": p99_itg / float(baseline["p99_inter_token_gap_ms"]) - 1,
             "transition_p95_drift": transition / float(baseline["p95_transition_gap_ms"]) - 1,
             "valid_blocks": 5, "requests_per_block": 200,
             "pageout_swap_clean": all(int(row["pageouts_delta"]) == 0 and
                                        int(row["swap_used_delta_bytes"] or 0) == 0 for row in rows)}
    audit["decision"] = ("reuse calibration baseline and proceed" if
        abs(audit["tpot_drift"]) <= .03 and abs(audit["ttft_drift"]) <= .03 and
        audit["pageout_swap_clean"] else "stop and rerun calibration")
    write_csv(campaign / "baseline_drift_audit.csv", [audit])
    if audit["decision"] != "reuse calibration baseline and proceed":
        raise RuntimeError("baseline revalidation failed; held-out evaluation is prohibited")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); add_args(ap)
    ap.add_argument("--repeats", type=int, choices=(5, 7), default=7)
    user = ap.parse_args(); campaign = resolve(user.campaign_dir)
    assert_environment(); pilot = configure_root(campaign); append_command(campaign, sys.argv)
    selection_path = campaign / "frozen_slo_selection.json"
    if not selection_path.exists(): raise SystemExit(f"missing {selection_path}")
    selection_bytes = selection_path.read_bytes(); selected = json.loads(selection_bytes)
    baseline_path = campaign / "isolated_baseline/baseline.json"
    revalidate(pilot, user, campaign, baseline_path)
    names = {"llm-only", "fixed0", "fixed4", "fixed1", "phasegate4to1",
             "fixed2", "phasegate4to2"}
    for row in selected["selection_by_slo"]:
        for key in ("continuous_fixed", "continuous_phasegate", "decode_zero_phasegate"):
            if row[key] is not None: names.add(row[key]["policy"])
    items = [("llm-only", "llm-only", 0, 0)] + [item_by_name(name) for name in sorted(names)
                                                   if name != "llm-only"]
    args = namespace(pilot, user, "evaluation", int(selected["evaluation_seed_base"]),
                     250, baseline_path)
    orders = []
    for repeat in range(user.repeats):
        order = list(items)
        random.Random(int(selected["evaluation_policy_order_seed_base"]) + repeat).shuffle(order)
        orders.append([item[0] for item in order])
    manifest = {"started_from_frozen_selection": True,
        "started": datetime.now().astimezone().isoformat(), "held_out_seed_base": args.seed,
        "query_seed_offset": 5000,
        "policy_order_seed_base": int(selected["evaluation_policy_order_seed_base"]),
        "repeats": user.repeats, "requests_per_block": 250, "tokens_per_request": 128,
        "selection_file": str(selection_path),
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "baseline_file": str(baseline_path), "policies": [item[0] for item in items],
        "orders": orders, "traces_predeclared": True}
    (campaign / "evaluation/logs/matrix.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for repeat, order_names in enumerate(orders):
        for name in order_names:
            item = (("llm-only", "llm-only", 0, 0) if name == "llm-only"
                    else item_by_name(name))
            ensure_block(pilot, args, item, repeat)
        rows = pilot.read_jsonl(pilot.raw_path("evaluation", False))
        completed = sorted({int(row["repeat"]) for row in rows if row.get("status") == "valid"})
        (campaign / "evaluation/progress.json").write_text(json.dumps({
            "completed_paired_repeats": completed,
            "target_paired_repeats": user.repeats,
            "policies_per_repeat": len(items),
            "updated": datetime.now().astimezone().isoformat(),
            "resume_source": "evaluation/raw/runs.jsonl",
        }, indent=2) + "\n")
    manifest["finished"] = datetime.now().astimezone().isoformat()
    (campaign / "evaluation/logs/matrix.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
