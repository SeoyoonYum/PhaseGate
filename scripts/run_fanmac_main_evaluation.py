#!/usr/bin/env python3
"""Run the frozen held-out evaluation for the fan-cooled Mac campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime

from fanmac_main_common import (add_common_args, assert_environment, base_namespace,
    configure_root, ensure_block, resolve_campaign)


def item_from_summary(row: dict) -> tuple[str, str, int, int]:
    policy = "fixed0" if row["policy"] == "fixed0" else row["policy_arg"]
    return row["policy"], policy, int(row["prefill_cap"]), int(row["decode_cap"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--repeats", type=int, choices=(5, 6, 7), default=5)
    user = ap.parse_args()
    campaign = resolve_campaign(user.campaign_dir)
    assert_environment()
    pilot = configure_root(campaign)
    selection_path = campaign / "frozen_selection.json"
    if not selection_path.exists():
        raise SystemExit(f"missing frozen calibration selection: {selection_path}")
    selection_bytes = selection_path.read_bytes()
    selected = json.loads(selection_bytes)
    baseline = selection_path.parent / "isolated_baseline/baseline.json"
    if str(baseline) != selected["baseline_file"] or not baseline.exists():
        raise RuntimeError("frozen isolated baseline path mismatch")

    items = [
        ("llm-only", "llm-only", 0, 0),
        ("fixed4", "fixed", 4, 4),
    ]
    if selected["best_fixed"] is not None:
        items.append(item_from_summary(selected["best_fixed"]))
    if selected["best_phasegate"] is not None:
        items.append(item_from_summary(selected["best_phasegate"]))
    by_name = {row["policy"]: row for row in selected["calibration_summary"]}
    for pair in selected["latency_matched_pairs"]:
        if pair["passes_calibration_condition"]:
            items.extend((item_from_summary(by_name[pair["fixed"]]),
                          item_from_summary(by_name[pair["phasegate"]])))
    items = list(dict.fromkeys(items))

    seed = int(selected["evaluation_seed"])
    args = base_namespace(pilot, user, "evaluation", seed, baseline)
    orders = []
    for repeat in range(user.repeats):
        order = list(items)
        random.Random(seed + repeat).shuffle(order)
        orders.append([item[0] for item in order])
        for item in order:
            ensure_block(pilot, args, item, repeat)
    manifest = {
        "started_from_frozen_selection": True,
        "finished": datetime.now().isoformat(), "held_out_seed": seed,
        "repeats": user.repeats, "selection_file": str(selection_path),
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "baseline_file": str(baseline), "policies": [item[0] for item in items],
        "orders": orders,
    }
    (campaign / "evaluation/logs/matrix.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (campaign / "commands.log").open("a") as handle:
        handle.write(f"{datetime.now().isoformat()} " + " ".join(sys.argv) + "\n")
    print(json.dumps({"campaign": str(campaign), "evaluation_policies": manifest["policies"],
                      "valid_repeats": user.repeats}, indent=2))


if __name__ == "__main__":
    main()
