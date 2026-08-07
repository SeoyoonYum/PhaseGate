#!/usr/bin/env python3
"""Independent completion audit for the frozen Base-M4 Mini contention r2 data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_POLICIES = {"llm-only": 0, "fixed1": 1, "fixed2": 2, "fixed4": 4}
EXPECTED_REPEATS = range(3)
EXPECTED_REQUESTS = 300
EXPECTED_TOKENS = 128


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        digest = sha256(path)
        size = path.stat().st_size
        files.append({"path": relative, "size": size, "sha256": digest})
        aggregate.update(f"{digest}  {relative}\n".encode())
    return {"file_count": len(files), "aggregate_sha256": aggregate.hexdigest(),
            "files": files}


def thermal_clean(snapshot: dict[str, Any]) -> bool:
    text = snapshot["thermal"]["stdout"].lower()
    return (snapshot["thermal"]["returncode"] == 0
            and "no thermal warning level has been recorded" in text
            and "no performance warning level has been recorded" in text)


def check_bundle(campaign: Path) -> dict[str, Any]:
    bundle = campaign / "m4_mini_contention_results_bundle.zip"
    checksum_file = campaign / "m4_mini_contention_results_bundle.sha256"
    expected = checksum_file.read_text().split()[0]
    actual = sha256(bundle)
    errors: list[str] = []
    if actual != expected:
        errors.append("outer SHA-256 mismatch")
    with zipfile.ZipFile(bundle) as archive:
        bad = archive.testzip()
        if bad:
            errors.append(f"ZIP CRC failure: {bad}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            errors.append("duplicate ZIP entry")
        contents = archive.read("BUNDLE_CONTENTS.sha256").decode().splitlines()
        expected_inner = {}
        for line in contents:
            digest, name = line.split("  ", 1)
            expected_inner[name] = digest
        for name, digest in expected_inner.items():
            if name not in names or hashlib.sha256(archive.read(name)).hexdigest() != digest:
                errors.append(f"inner checksum mismatch: {name}")
        unexpected = set(names) - set(expected_inner) - {"BUNDLE_CONTENTS.sha256"}
        if unexpected:
            errors.append(f"unhashed ZIP entries: {sorted(unexpected)}")
        private_hits = []
        secret_name_hits = []
        for name in names:
            data = archive.read(name)
            if b"/Users/m1" in data:
                private_hits.append(name)
            # Source files contain the audit's own marker literals. Restrict
            # credential-content scanning to collected campaign artifacts.
            if name.startswith("campaign/"):
                lowered = data.lower()
                if any(marker in lowered for marker in
                       (b"-----begin private key-----", b"aws_secret_access_key=",
                        b"huggingface_token=", b"github_token=")):
                    secret_name_hits.append(name)
        required = {
            "campaign/M4_MINI_CONTENTION_FINAL_REPORT.md",
            "campaign/m4_mini_contention_per_run.csv",
            "campaign/m4_mini_contention_request_metrics.csv",
            "campaign/m4_mini_contention_normalized_summary.csv",
            "campaign/m4_mini_contention_pageout_sensitivity.csv",
            "campaign/figure_m4_mini_ttft_tpot_contention.pdf",
            "campaign/figure_m4_mini_ttft_tpot_contention.png",
            "campaign/m4_mini_contention/raw/runs.jsonl",
            "campaign/m4_mini_contention/raw/requests.jsonl",
            "scripts/run_m4_mini_contention.py",
            "scripts/analyze_m4_mini_contention.py",
        }
        missing = sorted(required - set(names))
        if missing:
            errors.append(f"required ZIP entries missing: {missing}")
        if private_hits:
            errors.append(f"private home path remains: {private_hits}")
        if secret_name_hits:
            errors.append(f"secret-like material remains: {secret_name_hits}")
    return {"passed": not errors, "sha256": actual, "entry_count": len(names),
            "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--r1", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = args.output or campaign / "M4_MINI_CONTENTION_COMPLETION_AUDIT.json"
    failures: list[str] = []

    matrix = json.loads((campaign / "FROZEN_EXECUTION_MATRIX.json").read_text())
    baseline_freeze = json.loads((campaign / "BASELINE_PROTOCOL_FREEZE.json").read_text())
    complete = json.loads((campaign / "M4_MINI_CONTENTION_COLLECTION_COMPLETE.json").read_text())
    freeze_hashes = json.loads((campaign / "FREEZE_CHECKSUMS.json").read_text())
    for relative, expected in freeze_hashes["files"].items():
        path = campaign / relative
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"freeze checksum mismatch: {relative}")

    if matrix["frozen_K_hi"] != 2 or complete["frozen_K_hi"] != 2:
        failures.append("K_hi is not frozen at 2")
    if int(matrix["workload"]["measured_requests"]) != EXPECTED_REQUESTS:
        failures.append("contention request count is not 300")
    if int(baseline_freeze["workload"]["measured_requests"]) != EXPECTED_REQUESTS:
        failures.append("baseline request count is not 300")
    if complete["passed_baseline_set"] != "B":
        failures.append("official baseline is not Set B")

    baseline_results = {}
    baseline_audits = {}
    for set_name in ("A", "B"):
        result = json.loads((campaign / f"CONTENTION_BASELINE_{set_name}_RESULT.json").read_text())
        runs = jsonl(campaign / f"contention_baseline_{set_name}/raw/runs.jsonl")
        requests = {row["run_key"]: row["requests"] for row in
                    jsonl(campaign / f"contention_baseline_{set_name}/raw/requests.jsonl")}
        valid = [row for row in runs if row.get("status") == "valid"]
        checks = []
        for row in valid:
            request_rows = requests.get(row["run_key"], [])
            token_count = sum(len(item.get("token_timestamps", [])) for item in request_rows)
            monotonic = all(all(b > a for a, b in zip(item["token_timestamps"],
                                                       item["token_timestamps"][1:]))
                            for item in request_rows)
            event = row["observer_reconstruction_audit"]
            checks.append({"run_key": row["run_key"], "requests": len(request_rows),
                           "tokens": token_count, "token_monotonic": monotonic,
                           "status": row["status"], "swap_delta": row["swap_used_delta_bytes"],
                           "memory_pressure_clean": row["memory_pressure_clean"],
                           "observer_mode": row["observer_mode"],
                           "observer_subprocesses": row["observer_subprocess_count_during_block"],
                           "event_sequence_exact": event["sequence_exact"],
                           "event_timestamps_monotonic": event["timestamps_monotonic"],
                           "event_query_accounting_exact": event["query_accounting_exact"],
                           "event_admissions_within_cap": event["admissions_within_cap"]})
        if len(valid) != 5 or any(item["requests"] != EXPECTED_REQUESTS
                                  or item["tokens"] != EXPECTED_REQUESTS * EXPECTED_TOKENS
                                  or not item["token_monotonic"] or item["swap_delta"] != 0
                                  or not item["memory_pressure_clean"]
                                  or item["observer_mode"] != "event"
                                  or item["observer_subprocesses"] != 0
                                  or not item["event_sequence_exact"]
                                  or not item["event_timestamps_monotonic"]
                                  or not item["event_query_accounting_exact"]
                                  or not item["event_admissions_within_cap"]
                                  for item in checks):
            failures.append(f"baseline {set_name} raw validity audit failed")
        baseline_results[set_name] = result
        baseline_audits[set_name] = checks
    if baseline_results["A"]["passed"] is not False or baseline_results["B"]["passed"] is not True:
        failures.append("frozen A-fail/B-pass sequence is inconsistent")

    raw_runs = jsonl(campaign / "m4_mini_contention/raw/runs.jsonl")
    request_map = {row["run_key"]: row["requests"] for row in
                   jsonl(campaign / "m4_mini_contention/raw/requests.jsonl")}
    valid_runs = [row for row in raw_runs if row.get("status") == "valid"]
    expected_keys = {(policy, repeat) for policy in EXPECTED_POLICIES for repeat in EXPECTED_REPEATS}
    actual_keys = {(row["policy"], int(row["repeat"])) for row in valid_runs}
    if len(raw_runs) != 12 or len(valid_runs) != 12 or actual_keys != expected_keys:
        failures.append("contention matrix is not exactly 12 unique valid blocks")

    contention_audits = []
    for row in sorted(valid_runs, key=lambda item: (int(item["repeat"]), item["policy"])):
        policy = row["policy"]
        requests = request_map.get(row["run_key"], [])
        event = row["observer_reconstruction_audit"]
        token_count = sum(len(item.get("token_timestamps", [])) for item in requests)
        token_monotonic = all(all(b > a for a, b in zip(item["token_timestamps"],
                                                         item["token_timestamps"][1:]))
                              for item in requests)
        timeline_path = campaign / "m4_mini_contention/raw/timelines" / f"{row['run_key']}.json"
        timeline = json.loads(timeline_path.read_text())
        events = timeline["events"]
        measured_start = min(float(item["prefill_start"]) for item in requests)
        measured_end = max(float(item["decode_end"]) for item in requests)
        measured_query_events = [item for item in events
                                 if measured_start <= float(item["timestamp"]) <= measured_end
                                 and item["event_type"] in
                                 {"query_admitted", "query_started", "query_completed", "cap_change"}]
        expected_cap = EXPECTED_POLICIES[policy]
        measured_cap_exact = all(int(item["requested_cap"]) == expected_cap
                                 for item in measured_query_events)
        measured_active_max = max((int(item["active_query_count"])
                                   for item in measured_query_events), default=0)
        post_measure_drain_events = sum(float(item["timestamp"]) > measured_end
                                        and int(item.get("requested_cap", 0)) > expected_cap
                                        for item in events)
        item = {
            "run_key": row["run_key"], "policy": policy, "repeat": row["repeat"],
            "attempt": row["attempt"], "requests": len(requests), "tokens": token_count,
            "token_timestamps_monotonic": token_monotonic,
            "timeline_event_count": len(events),
            "event_sequence_exact_recomputed": [int(e["seq"]) for e in events]
                                               == list(range(1, len(events) + 1)),
            "event_timestamps_monotonic_recomputed": all(
                float(b["timestamp"]) >= float(a["timestamp"])
                for a, b in zip(events, events[1:])),
            "query_accounting_exact": event["query_accounting_exact"],
            "admissions_within_cap": event["admissions_within_cap"],
            "measured_requested_cap_exact": measured_cap_exact,
            "measured_active_max": measured_active_max,
            "expected_cap": expected_cap,
            "post_measure_drain_events_above_cap": post_measure_drain_events,
            "pageout_delta": row["pageouts_delta"],
            "pageout_soft_flag": int(row["pageouts_delta"]) > 0,
            "swap_delta": row["swap_used_delta_bytes"],
            "memory_pressure_clean": row["memory_pressure_clean"],
            "observer_mode": row["observer_mode"],
            "observer_subprocesses": row["observer_subprocess_count_during_block"],
            "queue_nonempty_fraction": row["queue_nonempty_fraction"],
        }
        contention_audits.append(item)
        if (item["requests"] != EXPECTED_REQUESTS
                or item["tokens"] != EXPECTED_REQUESTS * EXPECTED_TOKENS
                or not item["token_timestamps_monotonic"]
                or not item["event_sequence_exact_recomputed"]
                or not item["event_timestamps_monotonic_recomputed"]
                or not item["query_accounting_exact"] or not item["admissions_within_cap"]
                or not item["measured_requested_cap_exact"]
                or item["measured_active_max"] > expected_cap
                or item["swap_delta"] != 0 or not item["memory_pressure_clean"]
                or item["observer_mode"] != "event" or item["observer_subprocesses"] != 0
                or (policy != "llm-only" and float(item["queue_nonempty_fraction"]) < 0.95)):
            failures.append(f"contention raw validity audit failed: {row['run_key']}")
        del timeline

    thermal_rows = jsonl(campaign / "thermal_power_audit.jsonl")
    thermal_counts = Counter(row["stage"] for row in thermal_rows)
    if len(thermal_rows) != 22 or any(not thermal_clean(row["before"])
                                      or not thermal_clean(row["after"])
                                      for row in thermal_rows):
        failures.append("thermal audit is incomplete or not clean")

    frozen_r1 = json.loads((campaign / "R1_CHECKSUM_MANIFEST.json").read_text())
    current_r1 = tree_manifest(args.r1.resolve())
    r1_unchanged = (current_r1["file_count"] == frozen_r1["file_count"]
                    and current_r1["aggregate_sha256"] == frozen_r1["aggregate_sha256"]
                    and current_r1["files"] == frozen_r1["files"])
    if not r1_unchanged:
        failures.append("r1 tree differs from the frozen checksum manifest")

    forbidden_stage_dirs = [str(path.relative_to(campaign)) for path in campaign.rglob("*")
                            if path.is_dir() and any(term in path.name.lower() for term in
                                                    ("calibration", "phasegate", "timegate",
                                                     "heldout", "output_length"))]
    if forbidden_stage_dirs:
        failures.append(f"forbidden stage directories exist: {forbidden_stage_dirs}")

    required_files = [
        "BASELINE_PROTOCOL_FREEZE.json", "FROZEN_EXECUTION_MATRIX.json",
        "CAMPAIGN_MANIFEST.json", "CONTENTION_BASELINE_A_RESULT.json",
        "CONTENTION_BASELINE_B_RESULT.json", "contention_baseline_A_runs.csv",
        "contention_baseline_B_runs.csv", "m4_mini_contention_baseline_runs.csv",
        "m4_mini_contention_baseline_request_metrics.csv",
        "m4_mini_contention_per_run.csv", "m4_mini_contention_request_metrics.csv",
        "m4_mini_contention_normalized_summary.csv",
        "m4_mini_contention_pageout_sensitivity.csv",
        "figure_m4_mini_ttft_tpot_contention.pdf",
        "figure_m4_mini_ttft_tpot_contention.png", "M4_MINI_CONTENTION_FINAL_REPORT.md",
        "m4_mini_contention_results_bundle.zip",
        "m4_mini_contention_results_bundle.sha256",
    ]
    missing_files = [name for name in required_files if not (campaign / name).is_file()]
    if missing_files:
        failures.append(f"required deliverables missing: {missing_files}")

    bundle_audit = check_bundle(campaign)
    if not bundle_audit["passed"]:
        failures.extend(f"bundle: {error}" for error in bundle_audit["errors"])

    audit = {
        "passed": not failures,
        "failures": failures,
        "repository_commit": complete["repository_commit"],
        "baseline_sequence": {"set_A_passed": baseline_results["A"]["passed"],
                              "set_B_passed": baseline_results["B"]["passed"],
                              "official_normalization_set": complete["passed_baseline_set"]},
        "baseline_raw_audits": baseline_audits,
        "contention_valid_blocks": len(valid_runs),
        "contention_invalid_attempts": len(raw_runs) - len(valid_runs),
        "contention_raw_audits": contention_audits,
        "thermal_audit": {"rows": len(thermal_rows), "by_stage": dict(thermal_counts),
                          "all_clean": all(thermal_clean(row["before"])
                                           and thermal_clean(row["after"])
                                           for row in thermal_rows)},
        "r1_immutable": {"passed": r1_unchanged,
                         "frozen_file_count": frozen_r1["file_count"],
                         "current_file_count": current_r1["file_count"],
                         "frozen_aggregate_sha256": frozen_r1["aggregate_sha256"],
                         "current_aggregate_sha256": current_r1["aggregate_sha256"]},
        "forbidden_stage_directories": forbidden_stage_dirs,
        "missing_required_files": missing_files,
        "bundle_audit": bundle_audit,
        "analysis_protocol_note": {
            "official": "passed Set B five-run median, as required by the r2 handoff",
            "supplementary": "within-repeat contention LLM-only normalization",
            "freeze_text_discrepancy": (
                "FROZEN_EXECUTION_MATRIX.json says within-repeat normalization; it is retained "
                "unchanged as frozen metadata, while the higher-precedence r2 handoff governs "
                "the official endpoint and both normalizations are reported."),
        },
    }
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"passed": audit["passed"], "failures": failures,
                      "contention_valid_blocks": len(valid_runs),
                      "r1_unchanged": r1_unchanged,
                      "bundle_passed": bundle_audit["passed"]}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
