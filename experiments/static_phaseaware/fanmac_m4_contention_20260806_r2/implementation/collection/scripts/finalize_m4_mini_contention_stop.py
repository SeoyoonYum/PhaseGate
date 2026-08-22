#!/usr/bin/env python3
"""Finalize a stopped M4 Mini contention campaign after both baseline gates fail."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


REPO = Path(__file__).resolve().parents[1]


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def thermal_clean(snapshot: dict[str, Any]) -> bool:
    text = snapshot["thermal"]["stdout"].lower()
    return (snapshot["thermal"]["returncode"] == 0
            and "no thermal warning level has been recorded" in text
            and "no performance warning level has been recorded" in text)


def create_bundle(campaign: Path) -> tuple[Path, str, int]:
    output = campaign / "m4_mini_contention_results_bundle.zip"
    checksum = campaign / "m4_mini_contention_results_bundle.sha256"
    home = str(Path.home())
    entries: list[tuple[str, bytes]] = []
    for path in sorted(campaign.rglob("*")):
        if not path.is_file() or path in {output, checksum}:
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".csv", ".md", ".txt", ".log"}:
            continue
        data = path.read_text().replace(home, "$HOME").encode()
        entries.append((f"campaign/{path.relative_to(campaign)}", data))
    for name in ("run_m4_mini_contention.py", "analyze_m4_mini_contention.py",
                 "finalize_m4_mini_contention_stop.py"):
        path = REPO / "scripts" / name
        entries.append((f"scripts/{name}", path.read_text().replace(home, "$HOME").encode()))
    hashes = [f"{sha256_bytes(data)}  {name}" for name, data in entries]
    entries.append(("BUNDLE_CONTENTS.sha256", ("\n".join(hashes) + "\n").encode()))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries):
            if home.encode() in data:
                raise RuntimeError(f"private path remains in {name}")
            info = zipfile.ZipInfo(name, (2026, 8, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    digest = sha256(output)
    checksum.write_text(f"{digest}  {output.name}\n")
    return output, digest, len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("campaign", type=Path)
    args = parser.parse_args(); campaign = args.campaign.resolve()
    a_result = json.loads((campaign / "CONTENTION_BASELINE_A_RESULT.json").read_text())
    b_result = json.loads((campaign / "CONTENTION_BASELINE_B_RESULT.json").read_text())
    if a_result["passed"] or b_result["passed"]:
        raise RuntimeError("stop finalizer may only be used when both frozen sets fail")
    if (campaign / "m4_mini_contention").exists():
        raise RuntimeError("contention blocks exist despite failed baseline gate")
    rows_a = csv_rows(campaign / "contention_baseline_A_runs.csv")
    rows_b = csv_rows(campaign / "contention_baseline_B_runs.csv")
    if len(rows_a) != 5 or len(rows_b) != 5:
        raise RuntimeError("expected exactly five rows in each baseline set")
    combined = rows_a + rows_b
    write_csv(campaign / "m4_mini_contention_baseline_runs.csv", combined)

    request_rows = []
    for set_name in ("A", "B"):
        objects = jsonl(campaign / f"contention_baseline_{set_name}/raw/requests.jsonl")
        for obj in objects:
            repeat = int(obj["run_key"].split("_r")[-1].split("_")[0])
            for index, request in enumerate(obj["requests"]):
                gaps = [float(value) for value in request["tpot_intervals_ms"]]
                request_rows.append({
                    "baseline_set": set_name, "repeat": repeat, "run_key": obj["run_key"],
                    "request_index": index, "request_id": request["request_id"],
                    "ttft_ms": request["ttft_ms"], "mean_tpot_ms": request["mean_tpot_ms"],
                    "p50_tpot_ms": request["p50_tpot_ms"], "p95_tpot_ms": request["p95_tpot_ms"],
                    "p99_tpot_ms": request["p99_tpot_ms"], "max_gap_ms": max(gaps),
                    "token_timestamp_count": len(request["token_timestamps"]),
                })
    if len(request_rows) != 1500:
        raise RuntimeError(f"expected 1,500 request rows, found {len(request_rows)}")
    write_csv(campaign / "m4_mini_contention_baseline_request_metrics.csv", request_rows)

    thermal = jsonl(campaign / "thermal_power_audit.jsonl")
    thermal_clean_count = sum(thermal_clean(row["before"]) and thermal_clean(row["after"])
                              for row in thermal)
    swap_growth = sum(int(row["swap_used_delta_bytes"]) > 0 for row in combined)
    pressure_failures = sum(row["memory_pressure_clean"] != "True" for row in combined)
    observer_subprocess = sum(int(row["observer_subprocess_count"]) > 0 for row in combined)
    pageout_flags_a = sum(int(row["pageouts_delta"]) > 0 for row in rows_a)
    pageout_flags_b = sum(int(row["pageouts_delta"]) > 0 for row in rows_b)
    tpot_fail_a = [row for row in rows_a if row["within_3pct_both"] != "True"]
    tpot_fail_b = [row for row in rows_b if row["within_3pct_both"] != "True"]

    report = f"""# M4 Mini Contention Campaign Stop Report

## Outcome

The diagnostic contention campaign stopped before its 12-block LLM-only/Fixed-1/Fixed-2/Fixed-4 matrix because both preregistered isolated-baseline sets failed. Set A failed, one documented environment-only correction was applied, and the single pre-frozen set B also failed. No set C was run. No contention curve, normalized cap comparison, MacBook-Air comparison, or cap conclusion was produced.

The completed Base-M4 PhaseGate r5 campaign is separate and remains valid; this stop applies only to `fanmac_m4_contention_20260806_r1`.

## Frozen device and workload

- Fan-cooled base-M4 Mac mini, Mac16,10, 4P+6E CPU cores, 10 GPU cores, 16 GB unified memory.
- Qwen2.5-1.5B-Instruct 4-bit, revision `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`.
- Context 2,048; output 128; 150 measured requests per baseline repeat; five repeats per set.
- Event observer, <=1 Hz memory sampling, zero repeated `ps` subprocesses.
- Measurement code freeze: `{json.loads((campaign / 'BASELINE_PROTOCOL_FREEZE.json').read_text())['repository_commit']}`.
- `K_hi=2` remained frozen; cap 4 was diagnostic-only and was never executed.

## Baseline set A

- Median p95 TPOT: {a_result['median_p95_tpot_ms']:.3f} ms.
- Median p95 TTFT: {a_result['median_p95_ttft_ms']:.2f} ms.
- TPOT/TTFT stability failures: {len(tpot_fail_a)}/5; repeat 1 had {float(rows_a[1]['tpot_abs_deviation'])*100:.2f}% absolute TPOT deviation.
- Positive-pageout blocks: {pageout_flags_a}/5; deltas: {', '.join(row['pageouts_delta'] for row in rows_a)}.
- Swap growth: 0/5; warning/critical memory pressure: 0/5.

After set A, the machine was at 93% reported free memory, zero compressor pages, zero swap, AC power, and no thermal/performance warning. A quiet idle/cooldown window was documented in `BASELINE_B_ENVIRONMENT_CORRECTION.json`; the global pageout counter remained 1494 to 1494 during that window. No code, observer, model, workload, seed, or measurement definition changed.

## Baseline set B

- Median p95 TPOT: {b_result['median_p95_tpot_ms']:.3f} ms.
- Median p95 TTFT: {b_result['median_p95_ttft_ms']:.2f} ms.
- Stability failures: {len(tpot_fail_b)}/5. Repeat 2 TPOT deviation was {float(rows_b[2]['tpot_abs_deviation'])*100:.2f}%; repeat 3 was {float(rows_b[3]['tpot_abs_deviation'])*100:.2f}%.
- Positive-pageout blocks: {pageout_flags_b}/5; deltas: {', '.join(row['pageouts_delta'] for row in rows_b)}.
- Swap growth: 0/5; warning/critical memory pressure: 0/5.

Set B independently failed both the TPOT stability requirement and the zero-pageout requirement. Pooling A and B, deleting tail requests, or running until a favorable set appeared was forbidden.

## Harness and telemetry audit

- Valid baseline blocks: 10/10; each produced exactly 150 requests and 19,200 token events.
- Observer subprocess blocks: {observer_subprocess}/10.
- Swap-growth blocks: {swap_growth}/10.
- Memory-pressure failures: {pressure_failures}/10.
- Clean pre/post thermal status: {thermal_clean_count}/10.
- Request-, run-, timeline-, manifest-, and thermal/power data are retained in the campaign bundle.

The set B command manifests contain a misspelled `.faissa` index argument. This is recorded as an orchestration metadata defect. The condition was LLM-only, `shared_index_load_count` was zero in every baseline block, no retrieval query was submitted, and the FAISS path was never opened; therefore the typo did not alter baseline execution. The exact index was verified during campaign preparation. No contention block was run with the misspelled argument.

## Scientific disposition

- Do not report a Mini Fixed-1/2/4 contention curve from this campaign; none was collected.
- Do not compare Mini versus MacBook-Air normalized contention slopes from this campaign.
- Do not revise `K_hi=2`; cap 4 remained diagnostic-only and unexecuted.
- Do not treat the clean swap/pressure/thermal telemetry as overriding the failed frozen baseline gate.
- Any future attempt requires a new campaign directory and a written protocol amendment; it must not append to or relabel this campaign.
"""
    (campaign / "M4_MINI_CONTENTION_STOP_REPORT.md").write_text(report)
    stop_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign.name, "status": "stopped-before-contention-matrix",
        "measurement_commit": json.loads((campaign / "BASELINE_PROTOCOL_FREEZE.json").read_text())["repository_commit"],
        "finalizer_source_head": git("rev-parse", "HEAD"),
        "baseline_sets_run": ["A", "B"], "baseline_A_passed": False,
        "baseline_B_passed": False, "contention_valid_blocks": 0,
        "contention_expected_blocks": 12, "set_C_run": False,
        "frozen_K_hi": 2, "cap4_diagnostic_only": True, "cap4_executed": False,
        "valid_baseline_blocks": 10, "request_rows": 1500,
        "swap_growth_blocks": swap_growth, "memory_pressure_failure_blocks": pressure_failures,
        "thermal_clean_blocks": thermal_clean_count,
        "observer_subprocess_blocks": observer_subprocess,
        "set_A_pageout_deltas": [int(row["pageouts_delta"]) for row in rows_a],
        "set_B_pageout_deltas": [int(row["pageouts_delta"]) for row in rows_b],
        "set_B_unused_index_argument_typo": True,
        "raw_data_retained": True,
    }
    (campaign / "m4_mini_contention_stop_manifest.json").write_text(
        json.dumps(stop_manifest, indent=2) + "\n")
    output, digest, count = create_bundle(campaign)
    print(report)
    print(f"bundle={output} entries={count} bytes={output.stat().st_size} sha256={digest}")


if __name__ == "__main__":
    main()
