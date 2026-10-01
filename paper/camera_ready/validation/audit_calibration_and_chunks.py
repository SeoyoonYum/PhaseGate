#!/usr/bin/env python3
"""Read-only camera-ready audit of existing M4 traces; runs no new experiments.

Uses only the standard library. Run from any working directory. All outputs
are placed beside this script. Percentiles use linear interpolation, matching
NumPy's default. Each output records existing experimental runs; run medians
must not be confused with pooling individual observations across runs.

Use --summary-only to verify calibration from the public processed tables.
Full transition analysis additionally requires the archived raw timelines.
--data-root and --output-dir allow either public or original local layouts.

Event semantics: occupancy increments at query_admitted and decrements at
query_completed. query_started repeats the admitted count and starts the
chunk service clock. phase_enter_decode precedes cap_change, so the former
can still carry the prefill requested_cap. Drain time starts at decode entry.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PUBLIC_DATA = ROOT / "experiments/static_phaseaware/fanmac_m4_causal_shape_20260805_r5"
LOCAL_DATA = ROOT / "m4 mac mini/fanmac_m4_causal_shape_20260805_r5"
DATA = PUBLIC_DATA if PUBLIC_DATA.exists() else LOCAL_DATA


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def calibration():
    groups = defaultdict(list)
    runs = read_csv(DATA / "m4_calibration_runs.csv")
    for row in runs:
        groups[row["policy"]].append(row)
    output = []
    for policy, rows in sorted(groups.items()):
        rows.sort(key=lambda row: int(row["repeat"]))
        tpot = [float(row["normalized_p95_tpot"]) for row in rows]
        ttft = [float(row["normalized_p95_ttft"]) for row in rows]
        qps = [float(row["total_retrieval_goodput_qps"]) for row in rows]
        assert all(abs(float(row["p95_tpot_ms"]) /
                       float(row["baseline_p95_tpot_ms"]) -
                       float(row["normalized_p95_tpot"])) < 1e-12
                   for row in rows)
        output.append(dict(
            policy=policy, n=len(rows), median_qps=median(qps), min_qps=min(qps),
            max_qps=max(qps), median_kqps_3dp=f"{median(qps)/1000:.3f}",
            tpot_r0=tpot[0], tpot_r1=tpot[1], tpot_r2=tpot[2],
            min_tpot_ratio=min(tpot), max_tpot_ratio=max(tpot),
            max_tpot_ratio_3dp=f"{max(tpot):.3f}", max_ttft_ratio=max(ttft),
            eligible_at_1_25=all(t <= 1.25 and f <= 1.25
                                for t, f in zip(tpot, ttft)),
            baseline_tpot_ms=rows[0]["baseline_p95_tpot_ms"],
            baseline_ttft_ms=rows[0]["baseline_p95_ttft_ms"],
        ))
    save("calibration_audit.csv", output)
    return output


def inputs():
    baseline = read_csv(DATA / "r5_baseline_runs.csv")
    frozen = json.loads((DATA / "r5_normalization_baseline.json").read_text())
    output = []
    for metric in ("tpot", "ttft"):
        values = [float(r[f"official_p95_{metric}_ms"]) for r in baseline]
        assert median(values) == frozen[f"p95_{metric}_ms"]
        output.append(dict(metric=metric, n_runs=len(values),
                           baseline_median_of_run_p95_ms=median(values),
                           minimum_run_p95_ms=min(values),
                           maximum_run_p95_ms=max(values)))
    save("normalization_baseline_audit.csv", output)
    output = []
    for stage in ("m4_mechanism_clean", "m4_calibration", "m4_heldout"):
        for path in sorted((DATA / stage / "logs").glob("*manifest.json")):
            args = json.loads(path.read_text())["arguments"]
            output.append(dict(stage=stage, manifest=str(path.relative_to(ROOT)),
                               chunk_queries=args["chunk"],
                               feeder_task_queries=args["queries_per_task"],
                               ef_search=args["ef_search"], top_k=args["top_k"]))
    save("retrieval_configuration_audit.csv", output)


def analyze_trace(path, decode_cap, expected_requests):
    trace = json.loads(path.read_text())
    events = trace["events"]
    first = next(e["timestamp"] for e in events if e["event_type"] == "request_start")
    last = max(e["timestamp"] for e in events if e["event_type"] == "request_complete")
    active = {}
    pending_durations = {}
    admitted_ids, started_ids, completed_ids = set(), set(), set()
    request_ids, completed_request_ids = set(), set()
    token_indices = defaultdict(list)
    chunk_ms = []
    chunk_counts = set()
    transitions = []
    current = None
    prefill_complete = {}
    start_prefill = {}
    last_event = None
    above_cap_s = 0.0
    decode_s = 0.0
    for sequence, e in enumerate(events, 1):
        typ, ts = e["event_type"], e["timestamp"]
        assert e["seq"] == sequence, (path.name, "sequence", sequence)
        assert e["run_key"] == path.stem, (path.name, "run key")
        assert last_event is None or ts >= last_event["timestamp"], (path.name, "timestamp", sequence)
        if last_event is not None and last_event["llm_phase"] == "DECODE":
            dt = ts - last_event["timestamp"]
            decode_s += dt
            if last_event["active_query_count"] > decode_cap:
                above_cap_s += dt
        if typ == "query_admitted":
            assert e["query_id"] not in admitted_ids, (path.name, "duplicate admission")
            admitted_ids.add(e["query_id"])
            active[e["query_id"]] = ts
            assert e["query_count"] == 16
        elif typ == "query_started":
            assert e["query_id"] in active and e["query_id"] not in started_ids
            assert ts >= active[e["query_id"]]
            started_ids.add(e["query_id"])
            active[e["query_id"]] = ts
            pending_durations[e["query_id"]] = ts
            chunk_counts.add(e["query_count"])
        elif typ == "query_completed":
            assert e["query_id"] in started_ids and e["query_id"] not in completed_ids
            assert e["query_count"] == 16
            completed_ids.add(e["query_id"])
            start = pending_durations.pop(e["query_id"])
            assert ts >= start
            if first <= start and ts <= last:
                chunk_ms.append((ts - start) * 1000)
            active.pop(e["query_id"])
        elif typ == "phase_enter_prefill":
            assert e["request_id"] not in start_prefill
            start_prefill[e["request_id"]] = ts
        elif typ == "request_start":
            assert e["request_id"] in start_prefill and e["request_id"] not in request_ids
            request_ids.add(e["request_id"])
        elif typ == "prefill_complete":
            assert e["request_id"] in request_ids and e["request_id"] not in prefill_complete
            prefill_complete[e["request_id"]] = ts
        elif typ == "phase_enter_decode":
            assert current is None and e["request_id"] in prefill_complete
            assert ts >= prefill_complete[e["request_id"]]
            current = dict(
                run_key=path.stem, request_id=e["request_id"], start=ts,
                active_at_decode=len(active), residual_ids=set(active),
                all_carry_in_complete_ms=0.0 if not active else None,
                stable_cap_ms=0.0 if len(active) <= decode_cap else None,
                cap_applied_ms=None, transition_gap_ms=None,
                prefill_ms=(prefill_complete[e["request_id"]] -
                            start_prefill[e["request_id"]]) * 1000,
            )
        assert e["active_query_count"] == len(active), (
            path.name, "active counter differs from admitted-minus-completed IDs", sequence)
        if typ == "token_ready":
            assert current is not None and e["request_id"] == current["request_id"]
            token_indices[e["request_id"]].append(e["token_index"])
        if current is not None:
            if current["all_carry_in_complete_ms"] is None:
                if not current["residual_ids"].intersection(active):
                    current["all_carry_in_complete_ms"] = (ts-current["start"])*1000
            # Last transition from >d to <=d within this decode interval.
            # This covers any admission before the lower cap is applied.
            if len(active) > decode_cap:
                current["stable_cap_ms"] = None
            elif current["stable_cap_ms"] is None:
                current["stable_cap_ms"] = (ts-current["start"])*1000
            if typ == "cap_change" and e["requested_cap"] == decode_cap:
                if current["cap_applied_ms"] is None:
                    current["cap_applied_ms"] = (ts-current["start"])*1000
            if typ == "token_ready" and current["transition_gap_ms"] is None:
                current["transition_gap_ms"] = (
                    ts-prefill_complete[current["request_id"]])*1000
            if typ == "request_complete":
                assert e["request_id"] == current["request_id"]
                assert e["request_id"] not in completed_request_ids
                completed_request_ids.add(e["request_id"])
                current["decode_ms"] = (ts-current["start"])*1000
                assert current["all_carry_in_complete_ms"] is not None
                assert current["stable_cap_ms"] is not None, (path.name, "undrained at decode end")
                assert current["cap_applied_ms"] is not None
                assert current["transition_gap_ms"] is not None
                del current["residual_ids"]
                del current["start"]
                transitions.append(current)
                current = None
        last_event = e
    assert not active and not pending_durations and current is None
    assert admitted_ids == started_ids == completed_ids
    assert request_ids == completed_request_ids == set(prefill_complete) == set(start_prefill)
    assert len(request_ids) == len(transitions) == expected_requests
    assert all(indices == list(range(128)) for indices in token_indices.values())
    assert set(token_indices) == request_ids
    reported = trace["observer_reconstruction_audit"]
    assert reported["event_count"] == len(events)
    assert reported["token_events"] == expected_requests * 128
    for key in ("admitted_queries", "started_queries", "completed_queries"):
        assert reported[key] == len(admitted_ids) * 16
    assert chunk_counts == {16}
    result = dict(
        run_key=path.stem, n_chunks_within_request_window=len(chunk_ms),
        chunk_queries=16, chunk_service_p50_ms=median(chunk_ms),
        chunk_service_p95_ms=pct(chunk_ms, .95), chunk_service_max_ms=max(chunk_ms),
        n_transitions=len(transitions),
        event_count=len(events), matched_chunk_ids=len(admitted_ids),
        event_integrity_verified=True,
        max_active_at_decode=max(t["active_at_decode"] for t in transitions),
        stable_cap_p50_ms=median(t["stable_cap_ms"] for t in transitions),
        stable_cap_p95_ms=pct([t["stable_cap_ms"] for t in transitions], .95),
        stable_cap_max_ms=max(t["stable_cap_ms"] for t in transitions),
        carry_in_finish_p95_ms=pct([t["all_carry_in_complete_ms"] for t in transitions], .95),
        carry_in_finish_max_ms=max(t["all_carry_in_complete_ms"] for t in transitions),
        cap_applied_p95_ms=pct([t["cap_applied_ms"] for t in transitions
                               if t["cap_applied_ms"] is not None], .95),
        transition_gap_p95_ms=pct([t["transition_gap_ms"] for t in transitions], .95),
        prefill_p50_ms=median(t["prefill_ms"] for t in transitions),
        decode_p50_ms=median(t["decode_ms"] for t in transitions),
        active_above_decode_cap_wall_fraction=above_cap_s/decode_s,
    )
    return result, transitions


def traces():
    runs = read_csv(DATA / "m4_heldout_runs.csv")
    out, requests = [], []
    for row in runs:
        if row["policy"] != "phasegate4to1":
            continue
        path = DATA / "m4_heldout/raw/timelines" / (row["run_key"] + ".json")
        result, transitions = analyze_trace(path, 1, 250)
        out.append(result)
        requests.extend(transitions)
    assert len(out) == 7 and len(requests) == 1750
    assert len({row["run_key"] for row in out}) == 7
    save("heldout_chunk_transition_runs.csv", out)
    save("heldout_chunk_transitions.csv", requests)
    return out


def zero_cap_traces():
    """Audit residual work without claiming it explains TPOT nonmonotonicity."""
    output = []
    for row in read_csv(DATA / "m4_calibration_runs.csv"):
        if row["policy"] not in {"phasegate2to0", "phasegate4to0"}:
            continue
        path = DATA / "m4_calibration/raw/timelines" / (row["run_key"] + ".json")
        result, _ = analyze_trace(path, 0, 100)
        result.update(
            policy=row["policy"], repeat=int(row["repeat"]),
            admitted_vectors_decode=int(row["admitted_queries_decode"]),
            completed_vectors_decode=int(row["completed_queries_decode"]),
            normalized_p95_tpot=float(row["normalized_p95_tpot"]),
        )
        output.append(result)
    assert len(output) == 6
    save("calibration_zero_cap_transitions.csv", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--summary-only", action="store_true",
                        help="Audit calibration using only public processed CSVs.")
    args = parser.parse_args()
    DATA = args.data_root.resolve()
    OUT = args.output_dir.resolve()
    if not (DATA / "m4_calibration_runs.csv").is_file():
        parser.error("m4_calibration_runs.csv not found; provide --data-root.")
    if not args.summary_only:
        required = [DATA / stage / "raw/timelines"
                    for stage in ("m4_heldout", "m4_calibration")]
        if any(not directory.is_dir() for directory in required):
            parser.error("Full audit requires archived raw timelines. Use "
                         "--summary-only for the public calibration tables, or "
                         "--data-root with the extracted raw campaign.")
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        inputs()
    for row in calibration():
        print(row["policy"], row["median_kqps_3dp"], row["max_tpot_ratio_3dp"],
              row["eligible_at_1_25"])
    if args.summary_only:
        raise SystemExit(0)
    rows = traces()
    for key in rows[0]:
        if key == "run_key":
            continue
        values = [r[key] for r in rows]
        print(key, "median", median(values), "min", min(values), "max", max(values))
    for row in zero_cap_traces():
        print(row["policy"], row["repeat"], "zero admission",
              row["admitted_vectors_decode"], "p95 drain", row["stable_cap_p95_ms"],
              "decode residual wall fraction", row["active_above_decode_cap_wall_fraction"])
