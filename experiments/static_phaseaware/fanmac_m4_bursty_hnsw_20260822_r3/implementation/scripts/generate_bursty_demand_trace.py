#!/usr/bin/env python3
"""Write one immutable seeded DemandGate trace using the frozen generator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from phaseguard.demand_gate import generate_demand_trace  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demand-level", type=float, required=True,
                        choices=(0.05, 0.25, 1.0))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--duration-s", type=float, default=7200.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = generate_demand_trace(demand_level=args.demand_level, seed=args.seed,
                                  duration_s=args.duration_s)
    trace.write(args.output)
    print(json.dumps({"output": str(args.output.resolve()),
                      "demand_level": trace.demand_level, "seed": trace.seed,
                      "duration_s": trace.duration_s,
                      "interval_count": len(trace.intervals)}, indent=2))


if __name__ == "__main__":
    main()
