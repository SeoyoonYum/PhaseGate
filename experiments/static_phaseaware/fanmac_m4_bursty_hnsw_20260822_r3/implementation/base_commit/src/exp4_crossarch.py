#!/usr/bin/env python3
"""
exp4_crossarch.py - Exp 4: cross-architecture contrast point (optional, Phase 2).
[RQ4]  SKELETON - not implemented.  See EXPERIMENTS.md Exp 4.

Show N* MOVES along the unified-memory spectrum (curves only, no full-system port):
  reproduce the Exp1 cost curves on one cloud discrete GPU (PCIe; A10/4090) -> N*_PCIe,
  contrast with the M4 N*_UMA. Position GH200/GB200 analytically (cite, do NOT run).

NOTE: the PCIe run is the ONLY part NOT on the M4 (RESEARCH.md §5) - run it on a rented
box and drop its Exp1 CSV into results/csv/ for the contrast plot.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import models  # noqa: F401  (skeleton imports)


def crossover_from_curves(prefill, decode):
    """TODO: derive N* (keep-vs-recompute crossover) from a platform's Exp1 curves."""
    raise NotImplementedError


def run(args):
    """TODO: load M4 (UMA) and PCIe Exp1 CSVs, compute N* for each, emit the single
    contrast figure (N* across PCIe vs single-pool UMA).
    Writes results/figures/exp4_crossarch.png."""
    raise NotImplementedError("Exp4 not implemented - see EXPERIMENTS.md Exp 4")


def main():
    ap = argparse.ArgumentParser(description="Exp4 cross-arch contrast (SKELETON).")
    ap.add_argument("--uma-csv", default="results/csv/", help="M4 (UMA) Exp1 CSV(s)")
    ap.add_argument("--pcie-csv", default="", help="discrete-GPU (PCIe) Exp1 CSV (rented box)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
