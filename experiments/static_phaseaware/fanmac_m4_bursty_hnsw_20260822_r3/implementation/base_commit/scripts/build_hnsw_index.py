#!/usr/bin/env python3
"""Build a deterministic persistent FAISS-HNSW retrieval index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from phaseguard.retrieval import build_faiss_hnsw  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="experiments/phaseguard/index/hnsw_100k_d384.faiss")
    ap.add_argument("--vectors", type=int, default=100_000)
    ap.add_argument("--dimensions", type=int, default=384)
    ap.add_argument("--ef-construction", type=int, default=80)
    ap.add_argument("--graph-degree", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    path = Path(args.output)
    if not path.is_absolute(): path = REPO / path
    meta = build_faiss_hnsw(path, args.vectors, args.dimensions, args.ef_construction,
                            args.graph_degree, args.seed, force=args.force)
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__": main()
