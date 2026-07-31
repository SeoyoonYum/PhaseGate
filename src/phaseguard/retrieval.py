"""Persistent FAISS-HNSW index construction and search helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def build_faiss_hnsw(path: str | Path, vectors: int, dimensions: int = 384,
                     ef_construction: int = 80, graph_degree: int = 32,
                     seed: int = 20260728, chunk_size: int = 10_000,
                     force: bool = False) -> dict[str, Any]:
    if vectors < 1_000 or dimensions < 8 or graph_degree < 2:
        raise ValueError("invalid HNSW index dimensions")
    index_path = Path(path)
    meta_path = index_path.with_suffix(index_path.suffix + ".json")
    wanted = {"backend": "faiss-hnsw", "vectors": vectors, "dimensions": dimensions,
              "ef_construction": ef_construction, "graph_degree": graph_degree, "seed": seed}
    if index_path.exists() and meta_path.exists() and not force:
        existing = json.loads(meta_path.read_text())
        if all(existing.get(k) == v for k, v in wanted.items()):
            return existing
        raise ValueError(f"existing index metadata differs: {meta_path}; use --force with a new path")
    import faiss
    faiss.omp_set_num_threads(1)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexHNSWFlat(dimensions, graph_degree)
    index.hnsw.efConstruction = ef_construction
    rng = np.random.default_rng(seed)
    for start in range(0, vectors, chunk_size):
        count = min(chunk_size, vectors - start)
        block = rng.standard_normal((count, dimensions), dtype=np.float32)
        faiss.normalize_L2(block)
        index.add(block)
    faiss.write_index(index, str(index_path))
    wanted["index_bytes"] = index_path.stat().st_size
    meta_path.write_text(json.dumps(wanted, indent=2, sort_keys=True) + "\n")
    return wanted


def load_index(path: str | Path, ef_search: int) -> tuple[Any, dict[str, Any]]:
    if ef_search < 1:
        raise ValueError("ef_search must be positive")
    import faiss
    faiss.omp_set_num_threads(1)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    index = faiss.read_index(str(p))
    index.hnsw.efSearch = ef_search
    meta_path = p.with_suffix(p.suffix + ".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "backend": "faiss-hnsw", "vectors": index.ntotal, "dimensions": index.d}
    return index, meta


def make_queries(seed: int, count: int, dimensions: int) -> np.ndarray:
    import faiss
    rng = np.random.default_rng(seed)
    queries = rng.standard_normal((count, dimensions), dtype=np.float32)
    faiss.normalize_L2(queries)
    return queries


def search_chunks(index: Any, queries: np.ndarray, top_k: int, chunk: int):
    if top_k < 1 or chunk < 1:
        raise ValueError("top_k and chunk must be positive")
    for start in range(0, len(queries), chunk):
        block = queries[start:start + chunk]
        distances, labels = index.search(block, top_k)
        if distances.shape != labels.shape or labels.shape[0] != len(block):
            raise RuntimeError("invalid FAISS search result")
        yield len(block), float(distances[0, 0])
