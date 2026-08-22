#!/usr/bin/env python3
"""
rag_sidecar.py — controlled RAG-shaped CPU memory load for P2 (NOT an agent; no LLM, no network).

A single CPU process that loops retrieval-shaped memory work until `seconds` elapse (or SIGTERM/
SIGINT), then prints its achieved logical bandwidth. It mirrors stream_load.c's contract so the P2
harness can drive it identically:
    READY <path>     printed once, after one-time setup (so setup is excluded from the window)
    GBPS  <value>    printed on exit (logical bytes meaningfully touched / wall-clock)

Work per iteration (interleaved, all DRAM-resident, sized > cache):
  1. vector search  — FAISS IndexIVFFlat over N_DB x D float32 (nprobe of nlist cells = approximate
     nearest-neighbour, the production RAG access pattern: random access to a small fraction of the
     index). numpy brute-force fallback if faiss is unavailable (full DB scan; labeled "numpy").
  2. embedding gather — gather GATHER_ROWS random rows from an N_EMB x D table and reduce (scattered
     row gather; low logical bytes, cache-line amplified).
  3. mmap chunk reads — read CHUNK bytes at MMAP_READS random offsets of a file-backed mmap (random
     page access, like reading document chunks from the OS page cache).

Single-threaded (env caps BLAS/faiss to 1 thread) to match the one-P-core C loads, isolating
bandwidth contention from multi-core scheduling pressure. Logical bytes UNDERSTATE true DRAM traffic
for the random parts (each gather/page touch pulls a full cache line / page) — reported as logical.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import mmap          # noqa: E402
import signal        # noqa: E402
import sys           # noqa: E402
import time          # noqa: E402

import numpy as np   # noqa: E402

D = 768
N_DB = 200_000               # FAISS index vectors  (~0.59 GB f32)
N_EMB = 500_000              # embedding gather table (~1.47 GB f32)
MMAP_BYTES = 2 * 1024**3     # 2 GB file-backed mmap
MMAP_PATH = "/tmp/rag_sidecar_2gb.bin"
NLIST, NPROBE = 256, 8       # IVF cells / probed cells (approximate NN = random access)
GATHER_ROWS = 256
MMAP_READS, CHUNK = 16, 4096
PAGE = 16 * 1024             # Apple Silicon page size
SEED = 1234

_stop = False


def _on_signal(signum, frame):
    global _stop
    _stop = True


def build_mmap():
    """Create (once) and mmap a 2 GB random file, then warm it into the page cache so the work
    loop's random reads hit DRAM, not disk. Reused across spawns (fixed path + size check)."""
    if not (os.path.exists(MMAP_PATH) and os.path.getsize(MMAP_PATH) == MMAP_BYTES):
        rng = np.random.default_rng(SEED)
        block = rng.integers(0, 256, size=64 * 1024 * 1024, dtype=np.uint8).tobytes()
        with open(MMAP_PATH, "wb") as f:
            written = 0
            while written < MMAP_BYTES:
                n = min(len(block), MMAP_BYTES - written)
                f.write(block[:n]); written += n
    fh = open(MMAP_PATH, "rb")
    mm = mmap.mmap(fh.fileno(), MMAP_BYTES, prot=mmap.PROT_READ)
    try:
        mm.madvise(mmap.MADV_WILLNEED)
    except Exception:
        pass
    s = 0
    for off in range(0, MMAP_BYTES, 256 * 1024):   # touch -> force resident
        s += mm[off]
    return fh, mm, s


def build_index():
    """Build the retrieval index. Returns (path_label, search_fn, bytes_per_search)."""
    rng = np.random.default_rng(SEED + 1)
    db = rng.standard_normal((N_DB, D)).astype(np.float32)
    try:
        import faiss
        faiss.omp_set_num_threads(1)
        quant = faiss.IndexFlatL2(D)
        index = faiss.IndexIVFFlat(quant, D, NLIST)
        index.train(db[:50_000])                   # one-time kmeans on a subset
        index.add(db)
        index.nprobe = NPROBE
        del db                                     # faiss owns the vectors now
        # logical bytes/search ~= coarse quantizer scan + nprobe inverted lists
        bps = NLIST * D * 4 + NPROBE * (N_DB // NLIST) * D * 4

        def search(q):
            index.search(q, 10)
        return "faiss-ivf", search, bps
    except Exception:                              # numpy brute-force fallback (full DB scan)
        def search(q):
            d = db @ q[0]
            np.argpartition(d, 10)[:10]
        return "numpy-flat", search, N_DB * D * 4


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    rng = np.random.default_rng(SEED + 2)
    emb = rng.standard_normal((N_EMB, D)).astype(np.float32)   # ~1.47 GB gather table
    path, search, bytes_db = build_index()
    fh, mm, warm = build_mmap()
    bytes_gather = GATHER_ROWS * D * 4
    print(f"READY {path}", flush=True)

    touched, sink = 0.0, float(warm) * 0.0
    t0 = time.perf_counter()
    while not _stop and (time.perf_counter() - t0) < seconds:
        q = rng.standard_normal((1, D)).astype(np.float32)     # 1. vector search
        search(q)
        touched += bytes_db
        idx = rng.integers(0, N_EMB, size=GATHER_ROWS)         # 2. scattered embedding gather
        sink += float(emb[idx].sum())
        touched += bytes_gather
        for _ in range(MMAP_READS):                            # 3. random mmap chunk reads
            off = int(rng.integers(0, MMAP_BYTES - CHUNK))
            b = mm[off:off + CHUNK]
            sink += b[0] + b[CHUNK // 2]
            touched += CHUNK
    elapsed = time.perf_counter() - t0
    mm.close(); fh.close()

    print(f"GBPS {touched / 1e9 / elapsed:.3f}", flush=True)
    print(f"BYTES {touched:.0f}\nELAPSED {elapsed:.3f}\nPATH {path}\nSINK {sink:.3e}", flush=True)


if __name__ == "__main__":
    main()
