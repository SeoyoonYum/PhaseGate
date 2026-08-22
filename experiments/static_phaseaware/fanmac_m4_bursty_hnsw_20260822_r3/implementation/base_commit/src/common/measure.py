"""mx.eval-forced timing primitives. See RESEARCH.md §6 trap #1 (MLX is LAZY).

Every timed region is forced to completion with mx.eval() (the MLX sync barrier).
prefill is measured BODY-ONLY (transformer body that builds the KV cache, without the
full-vocab lm_head projection): the lm_head is not part of KV reconstruction and would
blow memory to ~2.5GB at N=8192. decode isolates a single step AFTER the prefill is
fully eval'd, so cache-build cost cannot leak into the per-step number.
"""
import time

import numpy as np
import mlx.core as mx


def _resolve(mod, *names):
    """mlx relocates symbols between versions (top-level in 0.31.x, mx.metal.* older)."""
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    sub = getattr(mod, "metal", None)
    if sub is not None:
        for n in names:
            if hasattr(sub, n):
                return getattr(sub, n)
    return None


set_memory_limit  = _resolve(mx, "set_memory_limit")
get_peak_memory   = _resolve(mx, "get_peak_memory")
reset_peak_memory = _resolve(mx, "reset_peak_memory")
clear_cache       = _resolve(mx, "clear_cache")


def make_prompt_cache(model):
    for path in ("mlx_lm.models.cache", "mlx_lm.cache"):
        try:
            mod = __import__(path, fromlist=["make_prompt_cache"])
            return mod.make_prompt_cache(model)
        except Exception:
            continue
    raise ImportError("could not locate make_prompt_cache in mlx_lm")


def set_mem_limit_gb(gb):
    if set_memory_limit:
        set_memory_limit(int(gb * 1024**3))
        return gb
    return None


def peak_mb():
    return (get_peak_memory() / 1024**2) if get_peak_memory else float("nan")


def reset_peak():
    if reset_peak_memory:
        reset_peak_memory()


def free_buffers():
    if clear_cache:
        clear_cache()


def make_tokens(n):
    """Dummy (1, N) int32 token ids. Cost depends on shape(N), not on token values."""
    return (mx.arange(n, dtype=mx.int32) % 100).reshape(1, n)


def forward_body(model, x, cache):
    """Transformer body that BUILDS the KV cache, WITHOUT the full-vocab lm_head."""
    body = getattr(model, "model", None)
    if body is not None:
        try:
            return body(x, cache=cache)
        except TypeError:
            pass
    return model(x, cache=cache)


def kv_cache_bytes(cache):
    """Ground-truth KV bytes from the live cache arrays. mlx Dtype has no .itemsize,
    so per-element bytes are derived via nbytes//size. Returns (total_bytes, meta)."""
    total, meta = 0, {}
    for c in cache:
        state = c.state
        if not state or state[0] is None:
            continue
        k = state[0]
        L = getattr(c, "offset", k.shape[-2])
        b, nkv, _, hd = k.shape
        ib = int(k.nbytes // k.size)
        total += 2 * b * nkv * L * hd * ib
        meta = dict(n_layers=len(cache), n_kv_heads=nkv, head_dim=hd,
                    dtype=str(k.dtype).split(".")[-1], dtype_bytes=ib, seqlen=int(L))
    return total, meta


def summarize(times):
    a = np.asarray(times, float)
    return dict(med=float(np.median(a)), min=float(a.min()), max=float(a.max()))


def time_prefill(model, N, reps=7, warmup=2):
    """Fresh cache + body-only forward over N tokens, each forced with mx.eval.
    Returns list[ms] of the `reps` measured runs (warmups excluded)."""
    times = []
    for r in range(warmup + reps):
        cache = make_prompt_cache(model)
        x = make_tokens(N)
        mx.eval(x)                          # materialize input OUTSIDE the timer
        t0 = time.perf_counter()
        out = forward_body(model, x, cache)
        mx.eval(out)                        # lazy-eval barrier: force the whole prefill
        if r >= warmup:
            times.append((time.perf_counter() - t0) * 1e3)
        del cache, out, x
        free_buffers()
    return times


def time_decode(model, N, reps=7, warmup=2):
    """Build an N-token cache, FULLY eval it (trap #1: no build cost in the step), then
    time single decode steps. Returns (list[ms], kv_bytes, kv_meta, ctx_start)."""
    cache = make_prompt_cache(model)
    x = make_tokens(N)
    mx.eval(x)
    built = forward_body(model, x, cache)
    mx.eval(built)
    for c in cache:                         # force the K/V arrays themselves too
        s = c.state
        if s and s[0] is not None:
            mx.eval(s[0], s[1])
    kvb, kvmeta = kv_cache_bytes(cache)

    one = mx.array([[7]], dtype=mx.int32)
    mx.eval(one)
    y = None
    for _ in range(warmup):                 # decode-kernel warmup (advances cache)
        y = model(one, cache=cache)
        mx.eval(y)
    ctx_start = getattr(cache[0], "offset", N)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        y = model(one, cache=cache)
        mx.eval(y)
        times.append((time.perf_counter() - t0) * 1e3)

    del cache, built, x, one, y
    free_buffers()
    return times, kvb, kvmeta, int(ctx_start)


def global_warmup(model):
    """Compile one-time kernels AND ramp the GPU clock out of idle (DVFS) so the first
    measured point is at working clock, not a cold-clock outlier (distinct from thermal
    throttling: this is clock ramp-UP, not heat)."""
    for _ in range(4):
        time_prefill(model, 256, reps=1, warmup=1)
    time_decode(model, 256, reps=12, warmup=2)
