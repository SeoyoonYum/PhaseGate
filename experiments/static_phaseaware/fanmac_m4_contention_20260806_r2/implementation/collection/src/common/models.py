"""4bit model registry + load + config extraction. See RESEARCH.md §5.

Workhorse = 1.5B; size sweep = 0.5B/1.5B/3B/7B (all 4bit). 13B+ does not fit in 16GB.
"""

REGISTRY = {
    "0.5B": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "1.5B": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",   # workhorse
    "3B":   "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "7B":   "mlx-community/Qwen2.5-7B-Instruct-4bit",
}


def resolve_model_id(size_or_id):
    """Accept a registry key ('1.5B') or pass through a full HF id."""
    return REGISTRY.get(size_or_id, size_or_id)


def load_model(size_or_id):
    """Return (model, tokenizer)."""
    try:
        from mlx_lm import load
    except Exception:
        from mlx_lm.utils import load
    return load(resolve_model_id(size_or_id))


def model_config(model):
    cfg = getattr(model, "args", None) or getattr(getattr(model, "model", None), "args", None)
    d = {}
    if cfg is not None:
        for f in ("model_type", "num_hidden_layers", "hidden_size", "num_attention_heads",
                  "num_key_value_heads", "head_dim", "vocab_size"):
            if hasattr(cfg, f):
                d[f] = getattr(cfg, f)
    return d


def kv_bytes_per_token(cfg, dtype_bytes=2):
    """Analytic KV bytes/token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes.
    Prefer the measured value from common.measure.kv_cache_bytes when a live cache exists."""
    L = cfg.get("num_hidden_layers")
    nkv = cfg.get("num_key_value_heads", cfg.get("num_attention_heads"))
    hd = cfg.get("head_dim")
    if hd is None and cfg.get("hidden_size") and cfg.get("num_attention_heads"):
        hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    if None in (L, nkv, hd):
        return None
    return 2 * L * nkv * hd * dtype_bytes
