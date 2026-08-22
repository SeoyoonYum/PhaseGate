"""workloads.py - multiturn trace loaders + token builders for Exp 3.

Exp 3 is a cost-model SIMULATION (no model execution), so a conversation is reduced to its
shape: a list of turns, each (prompt_tokens, response_tokens). Two sources:
  - load_sharegpt(): real ShareGPT / LMSys-Chat style JSON (download separately into data/raw/).
  - synthetic_conversations(): plausible length-only traces, runnable now with no download.
"""
import json
import math
import random


def dummy_tokens(n):
    """(1, N) int32 dummy ids — re-export of measure.make_tokens (lazy import so the pure
    cost-model simulation path does not pull in MLX)."""
    from common.measure import make_tokens
    return make_tokens(n)


def _approx_tokens(text):
    """Rough token count without a tokenizer (~1.3 tokens/word)."""
    return max(1, int(len(text.split()) * 1.3))


def load_sharegpt(path, max_convs=None, tokenizer=None):
    """ShareGPT / LMSys-Chat style JSON -> list of conversations; each is a list of
    (prompt_tokens, response_tokens) formed by pairing consecutive human->gpt turns.
    Uses the tokenizer for exact counts if given, else a word-count approximation."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data") or data.get("conversations") or list(data.values())
    convs = []
    for item in data:
        turns = item.get("conversations") or item.get("conversation") or item.get("turns") or []
        pairs, pending = [], None
        for t in turns:
            role = (t.get("from") or t.get("role") or "").lower()
            val = t.get("value") or t.get("content") or t.get("text") or ""
            n = len(tokenizer.encode(val)) if tokenizer else _approx_tokens(val)
            if role in ("human", "user"):
                pending = n
            elif role in ("gpt", "assistant", "bot") and pending is not None:
                pairs.append((pending, n))
                pending = None
        if pairs:
            convs.append(pairs)
            if max_convs and len(convs) >= max_convs:
                break
    return convs


def synthetic_conversations(n_convs=300, seed=0, mean_turns=8, turn_sigma=0.6,
                            mean_prompt=110, mean_resp=240, len_sigma=0.7):
    """Length-only multiturn traces for offline policy simulation (no download).
    Turn counts and per-turn prompt/response lengths are lognormal (heavy-tailed, like real
    chat). Deterministic given `seed`."""
    rng = random.Random(seed)
    convs = []
    for _ in range(n_convs):
        n_turns = max(1, round(rng.lognormvariate(math.log(mean_turns), turn_sigma)))
        pairs = []
        for _ in range(n_turns):
            p = max(8, round(rng.lognormvariate(math.log(mean_prompt), len_sigma)))
            r = max(8, round(rng.lognormvariate(math.log(mean_resp), len_sigma)))
            pairs.append((p, r))
        convs.append(pairs)
    return convs


def trace_stats(convs):
    """Summary of a trace set: #convs, turns, total context tokens (sum of p+r)."""
    turns = [len(c) for c in convs]
    ctx = [sum(p + r for p, r in c) for c in convs]
    summ = lambda a: (min(a), int(sum(a) / len(a)), max(a)) if a else (0, 0, 0)
    return dict(n_convs=len(convs), turns_min_mean_max=summ(turns),
                ctx_tokens_min_mean_max=summ(ctx))
