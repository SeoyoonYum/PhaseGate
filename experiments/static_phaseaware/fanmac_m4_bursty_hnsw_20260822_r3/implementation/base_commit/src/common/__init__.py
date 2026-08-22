"""Shared measurement infrastructure for the kv-uma-research experiments.

Modules:
  measure   - mx.eval-forced timing primitives (prefill / decode), KV bytes, peak memory.
  thermal   - power-state guard, throttle detection, optional powermetrics logging.
  models    - 4bit model registry (0.5B/1.5B/3B/7B) + load + config extraction.
  workloads - dummy token builders + (TODO) multi-turn chat trace loaders.
"""
