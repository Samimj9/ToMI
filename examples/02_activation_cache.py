"""
examples/02_activation_cache.py
---------------------------------
Example 2: Activation caching and inspection.

Demonstrates:
- Running a model with run_with_cache()
- Accessing individual activations
- Stacking the residual stream across layers
- Comparing clean vs corrupted runs
- Using ActivationStore for multi-run statistics

Run with:
    python examples/02_activation_cache.py
"""

from __future__ import annotations

import torch
import tomi
from tomi.activations.activation_cache import ActivationCache
from tomi.activations.activation_store import ActivationStore
from tomi.activations.statistics import summarise_cache, compare_activations, top_neurons
from tomi.utils.logging import enable_info

enable_info()

MODEL_NAME = "Qwen/Qwen2-0.5B"

print(f"\n{'='*60}\nLoading {MODEL_NAME} …\n{'='*60}")
model = tomi.load_model(MODEL_NAME)

# ── 1. Basic run_with_cache ───────────────────────────────────────────────
PROMPT = "Sally places the marble in the basket. Sally leaves. Anne moves the marble to the box. Where will Sally look?"

tokens = model.tokenize(PROMPT, padding=False)
input_ids = tokens["input_ids"]

print(f"\n{'='*60}")
print("Running with cache (all hook points) …")
print('='*60)

logits, cache = model.run_with_cache(input_ids)

print(f"\nCache summary:")
print(f"  Total keys : {len(cache)}")
print(f"  Sample keys: {cache.keys()[:8]}")

# ── 2. Access individual activations ─────────────────────────────────────
print(f"\n{'='*60}")
print("Accessing individual activations …")
print('='*60)

mid_layer = model.n_layers // 2

resid = cache[tomi.resid_hook(mid_layer, "post")]
attn  = cache[tomi.attn_hook(mid_layer, "out")]
mlp   = cache[tomi.mlp_hook(mid_layer, "post")]

print(f"  resid.post at layer {mid_layer}: {resid.shape}")
print(f"  attn.out   at layer {mid_layer}: {attn.shape}")
print(f"  mlp.post   at layer {mid_layer}: {mlp.shape}")

# ── 3. Filter by component ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Filtering cache by component …")
print('='*60)

attn_cache = cache.filter_by_component("attn")
mlp_cache  = cache.filter_by_component("mlp")
print(f"  Attention-only keys : {len(attn_cache)}")
print(f"  MLP-only keys       : {len(mlp_cache)}")

# ── 4. Stack residual stream ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("Stacking residual stream across all layers …")
print('='*60)

resid_stack = cache.get_residual_stream(slot="post")   # (n_layers, B, S, d_model)
print(f"  Residual stream shape : {resid_stack.shape}")
print(f"    → (n_layers={model.n_layers}, batch=1, seq_len={input_ids.shape[1]}, d_model={model.d_model})")

# Track how the residual norm evolves across layers
norms = resid_stack[:, 0, -1, :].norm(dim=-1)  # norm at last token per layer
print(f"\n  Residual norm at last token, per layer:")
for l in range(model.n_layers):
    bar = "█" * int(norms[l].item() / norms.max().item() * 20)
    print(f"    L{l:02d}: {norms[l].item():7.2f}  {bar}")

# ── 5. Filter to specific layers only ────────────────────────────────────
print(f"\n{'='*60}")
print("Caching only selected layers …")
print('='*60)

layers_of_interest = [0, model.n_layers // 2, model.n_layers - 1]
names_filter = [tomi.resid_hook(l, "post") for l in layers_of_interest]
_, small_cache = model.run_with_cache(input_ids, names_filter=names_filter)
print(f"  Filtered cache keys: {small_cache.keys()}")

# ── 6. Activation statistics ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("Computing activation statistics …")
print('='*60)

stats = summarise_cache(small_cache)
for s in stats:
    print(f"  {s.name:<30}  mean={s.mean:+.3f}  std={s.std:.3f}  norm={s.norm:.1f}")

# ── 7. Clean vs corrupted comparison ─────────────────────────────────────
print(f"\n{'='*60}")
print("Comparing clean vs corrupted activations …")
print('='*60)

# Corrupted prompt: swap the agent name
CORRUPTED = PROMPT.replace("Sally", "Mary")
corrupted_ids = model.tokenize(CORRUPTED, padding=False)["input_ids"]

_, corrupted_cache = model.run_with_cache(corrupted_ids, names_filter=names_filter)

distances = compare_activations(small_cache, corrupted_cache, names=names_filter)
for name, dist in distances.items():
    print(f"  L2 dist at {name:<30}: {dist:.4f}")

# ── 8. ActivationStore for multi-run statistics ───────────────────────────
print(f"\n{'='*60}")
print("Using ActivationStore across 3 prompts …")
print('='*60)

store = ActivationStore()
prompts = [
    "Where will Sally look for the marble?",
    "Where will Alice look for the book?",
    "Where will Bob look for the key?",
]
target_hook = tomi.resid_hook(mid_layer, "post")
for p in prompts:
    ids = model.tokenize(p, padding=False)["input_ids"]
    _, c = model.run_with_cache(ids, names_filter=[target_hook])
    store.add(c)

print(f"  Store has {store.n_runs} caches")
# Note: can't stack directly due to different seq lengths; demonstrate mean on same-length
print("  (Multi-prompt mean requires same sequence length; store supports cat() for batch dim)")
print(f"  Available keys: {store.available_keys}")

print("\n✓ Example 02 complete.\n")
