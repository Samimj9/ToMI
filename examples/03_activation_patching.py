"""
examples/03_activation_patching.py
------------------------------------
Example 3: Full activation patching sweep.

Demonstrates the core causal tracing methodology:
1. Define a clean prompt and a corrupted prompt.
2. Run activation_patching() to score every hook point.
3. Visualise the importance matrix as a heatmap.
4. Run attribution_patching() for a fast gradient-based approximation.

The task used here is a simplified false-belief scenario where:
  - clean prompt    → model should predict "basket"
  - corrupted prompt → a version where the belief-relevant context is removed

Run with:
    python examples/03_activation_patching.py
"""

from __future__ import annotations

import torch
import tomi
from tomi.utils.logging import enable_info
from tomi.utils.tokenizer import get_token_ids

enable_info()

MODEL_NAME = "Qwen/Qwen2-0.5B"
print(f"\n{'='*60}\nLoading {MODEL_NAME} …\n{'='*60}")
model = tomi.load_model(MODEL_NAME)

# ── 1. Define prompts ─────────────────────────────────────────────────────
# Clean: full false-belief context
CLEAN = (
    "Sally puts the marble in the basket. "
    "Sally leaves. "
    "Anne moves the marble to the box. "
    "Sally comes back. Where will Sally look? The"
)

# Corrupted: Sally directly observes the move (no false belief)
CORRUPTED = (
    "Sally puts the marble in the basket. "
    "Sally stays. "
    "Anne moves the marble to the box. "
    "Where will Sally look? The"
)

# Tokenise
clean_ids = model.tokenize(CLEAN, padding=False)["input_ids"]
corrupted_ids = model.tokenize(CORRUPTED, padding=False)["input_ids"]

print(f"\nClean prompt tokens    : {clean_ids.shape[1]}")
print(f"Corrupted prompt tokens: {corrupted_ids.shape[1]}")

# ── 2. Identify answer tokens ─────────────────────────────────────────────
# We compare "basket" vs "box" as the two competing answers
try:
    correct_ids = get_token_ids(model.tokenizer, "basket", add_prefix_space=True)
    incorrect_ids = get_token_ids(model.tokenizer, "box", add_prefix_space=True)
    correct_token_id = correct_ids[0]
    incorrect_token_id = incorrect_ids[0]
except ValueError as e:
    # Fallback: use token ids directly if multi-token
    print(f"Warning: {e}")
    print("Falling back to first-token approximation …")
    correct_token_id = model.tokenizer.encode(" basket", add_special_tokens=False)[0]
    incorrect_token_id = model.tokenizer.encode(" box", add_special_tokens=False)[0]

print(f"\nCorrect token   : id={correct_token_id}  "
      f"'{model.tokenizer.decode([correct_token_id])}'")
print(f"Incorrect token : id={incorrect_token_id}  "
      f"'{model.tokenizer.decode([incorrect_token_id])}'")

# ── 3. Baseline metrics ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Computing baselines …")
print('='*60)

clean_logits = model.get_logits(clean_ids)
corrupted_logits = model.get_logits(corrupted_ids)

clean_diff = tomi.logit_diff(clean_logits, correct_token_id, incorrect_token_id)
corrupted_diff = tomi.logit_diff(corrupted_logits, correct_token_id, incorrect_token_id)

print(f"  Clean logit diff     : {clean_diff.item():+.4f}")
print(f"  Corrupted logit diff : {corrupted_diff.item():+.4f}")

if clean_ids.shape[1] != corrupted_ids.shape[1]:
    print(
        "\n⚠  Clean and corrupted prompts have different lengths. "
        "Patching requires matching sequence lengths.\n"
        "   Padding to the same length …"
    )
    max_len = max(clean_ids.shape[1], corrupted_ids.shape[1])
    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    if clean_ids.shape[1] < max_len:
        pad = torch.full((1, max_len - clean_ids.shape[1]), pad_id, device=model.device)
        clean_ids = torch.cat([pad, clean_ids], dim=1)
    if corrupted_ids.shape[1] < max_len:
        pad = torch.full((1, max_len - corrupted_ids.shape[1]), pad_id, device=model.device)
        corrupted_ids = torch.cat([pad, corrupted_ids], dim=1)

# ── 4. Full activation patching ───────────────────────────────────────────
print(f"\n{'='*60}")
print("Running full activation patching …")
print(f"  (patches {len(model.hook_names)} hook points, one at a time)")
print('='*60)

result = tomi.activation_patching(
    model=model,
    clean_tokens=clean_ids,
    corrupted_tokens=corrupted_ids,
    correct_token_id=correct_token_id,
    incorrect_token_id=incorrect_token_id,
    show_progress=True,
)

print(f"\n  Patching complete:")
print(f"    Hooks scored          : {len(result.hook_names)}")
print(f"    Baseline (clean)      : {result.baseline_clean:+.4f}")
print(f"    Baseline (corrupted)  : {result.baseline_corrupted:+.4f}")

# Top 10 most important hooks
top_10 = sorted(
    zip(result.hook_names, result.importance_matrix),
    key=lambda x: abs(x[1]),
    reverse=True,
)[:10]

print(f"\n  Top 10 most important hook points:")
for name, score in top_10:
    bar = "█" * int(abs(score) * 20)
    print(f"    {name:<35} NCE={score:+.4f}  {bar}")

# ── 5. Visualise as layer matrix ──────────────────────────────────────────
print(f"\n{'='*60}")
print("Building importance matrix …")
print('='*60)

mat = result.as_layer_matrix(
    n_layers=model.n_layers,
    components=("resid.post", "attn.out", "mlp.post"),
)
print(f"  Importance matrix shape: {mat.shape}  (layers × components)")

try:
    fig = tomi.plot_activation_patching_results(result, n_layers=model.n_layers)
    fig.savefig("patching_heatmap.png", dpi=150, bbox_inches="tight")
    print("  Saved heatmap to patching_heatmap.png")
except Exception as e:
    print(f"  (Could not save plot: {e})")

# ── 6. Attribution patching (fast alternative) ────────────────────────────
print(f"\n{'='*60}")
print("Running attribution patching (gradient-based, ~2× faster) …")
print('='*60)

attr_result = tomi.attribution_patching(
    model=model,
    clean_tokens=clean_ids,
    corrupted_tokens=corrupted_ids,
    correct_token_id=correct_token_id,
    incorrect_token_id=incorrect_token_id,
)

print(f"\n  Top 10 attribution scores:")
for name, score in attr_result.top_k(10):
    bar = "█" * min(40, int(abs(score) * 5))
    print(f"    {name:<35} attr={score:+.4f}  {bar}")

print("\n✓ Example 03 complete.\n")
