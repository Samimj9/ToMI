"""
examples/04_head_patching.py
------------------------------
Example 4: Per-attention-head patching.

Identifies which individual attention heads contribute causally to
false-belief reasoning by patching their outputs one at a time.

Demonstrates:
- head_patching() producing a (n_layers × n_heads) importance matrix
- Visualising as a heatmap
- Isolating the top-5 most important heads
- Residual-stream patching for causal tracing (layer × token position)

Run with:
    python examples/04_head_patching.py
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

# ── 1. Prompts ────────────────────────────────────────────────────────────
# For head patching the two prompts must have the same sequence length.
# We use padding-aligned versions here.
CLEAN = (
    "Sally puts the marble in the basket. "
    "Sally leaves the room. "
    "Anne moves the marble to the box. "
    "Sally comes back. Where will Sally look? The"
)

CORRUPTED = (
    "Sally puts the marble in the basket. "
    "Sally stays in the room. "
    "Anne moves the marble to the box. "
    "Sally sees this. Where will Sally look? The"
)

clean_ids = model.tokenize(CLEAN, padding=False)["input_ids"]
corrupted_ids = model.tokenize(CORRUPTED, padding=False)["input_ids"]

# Align lengths by padding the shorter one
max_len = max(clean_ids.shape[1], corrupted_ids.shape[1])
pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id

def pad_left(ids: torch.Tensor, length: int, pad_id: int) -> torch.Tensor:
    if ids.shape[1] < length:
        pad = torch.full((1, length - ids.shape[1]), pad_id, device=ids.device)
        return torch.cat([pad, ids], dim=1)
    return ids

clean_ids = pad_left(clean_ids, max_len, pad_id)
corrupted_ids = pad_left(corrupted_ids, max_len, pad_id)

# Answer tokens
correct_token_id = model.tokenizer.encode(" basket", add_special_tokens=False)[0]
incorrect_token_id = model.tokenizer.encode(" box", add_special_tokens=False)[0]

print(f"\nSequence length (after padding) : {max_len}")
print(f"Correct token   : '{model.tokenizer.decode([correct_token_id])}'  (id={correct_token_id})")
print(f"Incorrect token : '{model.tokenizer.decode([incorrect_token_id])}'  (id={incorrect_token_id})")

# ── 2. Head patching ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Running per-head patching …")
print(f"  Grid: {model.n_layers} layers × {model.n_heads} heads = {model.n_layers * model.n_heads} patches")
print('='*60)

head_result = tomi.head_patching(
    model=model,
    clean_tokens=clean_ids,
    corrupted_tokens=corrupted_ids,
    correct_token_id=correct_token_id,
    incorrect_token_id=incorrect_token_id,
    show_progress=True,
)

importance = head_result.importance_matrix  # (n_layers, n_heads)
print(f"\n  Importance matrix: {importance.shape}")
print(f"  Max importance : {importance.max().item():.4f}")
print(f"  Mean importance: {importance.mean().item():.4f}")

# ── 3. Top-5 heads ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Top 5 most important attention heads:")
print('='*60)

flat = importance.flatten()
top5_flat = flat.topk(5)
for rank, (score, flat_idx) in enumerate(zip(top5_flat.values, top5_flat.indices), 1):
    layer = flat_idx.item() // model.n_heads
    head  = flat_idx.item() %  model.n_heads
    bar = "█" * int(score.item() * 20)
    print(f"  #{rank}  L{layer:02d}H{head:02d}  NCE={score.item():.4f}  {bar}")

# ── 4. Visualise ─────────────────────────────────────────────────────────
try:
    fig = tomi.plot_head_importance(
        importance,
        title="Head Patching Importance — False Belief Task",
    )
    fig.savefig("head_patching.png", dpi=150, bbox_inches="tight")
    print("\n  Saved head-importance heatmap to head_patching.png")
except Exception as e:
    print(f"\n  (Could not save plot: {e})")

# ── 5. Residual-stream patching ───────────────────────────────────────────
print(f"\n{'='*60}")
print("Running residual-stream patching (layer × token position) …")
print('='*60)

resid_result = tomi.residual_patching(
    model=model,
    clean_tokens=clean_ids,
    corrupted_tokens=corrupted_ids,
    correct_token_id=correct_token_id,
    incorrect_token_id=incorrect_token_id,
    slot="post",
    show_progress=True,
)

print(f"\n  Residual importance shape: {resid_result.importance_matrix.shape}")
print(f"  (n_layers={resid_result.n_layers}, n_positions={resid_result.n_positions})")

# Peak importance location
flat_resid = resid_result.importance_matrix.flatten()
peak_flat = flat_resid.argmax()
peak_layer = peak_flat.item() // resid_result.n_positions
peak_pos   = peak_flat.item() %  resid_result.n_positions
peak_val   = flat_resid[peak_flat].item()
print(f"  Peak: layer={peak_layer}, position={peak_pos}, NCE={peak_val:.4f}")

# Decode token at peak position
try:
    peak_tok = model.tokenizer.decode([clean_ids[0, peak_pos].item()])
    print(f"  Token at peak position: '{peak_tok}'")
except Exception:
    pass

try:
    # Attach token labels to the result for nicer visualisation
    token_labels = [
        model.tokenizer.decode([clean_ids[0, p].item()]).replace(" ", "_")
        for p in range(clean_ids.shape[1])
    ]
    resid_result.token_labels = token_labels
    fig2 = tomi.plot_residual_patching_results(
        resid_result,
        title="Residual Stream Patching — False Belief Task",
        figsize=(16, 5),
    )
    fig2.savefig("residual_patching.png", dpi=150, bbox_inches="tight")
    print("  Saved residual patching heatmap to residual_patching.png")
except Exception as e:
    print(f"  (Could not save plot: {e})")

print("\n✓ Example 04 complete.\n")
