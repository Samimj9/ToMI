"""
examples/01_load_model.py
--------------------------
Example 1: Loading a model and running a basic forward pass.

This example shows how to:
- Load a Qwen or Pythia model via tomi.load_model()
- Inspect model properties (n_layers, d_model, n_heads)
- Run tokenisation
- Run a forward pass and get logits
- List all available hook points
- Generate text

Run with:
    python examples/01_load_model.py

You can substitute any HuggingFace model name that is supported by ToMI.
"""

from __future__ import annotations

import torch
import tomi
from tomi.utils.logging import enable_info

# ── 0. Enable logging ──────────────────────────────────────────────────────
enable_info()

# ── 1. Load a small model ─────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2-0.5B"   # swap for any supported model

print(f"\n{'='*60}")
print(f"Loading model: {MODEL_NAME}")
print('='*60)

model = tomi.load_model(
    MODEL_NAME,
    device=None,          # auto-selects CUDA > MPS > CPU
    dtype=torch.float32,  # use float32 on CPU
)

print(f"\nModel summary:")
print(f"  Adapter    : {type(model).__name__}")
print(f"  n_layers   : {model.n_layers}")
print(f"  d_model    : {model.d_model}")
print(f"  n_heads    : {model.n_heads}")
print(f"  d_head     : {model.d_head}")
print(f"  device     : {model.device}")

# ── 2. Tokenise a prompt ──────────────────────────────────────────────────
PROMPT = "The capital of France is"

print(f"\n{'='*60}")
print(f"Tokenising prompt: '{PROMPT}'")
print('='*60)

tokens = model.tokenize(PROMPT, padding=False)
input_ids = tokens["input_ids"]
print(f"  input_ids shape : {input_ids.shape}")
print(f"  token ids       : {input_ids[0].tolist()}")

# ── 3. Forward pass → logits ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("Running forward pass …")
print('='*60)

logits = model.get_logits(input_ids)
print(f"  logits shape : {logits.shape}")   # (1, seq_len, vocab_size)

# Most likely next token
next_token_logits = logits[0, -1, :]
top_token_id = int(next_token_logits.argmax().item())
top_token_str = model.tokenizer.decode([top_token_id])
top_prob = float(torch.softmax(next_token_logits, dim=-1).max().item())

print(f"  Top predicted token : '{top_token_str}' (id={top_token_id}, p={top_prob:.3f})")

# ── 4. List hook points ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Available hook points (first 20 of {len(model.hook_names)}):")
print('='*60)
for name in model.hook_names[:20]:
    print(f"  {name}")
if len(model.hook_names) > 20:
    print(f"  … and {len(model.hook_names) - 20} more")

# ── 5. Hook naming helpers ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Hook naming helpers:")
print('='*60)
print(f"  attn_hook(3, 'out')   → {tomi.attn_hook(3, 'out')}")
print(f"  mlp_hook(5, 'post')   → {tomi.mlp_hook(5, 'post')}")
print(f"  resid_hook(0, 'pre')  → {tomi.resid_hook(0, 'pre')}")

# ── 6. Text generation ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Generating text …")
print('='*60)

generated = model.generate(input_ids, max_new_tokens=10, do_sample=False)
generated_text = model.tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"  Prompt + generation: '{generated_text}'")

print("\n✓ Example 01 complete.\n")
