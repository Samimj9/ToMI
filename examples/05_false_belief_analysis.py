"""
examples/05_false_belief_analysis.py
---------------------------------------
Example 5: Full Theory-of-Mind evaluation and mechanistic analysis.

This is the flagship example demonstrating the complete ToMI workflow:

1. Build a False Belief dataset (Sally-Anne + Maxi variants)
2. Evaluate the model's ToM accuracy
3. Select a strongly-answered instance for mechanistic analysis
4. Run activation patching and head patching on that instance
5. Discover the ToM circuit
6. Visualise everything

This is the direct analogue of a research paper's main analysis loop.

Run with:
    python examples/05_false_belief_analysis.py
"""

from __future__ import annotations

import torch
import tomi
from tomi.theory_of_mind import (
    build_false_belief_dataset,
    ToMEvaluator,
    FalseBelief,
)
from tomi.circuits import CircuitFinder
from tomi.utils.logging import enable_info
from tomi.utils.tokenizer import get_token_ids

enable_info()

MODEL_NAME = "Qwen/Qwen2-0.5B"
print(f"\n{'='*60}")
print(f"ToMI — False Belief Analysis")
print(f"Model: {MODEL_NAME}")
print('='*60)

model = tomi.load_model(MODEL_NAME)
print(f"Model: {model}")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 1: Evaluation
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("PHASE 1: Theory-of-Mind Evaluation")
print('='*60)

dataset = build_false_belief_dataset(n_variants=10)  # 20 tasks total
print(f"\nDataset: {len(dataset)} false-belief tasks")
for t in dataset[:3]:
    print(f"\n  Agent   : {t.agent}")
    print(f"  Object  : {t.object_}")
    print(f"  Belief  : {t.belief_answer}  (expected)")
    print(f"  Reality : {t.reality_answer}  (foil)")
    print(f"  Prompt snippet: …{t.prompt[-80:]}")

evaluator = ToMEvaluator(model)
print("\nRunning evaluation …")
report = evaluator.evaluate(dataset)

print(f"\n{report.summary()}")

# Separate correct / incorrect instances
correct_tasks   = [r for r in report.task_results if r.is_correct]
incorrect_tasks = [r for r in report.task_results if not r.is_correct]
print(f"  Correctly answered   : {len(correct_tasks)} / {len(dataset)}")
print(f"  Incorrectly answered : {len(incorrect_tasks)} / {len(dataset)}")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 2: Select an instance for mechanistic analysis
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("PHASE 2: Selecting instance for mechanistic analysis")
print('='*60)

# We want an instance with a large belief_logit_diff (model is clearly "trying")
all_results = sorted(
    report.task_results,
    key=lambda r: abs(r.belief_score_result.belief_logit_diff),
    reverse=True,
)
target_result = all_results[0]
target_task   = dataset[target_result.task_index]

print(f"\nSelected task #{target_result.task_index}:")
print(f"  Prompt            : {target_task.prompt}")
print(f"  Expected answer   : '{target_task.belief_answer}'")
print(f"  Predicted answer  : '{target_result.predicted_answer}'")
print(f"  Belief logit diff : {target_result.belief_score_result.belief_logit_diff:+.4f}")
print(f"  Is correct        : {target_result.is_correct}")

# Build prompt ending with a blank for the answer token
clean_prompt = target_task.prompt + " The"
# Corrupted: agent witnesses the move (no false belief)
corrupted_prompt = (
    clean_prompt
    .replace("leaves", "stays")
    .replace("leaves the room", "stays in the room")
)
if corrupted_prompt == clean_prompt:
    # Fallback: swap the initial location
    corrupted_prompt = clean_prompt.replace(
        target_task.initial_location,
        target_task.final_location,
    )

# Tokenise & align
clean_ids     = model.tokenize(clean_prompt,     padding=False)["input_ids"]
corrupted_ids = model.tokenize(corrupted_prompt, padding=False)["input_ids"]

max_len = max(clean_ids.shape[1], corrupted_ids.shape[1])
pad_id  = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id

def _pad(ids, length, pad_id):
    if ids.shape[1] < length:
        p = torch.full((1, length - ids.shape[1]), pad_id, device=ids.device)
        return torch.cat([p, ids], dim=1)
    return ids

clean_ids     = _pad(clean_ids,     max_len, pad_id)
corrupted_ids = _pad(corrupted_ids, max_len, pad_id)

# Answer token ids
try:
    correct_id   = get_token_ids(model.tokenizer, target_task.belief_answer, add_prefix_space=True)[0]
    incorrect_id = get_token_ids(model.tokenizer, target_task.reality_answer, add_prefix_space=True)[0]
except ValueError:
    correct_id   = model.tokenizer.encode(f" {target_task.belief_answer}",   add_special_tokens=False)[0]
    incorrect_id = model.tokenizer.encode(f" {target_task.reality_answer}",  add_special_tokens=False)[0]

print(f"\n  Answer tokens:")
print(f"    Correct   (belief)  : '{model.tokenizer.decode([correct_id])}'   id={correct_id}")
print(f"    Incorrect (reality) : '{model.tokenizer.decode([incorrect_id])}'  id={incorrect_id}")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 3: Activation patching
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("PHASE 3: Activation Patching")
print('='*60)

patch_result = tomi.activation_patching(
    model=model,
    clean_tokens=clean_ids,
    corrupted_tokens=corrupted_ids,
    correct_token_id=correct_id,
    incorrect_token_id=incorrect_id,
    show_progress=True,
)

print(f"\n  Baselines — clean: {patch_result.baseline_clean:+.4f}  |  corrupted: {patch_result.baseline_corrupted:+.4f}")

top_patches = sorted(
    zip(patch_result.hook_names, patch_result.importance_matrix),
    key=lambda x: abs(x[1]),
    reverse=True,
)[:8]
print(f"\n  Top 8 hooks by causal importance:")
for name, nce in top_patches:
    bar = "█" * int(abs(nce) * 25)
    print(f"    {name:<35}  NCE={nce:+.4f}  {bar}")

try:
    fig1 = tomi.plot_activation_patching_results(
        patch_result, n_layers=model.n_layers,
        title="Activation Patching — False Belief Task",
    )
    fig1.savefig("fb_activation_patching.png", dpi=150, bbox_inches="tight")
    print("\n  Saved: fb_activation_patching.png")
except Exception as e:
    print(f"  (Plot error: {e})")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 4: Head patching
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("PHASE 4: Per-Head Patching")
print('='*60)

head_result = tomi.head_patching(
    model=model,
    clean_tokens=clean_ids,
    corrupted_tokens=corrupted_ids,
    correct_token_id=correct_id,
    incorrect_token_id=incorrect_id,
    show_progress=True,
)

imp = head_result.importance_matrix
print(f"\n  Head importance matrix: {imp.shape}")

flat   = imp.flatten()
top5   = flat.topk(5)
print(f"\n  Top 5 attention heads:")
for rank, (score, idx) in enumerate(zip(top5.values, top5.indices), 1):
    l = idx.item() // model.n_heads
    h = idx.item() %  model.n_heads
    bar = "█" * int(score.item() * 25)
    print(f"    #{rank}  L{l:02d}H{h:02d}  NCE={score.item():.4f}  {bar}")

try:
    fig2 = tomi.plot_head_importance(
        imp, title="Head Patching Importance — False Belief Task",
    )
    fig2.savefig("fb_head_patching.png", dpi=150, bbox_inches="tight")
    print("\n  Saved: fb_head_patching.png")
except Exception as e:
    print(f"  (Plot error: {e})")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 5: Circuit discovery
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("PHASE 5: Circuit Discovery")
print('='*60)

finder  = CircuitFinder(model, importance_threshold=0.05)
circuit = finder.find_circuit(
    clean_tokens=clean_ids,
    corrupted_tokens=corrupted_ids,
    correct_token_id=correct_id,
    incorrect_token_id=incorrect_id,
    circuit_name="false_belief_circuit",
)

print(f"\n  Circuit: {circuit}")
print(f"\n  Top nodes by importance:")
for node in circuit.top_nodes(k=10):
    print(f"    {node.label:<30}  score={node.score:.4f}")

try:
    fig3 = tomi.plot_circuit_graph(
        circuit,
        title="False Belief Circuit",
        figsize=(14, 8),
    )
    fig3.savefig("fb_circuit.png", dpi=150, bbox_inches="tight")
    print("\n  Saved: fb_circuit.png")
except Exception as e:
    print(f"  (Plot error: {e})")

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
print(f"  Model             : {MODEL_NAME}")
print(f"  ToM Accuracy      : {report.accuracy:.2%}  ({len(correct_tasks)}/{len(dataset)} tasks)")
print(f"  Mean Belief Logit Δ: {report.mean_belief_logit_diff:+.4f}")
print(f"  Circuit nodes     : {len(circuit.nodes)}")
print(f"  Circuit edges     : {len(circuit.edges)}")
print(f"\n  Files generated:")
for fname in [
    "fb_activation_patching.png",
    "fb_head_patching.png",
    "fb_circuit.png",
]:
    print(f"    {fname}")

print("\n✓ Example 05 complete.\n")
