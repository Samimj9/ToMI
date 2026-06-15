# ToMI — Theory of Mind Interpretability

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-tomi-orange.svg)](https://pypi.org/project/tomi)

> **Mechanistic interpretability for LLMs — built for Theory-of-Mind research.**

ToMI is an open-source Python library for mechanistic interpretability of
modern transformer-based language models, with a primary focus on discovering
**Theory-of-Mind (ToM) circuits**: the internal mechanisms by which LLMs
track agents' beliefs, knowledge, and perspectives.

ToMI is architecture-agnostic and designed to plug into any HuggingFace model.
It is inspired by [TransformerLens](https://github.com/neelnanda-io/TransformerLens)
but supports Qwen, Pythia, LLaMA, Mistral, and other modern architectures
natively out of the box.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Architecture adapters** | Qwen, Pythia (GPT-NeoX), LLaMA/Mistral — auto-detected |
| **Universal hook naming** | `blocks.L.attn.out`, `blocks.L.mlp.post`, `blocks.L.resid.post` |
| **Activation caching** | `run_with_cache()` → `ActivationCache` |
| **Activation patching** | Full, per-head, per-neuron, residual-stream, attribution |
| **Theory-of-Mind datasets** | False Belief, Perspective Taking, Belief Tracking |
| **Circuit discovery** | `CircuitFinder` builds `CircuitGraph` from patching results |
| **Visualization** | Heatmaps, attention patterns, circuit graphs (Matplotlib + Plotly) |
| **Type-safe** | Type hints throughout, `dataclass` results |

---

## 📦 Installation

```bash
pip install tomi
```

For development:

```bash
git clone https://github.com/tomi-interpretability/tomi
cd tomi
pip install -e ".[dev]"
```

---

## 🚀 Quickstart

### 1 — Load a model

```python
import tomi

# Auto-detects adapter (Qwen / Pythia / LLaMA)
model = tomi.load_model("Qwen/Qwen2-0.5B")
```

### 2 — Run with activation cache

```python
tokens = model.tokenize("Where will Sally look for the marble?")
input_ids = tokens["input_ids"]

logits, cache = model.run_with_cache(input_ids)

# Access any activation
resid_post = cache["blocks.11.resid.post"]   # (1, seq_len, d_model)
attn_out   = cache["blocks.3.attn.out"]      # (1, seq_len, d_model)
mlp_post   = cache["blocks.7.mlp.post"]      # (1, seq_len, d_mlp)
```

### 3 — Activation patching

```python
result = tomi.activation_patching(
    model=model,
    clean_tokens=clean_ids,
    corrupted_tokens=corrupted_ids,
    correct_token_id=correct_id,
    incorrect_token_id=incorrect_id,
)

# Plot the result
tomi.plot_activation_patching_results(result, n_layers=model.n_layers)
```

### 4 — Per-head patching

```python
head_result = tomi.head_patching(
    model=model,
    clean_tokens=clean_ids,
    corrupted_tokens=corrupted_ids,
    correct_token_id=correct_id,
    incorrect_token_id=incorrect_id,
)

tomi.plot_head_importance(head_result.importance_matrix)
```

### 5 — False Belief evaluation

```python
from tomi.theory_of_mind import build_false_belief_dataset, ToMEvaluator

dataset = build_false_belief_dataset(n_variants=10)
evaluator = ToMEvaluator(model)
report = evaluator.evaluate(dataset)

print(report.summary())
# EvaluationReport
#   Tasks          : 20
#   Accuracy       : 45.00%
#   Belief Logit Δ : +1.2341
#   Belief P(ans)  : 0.6123
```

### 6 — Circuit discovery

```python
from tomi.circuits import CircuitFinder

finder = CircuitFinder(model, importance_threshold=0.1)
circuit = finder.find_circuit(
    clean_tokens=clean_ids,
    corrupted_tokens=corrupted_ids,
    correct_token_id=correct_id,
    incorrect_token_id=incorrect_id,
)

tomi.plot_circuit_graph(circuit)
```

---

## 🏗️ Architecture

```
tomi/
├── models/          # HuggingFace adapters (Qwen, Pythia, LLaMA, …)
├── hooks/           # HookPoint + HookManager infrastructure
├── activations/     # ActivationCache + ActivationStore
├── patching/        # Activation / head / neuron / residual / attribution patching
├── circuits/        # CircuitNode / CircuitEdge / CircuitGraph / CircuitFinder
├── theory_of_mind/  # False Belief, Perspective Taking, Belief Tracking, Evaluator
├── metrics/         # logit_diff, causal_effect, belief_score
├── visualization/   # Heatmaps, attention, circuit graphs
└── utils/           # Logging, device, tensor, tokenizer helpers
```

---

## 🔌 Supported Models

| Family | Example | Adapter |
|--------|---------|---------|
| Qwen | `Qwen/Qwen2-0.5B` | `QwenAdapter` |
| Qwen 2.5 | `Qwen/Qwen2.5-7B-Instruct` | `QwenAdapter` |
| Pythia | `EleutherAI/pythia-70m` | `PythiaAdapter` |
| LLaMA 2/3 | `meta-llama/Llama-2-7b-hf` | `LlamaAdapter` |
| Mistral | `mistralai/Mistral-7B-v0.1` | `LlamaAdapter` |

Add a new architecture:

```python
from tomi.models.qwen import QwenAdapter
from tomi.models.registry import register_adapter

register_adapter("MyNewModelForCausalLM", QwenAdapter)  # or a custom subclass
```

---

## 🪝 Hook Naming Convention

All hooks follow the schema:

```
blocks.<layer>.<component>.<slot>
```

| Hook | Description |
|------|-------------|
| `blocks.L.attn.q` | Query projections |
| `blocks.L.attn.k` | Key projections |
| `blocks.L.attn.v` | Value projections |
| `blocks.L.attn.out` | Attention output |
| `blocks.L.mlp.pre` | MLP input |
| `blocks.L.mlp.post` | MLP output |
| `blocks.L.resid.pre` | Residual stream (layer input) |
| `blocks.L.resid.mid` | Residual stream (after attention) |
| `blocks.L.resid.post` | Residual stream (layer output) |
| `embed.out` | Token embedding output |
| `unembed.pre` | Final layer norm output |
| `unembed.out` | Logits |

---

## 🧪 Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 📓 Example Notebooks

Located in `examples/`:

| Notebook | Description |
|----------|-------------|
| `01_load_model.py` | Load Qwen/Pythia and run a forward pass |
| `02_activation_cache.py` | Cache and inspect activations |
| `03_activation_patching.py` | Full activation patching sweep |
| `04_head_patching.py` | Per-head importance analysis |
| `05_false_belief_analysis.py` | ToM evaluation + patching on FBT |

---

## 📐 Design Principles

1. **Modular** — each subpackage is independently usable
2. **Architecture-agnostic** — adapters translate model internals to a common API
3. **Type-safe** — full type hints and dataclass result objects
4. **Tested** — pytest suite covering all core modules
5. **Pip-installable** — standard `pyproject.toml` packaging

---

## 🤝 Contributing

Pull requests are welcome.  Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run `pytest` and `ruff check tomi/`
5. Open a pull request

---

## 📄 Citation

If you use ToMI in your research, please cite:

```bibtex
@software{tomi2024,
  title  = {ToMI: Theory of Mind Interpretability},
  year   = {2024},
  url    = {https://github.com/tomi-interpretability/tomi},
}
```

---

## 📜 License

MIT — see [LICENSE](LICENSE).
