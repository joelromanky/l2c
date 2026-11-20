# L2C

Implementation of [**L2C** (*Feature-based Learning for Diverse and Privacy-Preserving Counterfactual Explanations*)](https://arxiv.org/abs/2209.13446), a PyTorch-based counterfactual explanation method that trains a Perturber/Selector network using Gumbel-Softmax (categorical) and Gumbel-Bernoulli (feature selection) relaxations to generate sparse, on-manifold counterfactuals.

L2C requires a differentiable PyTorch `nn.Module` oracle and supports immutable features, monotonic-increasing constraints, and correlated feature couplings.

## Installation

```bash
cd /home/jky/l2c
uv sync
```

Or with `pip`:

```bash
pip install -e /home/jky/l2c
```

## Quickstart

```python
from l2c.l2c import L2C

explainer = L2C(
    oracle=pytorch_model,                # differentiable torch.nn.Module
    continuous_features=["age", "income"],
    categorical_features=["edu", "gender"],
    immutable_features=["gender"],
    monotonic_increasing_column_names=["age"],
    epochs=100,
    batch_size=50,
    num_samples=100,
    device="cpu",
)

explainer.fit(X_train)
cfs = explainer.generate(X_test.iloc[[0]])
```

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `HP` / `HS` | `50` / `50` | Hidden dimensions for the Perturber and Selector networks |
| `tau` | `0.2` | Temperature for Gumbel-Softmax relaxation |
| `epochs` | `100` | Number of training epochs |
| `batch_size` | `50` | Training batch size |
| `learning_rate` | `1e-4` | Optimizer learning rate |
| `train_L` / `test_L` | `1` / `1000` | Number of Monte Carlo samples during training / inference |
| `train_smoothing` / `test_smoothing` | `1.0` / `0.0` | Smoothing parameters for training and inference |
| `num_samples` | `100` | Number of valid counterfactuals to accumulate at inference |
| `max_iter` | `10000` | Maximum inference iterations |
| `sparsity_threshold` | `1.0` | Sparsity threshold for feature selection |
| `device` | `"cpu"` | PyTorch device (`"cpu"` or `"cuda"`) |

## Package Structure

- `src/l2c/l2c.py` — `L2C` main explainer (`fit`, `generate`)
- `src/l2c/modules.py` — PyTorch `Model` (Perturber & Selector) architecture
