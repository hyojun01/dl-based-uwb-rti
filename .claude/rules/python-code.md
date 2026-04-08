---
globs: ["**/*.py"]
---

# Python Code Rules

## Style
- Use type hints for all function signatures
- Docstrings for all public functions (one-line for simple, numpy-style for complex)
- Constants in UPPER_SNAKE_CASE, defined in config.py only

## Scientific Computing
- Use numpy for matrix operations, torch for GPU tensors
- Never mix numpy and torch operations without explicit conversion
- Use float32 for training, float64 for forward model precision
- Always set random seeds: `torch.manual_seed()`, `np.random.seed()`, `random.seed()`

## Reproducibility
- Never use non-deterministic operations without documenting them
- Save all hyperparameters alongside model checkpoints
- Data split must use a fixed seed defined in config.py
