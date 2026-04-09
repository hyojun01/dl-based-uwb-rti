## 3. Data Generation Pipeline

### 3.1 Steps

Precomputation (once):
1. Define pixel grid (30×30) with center positions
2. Compute weight matrix `W` (16×900) using Inverse Area Elliptical Model
3. Precompute Tikhonov reconstruction matrix `Π = (W^T W + α_reg · C)^{-1} · W^T` (900×16)

For each sample:
4. Generate random ideal SLF change `Δf*_A` (place random objects)
5. Generate SLF noise `f̃_A ~ N(0, Σ_f)`, where `Σ_f(m,n) = σ_f² · exp(-D_mn / κ)`
6. Sample `σ_ε ~ U(0.3, 3.0)`, then `ε ~ N(0, σ_ε²·I)`
7. Compute `ΔR = c · W · (Δf*_A + f̃_A) + ε`

### 3.2 Input Normalization

Reference: Wu et al. (2024), Section 5.1.

Per-link (per-feature) z-score normalization is applied to `ΔR`. Since each of the 16 links has a different TX-RX geometry, the distribution of RSS differences varies across links.

Given the training set `ΔR_train ∈ R^{N_train × 16}`:

```python
# Compute statistics from the training set only
mean_per_link = ΔR_train.mean(axis=0)   # shape (16,)
std_per_link  = ΔR_train.std(axis=0)    # shape (16,)

# Apply to all splits using training statistics
ΔR_train_norm = (ΔR_train - mean_per_link) / std_per_link
ΔR_val_norm   = (ΔR_val   - mean_per_link) / std_per_link
ΔR_test_norm  = (ΔR_test  - mean_per_link) / std_per_link
```

Key rules:
- Statistics computed **only from the training set** to prevent data leakage
- Same statistics applied to validation and test sets
- Statistics saved for reuse at inference time

### 3.3 Dataset Size

- Total: 60,000 samples
- Training: 48,000 (80%)
- Validation: 6,000 (10%)
- Test: 6,000 (10%)

All models (proposed, ablation) use the same data split. The proposed model is trained end-to-end.

---
