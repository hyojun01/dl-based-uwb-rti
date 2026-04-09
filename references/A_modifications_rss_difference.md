## A. Modifications: Switching to RSS Difference-Based Model

### A.1 Motivation

The current project uses absolute RSS measurements `y` to estimate the absolute SLF image `θ`. Following Wu et al. (2024), we adopt an RSS difference-based approach where the difference `ΔR` between baseline (vacant environment) and current measurements is used to estimate the SLF changes `Δf_A`. This eliminates nuisance parameters (bias, path loss) and simplifies the forward model.

Reference: Wu et al., "Combining transformer with a latent variable model for radio tomography based robust device-free localization," Computer Communications, 2025.

---

### A.2 Mathematical Forward Model Changes

#### A.2.1 Previous Model (Section 2.3)

```
y = Z*b - c*W*θ - α*d + ε
```

Where:
- `b ∈ R^16`: bias vector (aggregates TX power, RX sensitivity, antenna gains)
- `α`: free space path loss exponent
- `d ∈ R^16`: log-distance vector
- `Z ∈ R^{16×16}`: identity matrix
- `W ∈ R^{16×900}`: weight matrix
- `θ ∈ R^900`: absolute SLF image
- `ε ∈ R^16`: measurement noise

#### A.2.2 New Model: RSS Difference Formulation

**Baseline measurement** (vacant environment, no target objects):

```
r̄(x', x'') = g - γ · 10·log10(d(x', x'')) - s̄(x', x'')
```

Where `s̄ = c · w^T · f̄_A` is the initial shadowing from static objects, and `f̄_A` is the baseline SLF.

**Current measurement** (with target objects present):

```
r(x', x'') = g - γ · 10·log10(d(x', x'')) - s(x', x'')
```

**RSS difference** (baseline minus current):

```
Δr(x', x'') = r̄ - r = s(x', x'') - s̄(x', x'') = c · w^T · Δf_A
```

The gain term `g`, path loss `γ · 10·log10(d)`, and static shadowing cancel out.

**Full forward model with noise** (for all 16 links):

```
ΔR = c · W · (Δf_A + f̃_A) + ε
```

Where:
- `ΔR ∈ R^16`: RSS difference vector (baseline - current)
- `W ∈ R^{16×900}`: weight matrix (unchanged)
- `Δf_A ∈ R^900`: SLF change vector (the estimation target)
- `f̃_A ∈ R^900`: zero-mean spatially correlated Gaussian noise field
- `ε ∈ R^16`: additive i.i.d. Gaussian measurement noise
- `c = 2`: scaling constant (unchanged)

#### A.2.3 Key Simplification

The following parameters are **eliminated** from the forward model:

| Removed Parameter | Previous Role |
|---|---|
| `b ∈ R^16` | Bias vector (TX power, RX sensitivity, antenna gains) |
| `α` | Free space path loss exponent |
| `d ∈ R^16` | Log-distance vector |
| `Z ∈ R^{16×16}` | Identity matrix linking bias to measurements |

The weight matrix `W`, pixel grid, elliptical weight model, and noise models remain unchanged.

---

### A.3 Parameter Range Changes

#### A.3.1 Parameters Removed

| Parameter | Previous Distribution | Status |
|---|---|---|
| Bias `b_i` | U(90, 100) per link | **Removed** (cancels in subtraction) |
| Path loss exponent `α` | U(0.9, 1.0) | **Removed** (cancels in subtraction) |

#### A.3.2 Parameters Retained (No Change)

| Parameter | Distribution/Value |
|---|---|
| Measurement noise `ε` | N(0, σ_ε² · I), σ_ε ~ U(0.3, 3.0) |
| SLF noise variance `σ_f` | U(0.01, 0.05) (or fixed at 0.05 following Wu et al.) |
| SLF spatial correlation `κ` | 0.21 m |
| Scaling constant `c` | 2 |

Note: Wu et al. use `σ_ε ∈ [0.5, 2]` for standard noise and `σ_ε = 3` for the large-noise test. The range can be adjusted based on experimental requirements.

---

### A.4 SLF Target Definition Changes

#### A.4.1 Previous Definition

```
θ = θ* + θ̃
```
- `θ*_k = 0` for free space
- `θ*_k ~ U(0.3, 1.0)` for object regions
- `θ̃`: spatially correlated Gaussian noise

#### A.4.2 New Definition

```
Δf_A = Δf*_A + f̃_A
```
- `Δf*_A_k = 0` for free space (no change from baseline)
- `Δf*_A_k ~ U(0.3, 1.0)` for object regions (same value range)
- `f̃_A`: spatially correlated Gaussian noise (same covariance model)

The object generation logic (types, sizes, placement from Section 2.7) remains **unchanged**. The only conceptual change is that the SLF now represents *changes* due to target objects rather than absolute attenuation.

---

### A.5 Data Generation Pipeline Changes

#### A.5.1 Previous Pipeline (Section 3.1)

```
For each sample:
  1. Generate θ* (place random objects)
  2. Generate θ̃ (spatially correlated noise)
  3. θ = θ* + θ̃
  4. Sample b ~ U(90, 100)^16
  5. Sample α ~ U(0.9, 1.0)
  6. Sample σ_ε ~ U(0.3, 3.0), then ε ~ N(0, σ_ε²·I)
  7. Compute y = Z*b - c*W*θ - α*d + ε
```

#### A.5.2 New Pipeline

```
Precomputation (once):
  1. Define pixel grid (30×30), compute pixel center positions
  2. Compute weight matrix W (16×900) using Inverse Area Elliptical Model
  3. Precompute Tikhonov matrix Π = (W^T·W + α_reg·C)^{-1} · W^T

For each sample:
  1. Generate Δf*_A (place random objects, same logic as before)
  2. Generate f̃_A ~ N(0, Σ_f), where Σ_f(m,n) = σ_f² · exp(-D_mn / κ)
  3. Sample σ_ε ~ U(0.3, 3.0), then ε ~ N(0, σ_ε²·I)
  4. Compute ΔR = c · W · (Δf*_A + f̃_A) + ε
```

**Removed steps:** Sampling `b`, sampling `α`, computing `Z*b`, computing `α*d`.

#### A.5.3 Input Normalization

Reference: Wu et al. (2024), Section 5.1 — "the z-score normalization is applied to the RSS data **r**."

We apply **per-link (per-feature) z-score normalization** to the RSS difference data `ΔR`. Since each of the 16 links has a different TX-RX geometry (distance, orientation), the distribution of RSS differences varies across links. Per-link normalization corrects for these scale differences.

**Computation:**

Given the training set `ΔR_train ∈ R^{N_train × 16}`:

```python
# Compute statistics from the training set only
mean_per_link = ΔR_train.mean(axis=0)   # shape (16,), mean per link across all samples
std_per_link  = ΔR_train.std(axis=0)    # shape (16,), std per link across all samples

# Apply to all splits using training statistics
ΔR_train_norm = (ΔR_train - mean_per_link) / std_per_link   # (N_train, 16)
ΔR_val_norm   = (ΔR_val   - mean_per_link) / std_per_link   # (N_val, 16)
ΔR_test_norm  = (ΔR_test  - mean_per_link) / std_per_link   # (N_test, 16)
```

**Key rules:**
- `mean_per_link` and `std_per_link` are computed **only from the training set** to prevent data leakage.
- The same statistics are applied to validation and test sets.
- These statistics should be saved and reused at inference time for deployment.

---

### A.6 Deep Learning Model Changes

#### A.6.1 Branch A: Tikhonov Reconstruction

The Tikhonov matrix formula is unchanged:
```
Π = (W^T · W + α_reg · C)^{-1} · W^T    ∈ R^{900×16}
```

The input changes from `y` to `ΔR`:
```
Previous: θ_tik = Π · y
New:      Δf_tik = Π · ΔR
```

Since `b`, `α`, and `d` no longer appear in the measurement model, the Tikhonov reconstruction directly targets `Δf_A` without interference from nuisance parameters. This is expected to produce a cleaner initial estimate.

#### A.6.2 Branch B: Fully Connected Layer

No structural change. Input dimension remains 16, output dimension remains 900.
```
Previous: θ_fc = W_fc · y + b_fc
New:      Δf_fc = W_fc · ΔR + b_fc
```

Initialization of `W_fc` from Π remains valid.

#### A.6.3 U-Net Refinement Network

No architectural change required. Input is still 2×30×30 (concatenation of Branch A and B outputs). Output is 1×30×30.

#### A.6.4 Loss Function

The existing regression loss remains unchanged:
- Loss: MSE + λ·(1 - SSIM)
- Output activation: linear

#### A.6.5 Training Configuration

All training hyperparameters (optimizer, scheduler, early stopping, batch size, epochs, data split) remain unchanged.

---

### A.7 Summary of All Changes

| Component | Change Type | Details |
|---|---|---|
| Forward model | **Modified** | `ΔR = c·W·(Δf_A + f̃_A) + ε` (simplified) |
| Bias `b`, path loss `α`, distance `d` | **Removed** | Cancel in RSS difference |
| Weight matrix `W` | Unchanged | Same Inverse Area Elliptical Model |
| SLF noise model `f̃_A` | Unchanged | Same covariance structure |
| Measurement noise `ε` | Unchanged | Same i.i.d. Gaussian |
| Object generation | Unchanged | Same types, sizes, placement |
| Data generation loop | **Simplified** | Fewer sampling steps |
| Network input | **Changed** | `ΔR ∈ R^16` instead of `y ∈ R^16` |
| Network output target | **Changed** | `Δf_A` (SLF change) instead of `θ` (absolute SLF) |
| Tikhonov branch | Unchanged structure | Input changes to `ΔR` |
| FC branch | Unchanged structure | Input changes to `ΔR` |
| U-Net | Unchanged | Same architecture |
| Loss function | Unchanged | MSE + SSIM (regression) |
| Input normalization | **Changed** | Z-score on `ΔR` |

---

### A.8 Practical Implications

1. **Deployment requirement**: A baseline RSS measurement of the vacant environment must be collected before the system can operate. This is a one-time calibration step.

2. **Robustness benefit**: Eliminating bias and path loss parameters removes major sources of model mismatch, as these vary across hardware and environments.

3. **Limitation**: The method assumes the static environment (walls, furniture not being tracked) remains unchanged between baseline and current measurements. Long-term environmental drift may require periodic baseline recalibration.

---
