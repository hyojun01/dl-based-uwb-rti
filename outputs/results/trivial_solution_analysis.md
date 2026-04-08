# Trivial Solution Analysis: Why All Models Predict Zeros

## Problem Statement

All three trained models (Proposed, Tikhonov-Only, FC-Only) produce near-zero
outputs, achieving MSE ≈ 0.008667 — identical to predicting all-zeros (0.008670).
The models have not learned to reconstruct the SLF.

## Root Cause Analysis

### 1. Extreme Class Imbalance in Targets

| Metric | Value |
|---|---|
| Nonzero pixel fraction | 2.4% |
| Mean (all pixels) | 0.0137 |
| Mean (nonzero only) | 0.581 |
| MSE(predict_all_zeros) | 0.008670 |

With 97.6% zero pixels, the MSE loss is minimized by predicting zeros everywhere.
The 2.4% nonzero pixels contribute negligibly to the total gradient.

### 2. Low Signal-to-Noise Ratio in RSS

| Component | Magnitude | % of RSS Variance |
|---|---|---|
| Bias b | ~95 dB | varies per sample |
| Path loss α·d | ~10 dB | varies per sample |
| Shadowing c·W·θ | ~0.3 dB mean | **4.82%** |
| Noise ε | ~0-3 dB | varies per sample |

The shadowing signal (which carries the spatial target information) explains only
4.82% of the RSS variance. The remaining 95% is bias, path loss, and noise — all
of which vary randomly per sample.

### 3. Branch Output Scale Mismatch

| Signal | Mean | Std | Range |
|---|---|---|---|
| Target θ* | 0.014 | 0.092 | [0, 1] |
| Tikhonov (Pi @ RSS) | 2.24 | 2.60 | [0, 14.3] |
| FC branch output | ~100s | ~100s | [-200, 150] |

The Tikhonov branch output is **163× larger** than the target mean.
The FC branch (trained) drifts to even larger scales.
The U-Net receives inputs on the wrong scale and learns to suppress to zero.

### 4. Tikhonov Reconstruction Has Inherently Low Accuracy

Even with **perfect input** (pure shadowing signal, no bias/noise):

| Input | Correlation with GT | MSE | Improvement |
|---|---|---|---|
| Pure shadowing (c·W·θ) | r = 0.20 | 0.008279 | 4.5% |
| Centered RSS | r = 0.04 | ~0.00847 | 2.3% |
| Z-scored RSS | r = 0.04 | — | — |

16 links for 900 pixels is a 56:1 underdetermined system. The Fresnel zone is
narrow (~2 pixels wide), so each link observes only ~4% of the imaging area.

## Proposed Fix

### Two Changes Required

**Change 1: Z-score normalize RSS input using training set statistics**

```python
# Computed once from training data
rss_mean = train_rss.mean(axis=0)  # (16,) per-link mean
rss_std = train_rss.std(axis=0)    # (16,) per-link std

# Applied to all data
rss_normalized = (rss - rss_mean) / rss_std
```

This brings RSS from ~85±4 dB to ~0±1, removing the dominant bias/path-loss
baseline. The remaining variation is primarily shadowing + per-sample noise.

**Change 2: Use pixel-weighted MSE loss to address class imbalance**

```python
# Weight nonzero pixels higher in the loss
weight = torch.where(target > 0, alpha, 1.0)  # alpha >> 1
loss = (weight * (pred - target) ** 2).mean()
```

With alpha ~10-20, the nonzero pixels contribute proportionally to their
importance rather than being drowned out by the 97.6% zero background.

### Why These Changes Should Work

1. **Normalization** removes the ~85 dB baseline that dominates branch outputs,
   putting inputs on a scale where shadowing differences are visible.

2. **Weighted loss** ensures the model receives meaningful gradients from the
   object pixels, preventing convergence to the trivial all-zeros solution.

3. The U-Net architecture and dual-branch design are fundamentally sound —
   the issue is entirely in input preprocessing and loss function design.

### What We Do NOT Change

- Model architectures (already verified)
- Data generation pipeline (data itself is correct)
- Tikhonov matrix Pi (correctly computed)
- Training hyperparameters (lr, batch size, scheduler)
