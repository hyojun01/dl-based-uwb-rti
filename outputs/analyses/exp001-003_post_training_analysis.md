# Post-Training Analysis: Experiments 001-003

**Date**: 2026-04-09
**Experiments**: exp_001 (proposed), exp_002 (tikhonov_only), exp_003 (fc_only)
**Status**: All phases (1-7) complete. Results require investigation before next iteration.

---

## 1. Training Summary

All three models were trained with identical configuration (Adam lr=1e-3, batch=64, weighted MSE + 0.1*(1-SSIM), early stopping patience=20) on the same 48k/6k/6k split.

| Model | Trainable Params | Best Val Loss | Best Epoch | Total Epochs |
|-------|-----------------|--------------|-----------|-------------|
| Proposed (DualBranch) | 1,880,133 | 0.136589 | 9 | 29 |
| Tikhonov-Only + U-Net | 1,864,257 | 0.136657 | 22 | 42 |
| FC-Only + U-Net | 1,879,557 | 0.136014 | 20 | 40 |

### Test Set Evaluation

| Model | MSE | PSNR (dB) | SSIM | RMSE |
|-------|-----|-----------|------|------|
| Proposed | 0.023087 | 18.26 | 0.5501 | 0.1519 |
| Tikhonov-Only | 0.019005 | 19.99 | 0.6187 | 0.1379 |
| FC-Only | 0.019350 | 19.51 | 0.6069 | 0.1391 |

**Unexpected result**: The proposed dual-branch model performs worst on all metrics. Both single-branch ablations outperform it.

---

## 2. Identified Issues

### 2.1 Low Signal-to-Noise Ratio (Mean SNR = 0.34)

The RSS difference vector is dominated by noise. Across all 16 links, signal accounts for only **13% of total variance** — the remaining 82% is measurement noise, with a negligible cross-term.

| Component | Variance | Share |
|-----------|---------|-------|
| Signal (c * W * delta_f_star) | 0.731 | 17.8% |
| Noise (c * W * f_tilde + epsilon) | 3.379 | 82.2% |
| Total | 4.111 | 100% |

Per-link SNR breakdown:

| Link Geometry | Links | SNR | Signal % of Variance |
|--------------|-------|-----|---------------------|
| Edge (d=3.0m, boundary) | 0, 15 | 0.000 | 0.0% |
| Center (d=3.0m, interior) | 5, 10 | 0.118 | 1.4% |
| Off-center (d=3.61m) | 2, 7, 8, 13 | 0.153 | 2.3% |
| Off-center (d=3.16m) | 1, 4, 6, 9, 11, 14 | 0.45 | 16.5% |
| Diagonal (d=4.24m) | 3, 12 | 0.83 | 41.2% |

The dominant noise source is measurement noise epsilon ~ N(0, sigma_eps^2), where sigma_eps ~ U(0.3, 3.0). Expected variance E[sigma_eps^2] = 3.33, which accounts for >98% of all noise on every link.

### 2.2 Dead Edge Links (Zero Signal)

Links 0 (TX0-RX0) and 15 (TX3-RX3) carry **exactly zero object signal**. Their LOS paths run along x=0 and x=3 respectively — the imaging area boundaries. The Fresnel zone semi-minor axis is beta_max = 0.168m, so non-zero weight pixels exist only at x=[0.05, 0.15] or x=[2.85, 2.95].

However, the SLF edge margin (0.2m) combined with minimum object half-width (~0.175m) means object centers must be placed at x >= 0.375m. **Objects can never overlap with edge link Fresnel zones.** These 2 of 16 inputs are pure noise that the model must learn to ignore.

### 2.3 Weight Matrix Imbalance (21.7x)

The weight matrix row sums vary dramatically by link geometry:

| Group | Row Sum | Non-Zero Pixels | Relative Sensitivity |
|-------|---------|----------------|---------------------|
| Edge | 1.17 | 42 | 1.0x (baseline) |
| Center | 2.34 | 84 | 2.0x |
| Off-center (d=3.16m) | 12.78 | 84 | 10.9x |
| Off-center (d=3.61m) | 3.65 | 102 | 3.1x |
| Diagonal | 25.38 | 132 | 21.7x |

This imbalance means diagonal links dominate the forward model output. A person near the diagonal path produces ~22x more RSS change than one near the center vertical path. After z-score normalization, the imbalance is partially reduced (std range narrows from [1.81, 2.43]) but still present.

### 2.4 Empty Rooms Indistinguishable from Occupied

Due to the low SNR, the RSS difference distributions for empty and non-empty rooms overlap almost completely:

| Metric | Empty (no objects) | Non-Empty (with objects) |
|--------|-------------------|-------------------------|
| Mean |delta_R| per sample | 1.33 +/- 0.66 | 1.50 +/- 0.66 |
| Median |delta_R| | 0.94 | 1.07 |
| P95 |delta_R| | 3.88 | 4.35 |

The distributions overlap approximately 95%. For any individual sample, it is nearly impossible to determine from the RSS vector alone whether objects are present.

### 2.5 DL Models Worse Than Raw Tikhonov Baseline

Decomposing MSE into object and background pixel contributions reveals the mechanism:

| Model | Total MSE | Object MSE | Background MSE |
|-------|----------|-----------|---------------|
| Normalized Tikhonov (Pi @ delta_R_norm) | **0.0100** | 0.352 | **0.002** |
| Raw Tikhonov (Pi @ delta_R) | 0.0137 | 0.331 | 0.006 |
| Tikhonov-Only + U-Net | 0.0190 | 0.182 | 0.015 |
| FC-Only + U-Net | 0.0194 | 0.179 | 0.015 |
| Proposed (DualBranch) | 0.0231 | **0.159** | 0.020 |

The DL models achieve **2x better object pixel MSE** but **8-10x worse background pixel MSE** compared to the Tikhonov baselines. Since 97.6% of pixels are background, the total MSE is dominated by false positive activations on background regions.

**Root cause**: The training loss uses 20x object pixel weighting. This makes object MSE roughly equally important as background MSE during optimization (0.48 * obj_MSE vs 0.976 * bg_MSE). The model learns to predict nonzero values broadly to avoid missing objects, at the cost of background accuracy. Evaluation uses standard MSE where background dominates.

### 2.6 Proposed Model Branch Redundancy

At initialization, both branches produce identical output (FC weights initialized from Pi). The U-Net receives two identical channels — completely redundant information. During training:

- The FC weight diverged by ||W_fc - Pi|| / ||Pi|| = 8.1 in 9 epochs (proposed)
- The FC-Only model diverged 12.3 in 20 epochs (1.5x more)

The proposed model early-stopped at epoch 9 because val loss plateaued while the U-Net was still learning to merge two gradually diverging channels. The ablation models, receiving a single stable channel, converge more efficiently.

---

## 3. Root Cause Hierarchy

```
1. [FUNDAMENTAL] Low SNR — noise is 5x signal
   |
   +-- Measurement noise range sigma_eps ~ U(0.3, 3.0) is too wide
   |   Expected E[sigma_eps^2] = 3.33, dominates all signal
   |
   +-- Small forward model signal magnitudes
   |   Inverse-area elliptical model with pixel_area weighting
   |   produces RSS differences of 0.3-4.7 dB (real UWB: 3-15 dB)
   |
   +-- 2 dead links (edge) contribute zero signal, only noise
   |
   +----> Empty and occupied rooms nearly indistinguishable

2. [STRUCTURAL] Weight matrix 21.7x imbalance
   |
   +-- Edge Fresnel zones clipped by imaging area boundary
   +-- Object margin prevents any overlap with edge link zones
   +-- Information concentrated in 2 diagonal links (41% of signal variance)
   |
   +----> 16-dim input has highly unequal information density

3. [TRAINING] Loss function / evaluation mismatch
   |
   +-- 20x object weighting causes background hallucinations
   +-- DL models sacrifice bg accuracy for obj accuracy
   +-- Evaluated on standard MSE where bg (97.6%) dominates
   |
   +----> All DL models perform WORSE than raw Tikhonov on total MSE

4. [MODEL-SPECIFIC] Proposed model branch redundancy
   |
   +-- Both branches identical at initialization (FC = Pi)
   +-- U-Net must learn channel merging + spatial refinement simultaneously
   +-- Early stops at epoch 9 before FC meaningfully diverges
   |
   +----> Proposed model is worst among all three
```

---

## 4. Future Directions

### 4.1 Noise Level Reduction (Addresses Root Cause 1)

The current sigma_eps ~ U(0.3, 3.0) produces noise variance E[sigma_eps^2] = 3.33, which overwhelms the signal on all links. Options:

- **Narrow the noise range** to U(0.3, 1.0) or U(0.5, 2.0) following Wu et al.'s standard-noise setting
- **Use a fixed sigma_eps** (e.g., 0.5 or 1.0) to reduce variance
- Separately validate with a high-noise test set (sigma_eps = 3.0) for robustness evaluation

### 4.2 Weight Matrix Normalization (Addresses Root Cause 2)

The 21.7x row sum imbalance means some links are far more informative than others. Options:

- **Row-normalize W** so each link contributes equally (W_norm[n] = W[n] / ||W[n]||_1), then recompute Pi
- **Remove dead edge links** from the input (reduce from 16 to 14 links)
- **Per-link signal weighting** in the loss function based on expected SNR

### 4.3 Loss Function Alignment (Addresses Root Cause 3)

The mismatch between training loss (20x object weight) and evaluation (standard MSE) causes all DL models to perform worse than the Tikhonov baseline on total MSE.

- **Reduce object_weight** from 20.0 to a moderate value (e.g., 3-5)
- **Use standard MSE** during training (no per-pixel weighting), relying on the SSIM component to provide structural sensitivity
- **Evaluate with the same weighted metric** used during training, for consistent comparison

### 4.4 Training Duration for Proposed Model (Addresses Root Cause 4)

The proposed model early-stopped at epoch 9 before branches could differentiate.

- **Increase max_epochs** and/or early stopping patience
- **Warm up the FC branch** for several epochs before unfreezing U-Net (staged training)
- **Initialize FC differently** from Pi (e.g., Pi + small random perturbation) so channels differ from the start

### 4.5 Forward Model Signal Magnitude (Addresses Root Cause 1, long-term)

The inverse-area elliptical model produces RSS differences (0.3-4.7 dB) below real-world UWB measurements (3-15 dB). After completing iterative improvements above, revisit:

- Whether the weight formula needs a different normalization
- Whether the pixel_area factor in the discrete weight is appropriately scaled
- Comparison against measured RSS data if available

### Priority Order

The issues are ordered by expected impact. Addressing 4.1 (noise reduction) and 4.3 (loss alignment) first would likely yield the largest improvements with minimal code changes.
