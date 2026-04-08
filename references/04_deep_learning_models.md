## 4. Deep Learning Models

### 4.1 Dual-Branch Physics-Informed U-Net (Proposed Model)

Input: 16-dimensional RSS vector y
Output: 30×30×1 SLF image θ_hat

Overall architecture:
```
y ∈ R^16
  ├── Branch A (Tikhonov): Π · y → reshape(1×30×30)     [fixed, no learnable params]
  ├── Branch B (FC):       FC(y) → reshape(1×30×30)     [learnable]
  └── Concat → (2×30×30) → U-Net → θ_hat (1×30×30)
```

#### 4.1.1 Branch A: Tikhonov Reconstruction (Fixed)

Precomputation:
```python
# Offline (performed once before training)
C = np.eye(K)                                    # Zero-order Tikhonov (C = I_900)
Pi = np.linalg.solve(W.T @ W + alpha * C, W.T)   # Π ∈ R^{900×16}
Pi_tensor = torch.from_numpy(Pi).float()          # Transfer to GPU and freeze
```

Forward pass:
```
θ_tik = Π · y ∈ R^900 → reshape → (1×30×30)
```

Parameters: None (registered as buffer)
Regularization parameter α: subject to hyperparameter search

#### 4.1.2 Branch B: Fully Connected Layer (Learnable)

Architecture:
```
Input(16) → FC(900) → reshape → (1×30×30)
```

- W_fc ∈ R^{900×16}, b_fc ∈ R^{900}: learnable parameters
- Initialization: W_fc = Π (transfer learning from Tikhonov matrix)
- No activation function (linear transformation)

Number of parameters: 900 × 16 + 900 = 15,300

#### 4.1.3 U-Net Refinement Network

Input: 2×30×30 (channel-wise concatenation of Branch A and B outputs)
Output: 1×30×30

Architecture:
```
Encoder:
  Level 1: Conv2d(2→64, 3×3, padding=1) → BN → ReLU
         → Conv2d(64→64, 3×3, padding=1) → BN → ReLU
         → MaxPool2d(2×2)                                  # → (64×15×15)

  Level 2: Conv2d(64→128, 3×3, padding=1) → BN → ReLU
         → Conv2d(128→128, 3×3, padding=1) → BN → ReLU
         → MaxPool2d(2×2)                                  # → (128×7×7)

Bottleneck:
  Conv2d(128→256, 3×3, padding=1) → BN → ReLU
  Conv2d(256→256, 3×3, padding=1) → BN → ReLU             # → (256×7×7)

Decoder:
  Level 2: Upsample(scale=2) → Concat(skip2)               # → (384×15×15) [*]
         → Conv2d(384→128, 3×3, padding=1) → BN → ReLU
         → Conv2d(128→128, 3×3, padding=1) → BN → ReLU

  Level 1: Upsample(scale=2) → Concat(skip1)               # → (192×30×30)
         → Conv2d(192→64, 3×3, padding=1) → BN → ReLU
         → Conv2d(64→64, 3×3, padding=1) → BN → ReLU

Output: Conv2d(64→1, 1×1) → Linear activation              # → (1×30×30)
```

[*] 7×7 → Upsample(2) → 14×14: requires padding/crop to align with 15×15.
    Alternatively, use ConvTranspose2d(128, 128, kernel=2, stride=2) for exact size control.

Alternative resolution design:
- 30×30 → 15×15 → 7×7 introduces non-integer scaling issues
- **Recommended**: use stride-2 convolutions instead of MaxPool, or zero-pad input to 32×32 and crop output

Approximate parameter count:
| Layer | Parameters |
|---|---|
| Encoder Level 1 | 2×(64×2×3×3 + 64) + 2×(64×64×3×3 + 64) ≈ 75K |
| Encoder Level 2 | 2×(128×64×3×3 + 128) ≈ 148K |
| Bottleneck | 2×(256×128×3×3 + 256) ≈ 590K |
| Decoder Level 2 | (128×384×3×3 + 128) + (128×128×3×3 + 128) ≈ 591K |
| Decoder Level 1 | (64×192×3×3 + 64) + (64×64×3×3 + 64) ≈ 148K |
| Output Conv | 1×64×1×1 + 1 = 65 |
| **Total** | **~1.55M** |

---

### 4.2 Tikhonov-Only + U-Net (Ablation)

Ablation model using only Branch A to verify the contribution of the FC branch in the proposed model.

Architecture:
```
y ∈ R^16
  └── Π · y → reshape(1×30×30) → U-Net → θ_hat (1×30×30)
```

U-Net architecture is identical to Section 4.1.3 except the input channel is changed to 1:
- Encoder Level 1 first Conv: Conv2d(1→64, 3×3)

---

### 4.3 FC-Only + U-Net (Ablation)

Ablation model using only Branch B to verify the contribution of the Tikhonov branch in the proposed model.

Reference: Oral et al. (2023), DeepFC.

Architecture:
```
y ∈ R^16
  └── FC(900) → reshape(1×30×30) → U-Net → θ_hat (1×30×30)
```

U-Net architecture is identical to Section 4.1.3 except the input channel is changed to 1.

---

### 4.4 Training Configuration

#### 4.4.1 Common Settings

| Item | Value |
|---|---|
| Loss | MSE + λ · (1 - SSIM), λ = 0.1 (initial) |
| Optimizer | Adam, lr=1e-3 |
| Scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Early stopping | patience=20, monitor=val_loss |
| Batch size | 64 |
| Max epochs | 200 |

#### 4.4.2 Data Split

| Purpose | Samples | Ratio |
|---|---|---|
| Training | 48,000 | 80% |
| Validation | 6,000 | 10% |
| Test | 6,000 | 10% |

All models are trained end-to-end using the **same** data split.

#### 4.4.3 Per-Model Training Details

| Model | Training | Notes |
|---|---|---|
| Proposed (4.1) | End-to-end, MSE + SSIM | Π frozen, FC + U-Net trained |
| Tikhonov-Only (4.2) | End-to-end, MSE + SSIM | Π frozen, U-Net only trained |
| FC-Only (4.3) | End-to-end, MSE + SSIM | FC + U-Net trained |

---

### 4.5 Evaluation Metrics

| Metric | Definition | Purpose |
|---|---|---|
| MSE | (1/K) · \|\|θ_hat - θ*\|\|² | Per-pixel reconstruction error |
| PSNR | 10 · log10(max² / MSE) | Signal-to-noise ratio |
| SSIM | Structural Similarity Index | Structural similarity |
| RMSE | sqrt(MSE) | Error in original scale |

All models are evaluated on the same held-out test set. Cross-model comparison quantifies the contribution of each component (Tikhonov branch, FC branch, U-Net).

---