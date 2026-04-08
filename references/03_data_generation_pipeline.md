## 3. Data Generation Pipeline

### 3.1 Steps

1. Define pixel grid (30×30) with center positions
2. Compute distance vector `d` for all 16 links
3. Compute weight matrix `W` (16×900) using Inverse Area Elliptical Model
4. Precompute Tikhonov reconstruction matrix `Π = (W^T W + α_reg · C)^{-1} · W^T` (900×16)
5. For each training sample:
   a. Generate random ideal SLF image θ* (place random objects)
   b. Generate SLF noise θ̃ from spatially correlated Gaussian
   c. θ = θ* + θ̃
   d. Sample b, α, σ_ε
   e. Compute y = Z*b - c*W*θ - α*d + ε

### 3.2 Dataset Size

- Total: 60,000 samples
- Training: 48,000 (80%)
- Validation: 6,000 (10%)
- Test: 6,000 (10%)

All models (proposed, ablation) use the same data split. The proposed model is trained end-to-end.

---