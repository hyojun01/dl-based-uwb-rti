---
name: forward-model
description: Use when implementing or debugging the UWB RTI forward model, weight matrix W, Tikhonov matrix Π, RSS computation, or model validation (Sections 1-2, 5). Trigger phrases include "forward model", "weight matrix", "Tikhonov", "RSS", "path loss", "elliptical model", "Fresnel zone", "validate model", "RSS vs distance", "shadowing". Even if the user just says "fix the weights" or "check the RSS formula", use this skill.
---

# Forward Model

Implement and validate the UWB RTI mathematical forward model per IEEE 802.15.4z HRP UWB specifications.

## Workflow

### Phase 1: Config & Constants
1. Read `references/experimental-setup.md` for all physical parameters
2. Implement `config.py` with node positions, pixel grid, UWB constants
3. Verify: pixel centers computed correctly, distances between all TX-RX pairs

### Phase 2: Weight Matrix W
1. Read `references/forward-model-math.md` for the Inverse Area Elliptical Model
2. Implement weight computation: β(s), β_min, β_max (Fresnel zone), area-based weighting
3. Build W ∈ R^{16×900}
4. Verify: visualize weight vectors reshaped to 30×30 — should show elliptical patterns

### Phase 3: Tikhonov Matrix Π
1. Compute Π = (W^T W + α·I)^{-1} · W^T ∈ R^{900×16}
2. Register as fixed buffer (never trainable)
3. Verify: apply Π to synthetic RSS, check spatial reconstruction quality

### Phase 4: RSS Generation
1. Implement: y = Z·b - c·W·θ - α·d + ε
2. Sample parameters from specified distributions (see references)
3. Verify: dimensions, signal levels, noise characteristics

### Phase 5: Model Validation
1. RSS vs Distance: no shadowing, vary D 0.5m–5m, expect log decay
2. Human Crossing: person traverses LOS, expect RSS dip at x=0
3. Both validations must pass before proceeding to data generation

## Rules

- Weight model uses **Inverse Area Elliptical Model** (Hamilton et al. 2014), NOT the simple elliptical model
- β_min = λ/20 (small positive value, not zero — division by zero guard)
- β_max = sqrt(λ · d_{n,m} / 4) — first Fresnel zone semi-minor axis
- λ = c_light / f_center ≈ 0.03755 m (CH9)
- Scaling constant c = 2 (always)
- The Tikhonov regularization parameter α is a hyperparameter — document the chosen value in experiment-state.json
