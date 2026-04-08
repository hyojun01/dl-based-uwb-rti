---
name: data-generation
description: Use when generating training data, implementing SLF targets, spatially correlated noise, or the data pipeline. Trigger phrases include "data generation", "generate samples", "SLF image", "training data", "dataset", "target types", "spatially correlated noise", "data split", "data pipeline". Even if the user says "make the data" or "create samples", use this skill.
---

# Data Generation

Generate the 60,000-sample dataset for UWB RTI model training.

## Workflow

### Phase 1: SLF Target Generation
1. Read `references/pipeline-specs.md` and `references/forward-model-math.md` Section 2.7
2. Implement all 10 target types: person (standing/walking), table, chair, cabinet, wall, multiple objects, empty, L/T-shaped, circular
3. Random placement within 3m×3m area (0.2m margin from edges)
4. Verify: visualize samples of each target type on 30×30 grid

### Phase 2: SLF Noise
1. Implement spatially correlated Gaussian noise
2. Covariance: C_θ(k,l) = σ_θ² · exp(-D_kl / κ), κ = 0.21m
3. σ_θ ~ U(0.01, 0.05) per sample
4. Verify: noise spatial correlation structure looks correct

### Phase 3: Full Pipeline
1. For each sample: generate θ* → add noise θ̃ → θ = θ* + θ̃ → sample b, α, σ_ε → compute RSS y
2. Generate 60,000 samples total
3. Split: 48,000 train / 6,000 val / 6,000 test (fixed random seed for reproducibility)
4. Save as PyTorch tensors or numpy arrays with metadata

### Phase 4: Data Quality Check
1. Visualize random samples: SLF images, RSS vectors, Tikhonov reconstructions
2. Check RSS distribution statistics
3. Verify target type balance across dataset
4. Present summary to human for approval before proceeding

## Rules

- Use a fixed random seed for the data split — all models must use identical splits
- Each sample randomly places 1–3 objects (except empty room type)
- Object attenuation values must follow the specified uniform distributions per type
- θ = θ* + θ̃ where θ* is ideal (0 or object value) and θ̃ is spatially correlated noise
- Precompute the noise covariance matrix once (it depends only on pixel positions and κ)
- Save the W matrix and Π matrix alongside the dataset for downstream use
