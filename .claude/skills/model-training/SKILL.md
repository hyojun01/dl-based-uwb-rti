---
name: model-training
description: Use when implementing DL model components, training, or evaluation. Trigger phrases include "U-Net", "dual branch", "model training", "train", "loss function", "SSIM", "MSE", "ablation", "Tikhonov-Only", "FC-Only", "evaluate", "metrics", "PSNR", "checkpoint", "learning rate", "early stopping". Even if the user says "build the model" or "run training", use this skill.
---

# Model Training

Implement, train, and evaluate the Dual-Branch Physics-Informed U-Net and ablation models.

## Workflow

### Phase 1: Model Components
1. Read `references/model-architecture.md` for full architecture specs
2. Implement in order: tikhonov_branch.py → fc_branch.py → unet.py
3. Tikhonov branch: Π as buffer, forward = matrix multiply, **no learnable params**
4. FC branch: Linear(16→900), init weights W_fc = Π, **no activation**
5. U-Net: 2-level encoder + bottleneck + decoder with skip connections
6. Handle 30→15→7 size mismatch (use ConvTranspose2d or pad to 32×32)
7. Verify each module: forward pass with dummy data, check shapes

### Phase 2: Composite Models
1. dual_branch_unet.py: concat(Tikhonov_out, FC_out) → 2ch → U-Net → 1ch
2. tikhonov_only.py: Tikhonov_out → 1ch → U-Net → 1ch
3. fc_only.py: FC_out → 1ch → U-Net → 1ch
4. Verify param counts: proposed ~1.55M + 15.3K (FC), ablations ~1.55M each

### Phase 3: Training
1. Loss: MSE + λ·(1 - SSIM), λ = 0.1 initially
2. Optimizer: Adam, lr=1e-3
3. Scheduler: ReduceLROnPlateau(patience=10, factor=0.5)
4. Early stopping: patience=20, monitor=val_loss
5. Batch size: 64, max epochs: 200
6. Train all 3 models with **same** config and **same** data split
7. Log every run to experiments/ with full config and metrics
8. Save best checkpoint per model to checkpoints/

### Phase 4: Evaluation
1. Load best checkpoints for all 3 models
2. Evaluate on test set: MSE, PSNR, SSIM, RMSE
3. Compare across models to quantify component contributions
4. Present results table to human

## Rules

- Π is ALWAYS frozen (register_buffer, not register_parameter)
- FC branch init: W_fc = Π — this is transfer learning, not coincidence. Document it.
- All 3 models use identical training config — no per-model tuning unless explicitly approved
- Log complete experiment config before starting each training run
- If training diverges or produces NaN, stop immediately, log the state, and ask human
- SSIM computation: use pytorch_msssim or equivalent, window_size appropriate for 30×30 images
- The 30→15→7→14→28 size mismatch in U-Net decoder MUST be handled explicitly (crop or pad)
