---
globs: ["models/*.py", "models/**/*.py"]
---

# Model Architecture Rules

## PyTorch Conventions
- All models inherit from `nn.Module`
- Tikhonov matrix Π: always `register_buffer`, never `nn.Parameter`
- FC branch init: `self.fc.weight.data.copy_(Pi_tensor)` — document this is transfer learning from Tikhonov

## Dimensional Safety
- Assert input shape in forward(): RSS vector must be (batch, 16)
- Assert output shape: must be (batch, 1, 30, 30)
- U-Net skip connections: explicitly handle 30→15→7 size mismatch with padding or ConvTranspose2d

## Ablation Consistency
- tikhonov_only.py and fc_only.py must use the SAME U-Net class as dual_branch_unet.py
- Only the input channel count changes (2→1)
- Do not create separate U-Net implementations per model
