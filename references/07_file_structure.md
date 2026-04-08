## 7. File Structure

```
uwb_rti/
├── config.py                  # All constants and parameters
├── forward_model.py           # Weight matrix, RSS generation, Tikhonov matrix (Π) computation
├── data_generator.py          # Training data generation
├── models/
│   ├── tikhonov_branch.py     # Fixed Tikhonov reconstruction branch (Π · y)
│   ├── fc_branch.py           # Learnable FC branch
│   ├── unet.py                # U-Net refinement network
│   ├── dual_branch_unet.py    # Proposed model (Tikhonov + FC → U-Net)
│   ├── tikhonov_only.py       # Ablation: Tikhonov-Only + U-Net
│   └── fc_only.py             # Ablation: FC-Only + U-Net
├── train.py                   # Training script for all models
├── evaluate.py                # Evaluation and metrics (MSE, PSNR, SSIM, RMSE)
├── validate_model.py          # Mathematical model validation (Section 5)
├── visualize.py               # All visualization functions
└── main.py                    # Main execution pipeline
```

---