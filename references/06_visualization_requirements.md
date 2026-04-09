## 6. Visualization Requirements

1. **Weight matrix visualization**: Show weight vector reshaped to 30×30 for a few representative links
2. **Tikhonov reconstruction visualization**: Show Π · ΔR reshaped to 30×30 for sample inputs to verify spatial prior quality
3. **Model validation plots**: RSS vs distance, RSS vs person position
4. **Training curves**: loss vs epoch for all models (proposed, Tikhonov-Only, FC-Only)
5. **Reconstruction comparison**: grid showing ground truth, Tikhonov-Only output, FC-Only output, proposed model output
6. **Quantitative comparison table**: MSE, PSNR, SSIM, RMSE for all models across different noise levels
7. **Error maps**: |predicted - ground truth| for each method
8. **Branch contribution analysis**: side-by-side visualization of Branch A (Tikhonov) and Branch B (FC) intermediate outputs before U-Net fusion

---