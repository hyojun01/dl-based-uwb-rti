## 8. Implementation Priority

1. `config.py` and `forward_model.py` (weight matrix W, Tikhonov matrix Π computation)
2. `validate_model.py` (verify forward model and Tikhonov reconstruction correctness)
3. `data_generator.py` (generate training data)
4. `unet.py`, `tikhonov_branch.py`, `fc_branch.py` (individual modules)
5. `dual_branch_unet.py` (proposed model) → train → evaluate
6. `tikhonov_only.py`, `fc_only.py` (ablation models) → train → evaluate
7. Comparative visualization and ablation analysis