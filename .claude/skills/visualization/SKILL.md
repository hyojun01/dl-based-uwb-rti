---
name: visualization
description: Use when creating plots, figures, or visual outputs for the UWB RTI project. Trigger phrases include "plot", "visualize", "figure", "graph", "show me", "training curve", "reconstruction comparison", "error map", "weight matrix plot", "heatmap". Even if the user says "draw it" or "let me see", use this skill.
---

# Visualization

Create publication-quality figures for the UWB RTI project.

## Workflow

1. Read `references/viz-requirements.md` for the 8 required visualization types
2. Use matplotlib with consistent styling (see conventions below)
3. Save all figures to `outputs/figures/` with descriptive filenames
4. Present figures to human with brief interpretation

## Required Figures

1. **Weight matrix** — W rows reshaped to 30×30, colormap showing elliptical patterns
2. **Tikhonov reconstruction** — Π·y for sample inputs, verify spatial prior
3. **Model validation** — RSS vs distance curve, RSS vs person position
4. **Training curves** — loss vs epoch, all 3 models overlaid
5. **Reconstruction grid** — ground truth | Tikhonov-Only | FC-Only | Proposed
6. **Quantitative table** — MSE, PSNR, SSIM, RMSE across models and noise levels
7. **Error maps** — |predicted - ground truth| for each method
8. **Branch analysis** — Branch A and B intermediate outputs before U-Net fusion

## Conventions

- Colormap for SLF images: `viridis` (sequential) or `RdBu_r` (diverging for errors)
- Figure size: single column = (6, 4), grid = (12, 8)
- Font size: labels 12pt, titles 14pt, tick labels 10pt
- DPI: 300 for saved figures
- Always include colorbars for heatmaps with units
- Model names in legends: "Proposed (Dual-Branch)", "Tikhonov-Only", "FC-Only"

## Rules

- All comparison figures must show all 3 models side-by-side
- Use the same colorbar range across models in comparison figures
- Include the 3m×3m physical scale on SLF image axes
- Training curves must show both train and validation loss
