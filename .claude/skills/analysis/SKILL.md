---
name: analysis
description: Use when analyzing experiment results, comparing models, interpreting metrics, debugging unexpected outputs, or performing ablation analysis. Trigger phrases include "analyze results", "compare models", "why is the loss", "debug training", "interpret", "ablation analysis", "what went wrong", "check metrics". Even if the user says "look at the numbers" or "explain these results", use this skill.
---

# Analysis

Analyze and interpret experiment results for the UWB RTI project.

## Workflow

1. Read `experiment-state.json` for latest results and `experiments/` for run history
2. Identify what is being analyzed (training curves, metrics, reconstructions, etc.)
3. Perform quantitative analysis with appropriate statistics
4. Present findings with clear explanations and recommendations
5. If results are unexpected, propose diagnostic steps before changing parameters

## Analysis Checklist

### Training Analysis
- Loss convergence: is it still decreasing? Overfitting (val > train gap)?
- Learning rate schedule: did reductions help or plateau?
- Gradient health: NaN, explosion, vanishing?
- Compare across models: does the proposed model outperform ablations?

### Reconstruction Quality
- MSE: per-pixel error magnitude
- PSNR: signal quality (higher = better, typical good: >25 dB)
- SSIM: structural fidelity (higher = better, target: >0.8)
- Error maps: where does each model fail?

### Component Contribution
- Proposed vs Tikhonov-Only: quantifies FC branch contribution
- Proposed vs FC-Only: quantifies Tikhonov branch contribution
- Branch intermediate outputs: what does each branch capture?

## Rules

- Always compare against ALL models, never evaluate one in isolation
- Present both absolute metrics and relative improvements
- If metrics are poor, diagnose before suggesting hyperparameter changes
- Log analysis findings to experiment-state.json decisions_log
