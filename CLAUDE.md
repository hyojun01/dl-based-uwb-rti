# UWB RTI: Deep Learning-Based Radio Tomographic Imaging

## Purpose

Implement a complete IEEE 802.15.4z HRP UWB RTI system: mathematical forward model, training data generation, Dual-Branch Physics-Informed U-Net, ablation models, training, evaluation, and visualization. All code in Python.

## Core workflow

1. Read `experiment-state.json` to determine current phase and context
2. Work on the current phase's next incomplete task from `progress.json`
3. Write code incrementally — implement one module, then verify before moving on
4. After each milestone, update `experiment-state.json` with results and `progress.json` with completion status
5. At phase transitions, stop and present results for human review before proceeding

## Phases (sequential)

| Phase | Modules | Skill |
|-------|---------|-------|
| 1. Forward model | `config.py`, `forward_model.py` | `forward-model` |
| 2. Model validation | `validate_model.py` | `forward-model` |
| 3. Data generation | `data_generator.py` | `data-generation` |
| 4. DL components | `models/*.py` | `model-training` |
| 5. Training | `train.py` | `model-training` |
| 6. Evaluation | `evaluate.py` | `model-training` |
| 7. Visualization | `visualize.py` | `visualization` |

## Rules (always apply)

- Never overwrite `experiment-state.json` without first reading and merging the current state
- Never skip model validation (Phase 2) — forward model correctness is prerequisite for all downstream work
- All models (proposed, Tikhonov-Only, FC-Only) must use the **same** data split and training configuration
- The Tikhonov matrix Π is precomputed once and frozen during training — never make it learnable
- FC branch weight initialization: W_fc = Π (transfer learning from Tikhonov matrix)
- When training fails or produces unexpected results, log the full configuration to `experiments/` before changing parameters
- Ask the human before: changing hyperparameters, moving to next phase, re-generating data, modifying model architecture

## Output conventions

- Source code: `uwb_rti/` (follows structure in `references/07_file_structure.md`)
- Experiment logs: `experiments/exp_{NNN}.json` (parameters, metrics, notes)
- Model checkpoints: `checkpoints/{model_name}_best.pt`
- Figures: `outputs/figures/`
- Final results: `outputs/results/`

## Context management

- `experiment-state.json`: current phase, active parameters, last results, decisions made
- `progress.json`: task-level tracking with verification status
- Before long computations (training), save state so session can resume
- On session start: always read both JSON files first

## Key constants (do not change without human approval)

- Imaging area: 3m × 3m, resolution: 30×30 (K=900)
- TX: 4 tags at y=0, RX: 4 anchors at y=3, links: N=16
- UWB CH9: f_center ≈ 7.9872 GHz, BW ≈ 499.2 MHz
- Dataset: 60,000 total (48k train / 6k val / 6k test)
