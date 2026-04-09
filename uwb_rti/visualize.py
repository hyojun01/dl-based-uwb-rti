"""Visualization module for UWB RTI experiment results.

Generates training curves, reconstruction comparisons, error maps,
and branch contribution analysis.
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uwb_rti.config import (
    N_PIXELS_X, N_PIXELS_Y, N_LINKS, AREA_WIDTH, AREA_HEIGHT,
    BATCH_SIZE,
)
from uwb_rti.models.dual_branch_unet import DualBranchUNet
from uwb_rti.models.tikhonov_only import TikhonovOnlyUNet
from uwb_rti.models.fc_only import FCOnlyUNet

FIGURES_DIR = "outputs/figures"


# ── 7.1 Training Curves ───────────────────────────────────────────────────

def plot_training_curves(exp_dir: str = "experiments", save: bool = True) -> None:
    """Plot loss vs epoch for all 3 models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    model_names = ["proposed", "tikhonov_only", "fc_only"]
    colors = {"train": "tab:blue", "val": "tab:orange"}

    for ax, name in zip(axes, model_names):
        # Find the experiment file for this model
        for p in sorted(Path(exp_dir).glob("exp_*.json")):
            with open(p) as f:
                exp = json.load(f)
            if exp.get("model_name") == name and "history" in exp:
                hist = exp["history"]
                epochs = range(1, len(hist["train_loss"]) + 1)
                ax.plot(epochs, hist["train_loss"], color=colors["train"],
                        label="Train", alpha=0.8)
                ax.plot(epochs, hist["val_loss"], color=colors["val"],
                        label="Val", alpha=0.8)

                best_epoch = exp["results"]["best_epoch"]
                best_val = exp["results"]["best_val_loss"]
                ax.axvline(x=best_epoch, color="red", linestyle="--",
                           alpha=0.5, label=f"Best @ {best_epoch}")

                # Secondary axis for LR
                ax2 = ax.twinx()
                ax2.plot(epochs, hist["lr"], color="gray", alpha=0.3,
                         linestyle=":", linewidth=1)
                ax2.set_ylabel("LR", color="gray", fontsize=8)
                ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
                break

        ax.set_title(f"{name}\n(best val={best_val:.6f} @ ep {best_epoch})",
                     fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Curves — All Models", fontsize=14)
    fig.tight_layout()

    if save:
        path = f"{FIGURES_DIR}/training_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ── 7.2 Reconstruction Comparison Grid ────────────────────────────────────

def plot_reconstruction_grid(
    data_dir: str = "data",
    ckpt_dir: str = "checkpoints",
    n_samples: int = 4,
    save: bool = True,
) -> None:
    """Side-by-side comparison: GT, Tikhonov-Only, FC-Only, Proposed."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data with z-score normalization
    data = np.load(f"{data_dir}/test.npz")
    norm = np.load(f"{data_dir}/norm_stats.npz")
    dr_raw = data["delta_r"]
    dr_norm = (dr_raw - norm["delta_r_mean"]) / norm["delta_r_std"]
    delta_r = torch.from_numpy(dr_norm.astype(np.float32))
    delta_f = torch.from_numpy(data["delta_f_star"]).view(-1, 1, N_PIXELS_Y, N_PIXELS_X)

    # Load Pi
    fm = np.load(f"{data_dir}/forward_model.npz")
    Pi = torch.from_numpy(fm["Pi"]).float()

    # Pick diverse samples (by finding ones with different target types)
    types = data["target_type"]
    indices = []
    seen_types = set()
    for i in range(len(types)):
        if types[i] not in seen_types and types[i] != 7:  # skip empty
            indices.append(i)
            seen_types.add(types[i])
            if len(indices) >= n_samples:
                break

    # Load models
    models = {}
    for name, ModelClass in [
        ("Proposed", DualBranchUNet),
        ("Tikhonov-Only", TikhonovOnlyUNet),
        ("FC-Only", FCOnlyUNet),
    ]:
        ckpt_name = name.lower().replace("-", "_")
        model = ModelClass(Pi)
        model.load_state_dict(torch.load(
            f"{ckpt_dir}/{ckpt_name}_best.pt",
            map_location=device, weights_only=True))
        model.to(device).eval()
        models[name] = model

    extent = [0, AREA_WIDTH, 0, AREA_HEIGHT]
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    col_names = ["Ground Truth", "Tikhonov-Only", "FC-Only", "Proposed"]

    for row, idx in enumerate(indices):
        dr_sample = delta_r[idx:idx+1].to(device)
        gt = delta_f[idx, 0].numpy()

        preds = {}
        with torch.no_grad():
            for name, model in models.items():
                pred = model(dr_sample).cpu().numpy()[0, 0]
                preds[name] = pred

        for col, (title, img) in enumerate([
            ("Ground Truth", gt),
            ("Tikhonov-Only", preds["Tikhonov-Only"]),
            ("FC-Only", preds["FC-Only"]),
            ("Proposed", preds["Proposed"]),
        ]):
            ax = axes[row, col]
            im = ax.imshow(img, origin="lower", extent=extent,
                           cmap="viridis", aspect="equal",
                           vmin=0, vmax=max(gt.max(), 0.01))
            if row == 0:
                ax.set_title(title, fontsize=11)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Reconstruction Comparison", fontsize=14, y=1.01)
    fig.tight_layout()

    if save:
        path = f"{FIGURES_DIR}/reconstruction_grid.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ── 7.3 Error Maps + Branch Analysis ─────────────────────────────────────

def plot_error_maps_and_branches(
    data_dir: str = "data",
    ckpt_dir: str = "checkpoints",
    n_samples: int = 3,
    save: bool = True,
) -> None:
    """Error maps |pred - GT| + branch intermediate outputs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data with z-score normalization
    data = np.load(f"{data_dir}/test.npz")
    norm = np.load(f"{data_dir}/norm_stats.npz")
    dr_raw = data["delta_r"]
    dr_norm = (dr_raw - norm["delta_r_mean"]) / norm["delta_r_std"]
    delta_r = torch.from_numpy(dr_norm.astype(np.float32))
    delta_f = torch.from_numpy(data["delta_f_star"]).view(-1, 1, N_PIXELS_Y, N_PIXELS_X)
    types = data["target_type"]

    fm = np.load(f"{data_dir}/forward_model.npz")
    Pi = torch.from_numpy(fm["Pi"]).float()

    # Pick samples
    indices = []
    seen = set()
    for i in range(len(types)):
        if types[i] not in seen and types[i] != 7:
            indices.append(i)
            seen.add(types[i])
            if len(indices) >= n_samples:
                break

    # Load proposed model (for branch analysis)
    proposed = DualBranchUNet(Pi)
    proposed.load_state_dict(torch.load(
        f"{ckpt_dir}/proposed_best.pt",
        map_location=device, weights_only=True))
    proposed.to(device).eval()

    # Load ablation models
    tik_model = TikhonovOnlyUNet(Pi)
    tik_model.load_state_dict(torch.load(
        f"{ckpt_dir}/tikhonov_only_best.pt",
        map_location=device, weights_only=True))
    tik_model.to(device).eval()

    fc_model = FCOnlyUNet(Pi)
    fc_model.load_state_dict(torch.load(
        f"{ckpt_dir}/fc_only_best.pt",
        map_location=device, weights_only=True))
    fc_model.to(device).eval()

    extent = [0, AREA_WIDTH, 0, AREA_HEIGHT]

    # Error maps: GT | TikOnly error | FCOnly error | Proposed error
    fig1, axes1 = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    col_names1 = ["Ground Truth", "|Tik-Only - GT|", "|FC-Only - GT|", "|Proposed - GT|"]

    # Branch analysis: GT | Branch A (Tik) | Branch B (FC) | Proposed output
    fig2, axes2 = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    col_names2 = ["Ground Truth", "Branch A (Tikhonov)", "Branch B (FC)", "Proposed Output"]

    for row, idx in enumerate(indices):
        dr_sample = delta_r[idx:idx+1].to(device)
        gt = delta_f[idx, 0].numpy()

        with torch.no_grad():
            # Get predictions
            pred_tik = tik_model(dr_sample).cpu().numpy()[0, 0]
            pred_fc = fc_model(dr_sample).cpu().numpy()[0, 0]
            pred_prop = proposed(dr_sample).cpu().numpy()[0, 0]

            # Get branch intermediate outputs
            branch_a = proposed.tikhonov(dr_sample).cpu().numpy()[0, 0]
            branch_b = proposed.fc(dr_sample).cpu().numpy()[0, 0]

        # Error maps
        vmax_gt = max(gt.max(), 0.01)
        errors = [
            np.abs(pred_tik - gt),
            np.abs(pred_fc - gt),
            np.abs(pred_prop - gt),
        ]
        vmax_err = max(e.max() for e in errors)

        for col, (title, img, vm) in enumerate([
            ("Ground Truth", gt, vmax_gt),
            ("|Tik-Only - GT|", errors[0], vmax_err),
            ("|FC-Only - GT|", errors[1], vmax_err),
            ("|Proposed - GT|", errors[2], vmax_err),
        ]):
            ax = axes1[row, col]
            cmap = "viridis" if col == 0 else "hot"
            im = ax.imshow(img, origin="lower", extent=extent,
                           cmap=cmap, aspect="equal", vmin=0, vmax=vm)
            if row == 0:
                ax.set_title(title, fontsize=11)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Branch analysis
        for col, (title, img) in enumerate([
            ("Ground Truth", gt),
            ("Branch A (Tikhonov)", branch_a),
            ("Branch B (FC)", branch_b),
            ("Proposed Output", pred_prop),
        ]):
            ax = axes2[row, col]
            im = ax.imshow(img, origin="lower", extent=extent,
                           cmap="viridis", aspect="equal")
            if row == 0:
                ax.set_title(title, fontsize=11)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig1.suptitle("Error Maps: |Prediction - Ground Truth|", fontsize=14, y=1.01)
    fig1.tight_layout()
    fig2.suptitle("Branch Contribution Analysis", fontsize=14, y=1.01)
    fig2.tight_layout()

    if save:
        path1 = f"{FIGURES_DIR}/error_maps.png"
        fig1.savefig(path1, dpi=150, bbox_inches="tight")
        print(f"Saved: {path1}")
        path2 = f"{FIGURES_DIR}/branch_analysis.png"
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        print(f"Saved: {path2}")

    plt.close(fig1)
    plt.close(fig2)


# ── Run All ───────────────────────────────────────────────────────────────

def generate_all_figures() -> None:
    """Generate all Phase 7 visualizations."""
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    print("Generating training curves...")
    plot_training_curves()
    print("Generating reconstruction grid...")
    plot_reconstruction_grid()
    print("Generating error maps and branch analysis...")
    plot_error_maps_and_branches()
    print("Done.")


if __name__ == "__main__":
    generate_all_figures()
