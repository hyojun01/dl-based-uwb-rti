"""Training script for all UWB RTI models.

Trains proposed (DualBranch), Tikhonov-Only, and FC-Only models with
identical configuration and data splits. Logs experiments to experiments/.
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_msssim import ssim

from uwb_rti.config import (
    N_PIXELS_X, N_PIXELS_Y, N_LINKS, K,
    LEARNING_RATE, BATCH_SIZE, MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE, SCHEDULER_PATIENCE, SCHEDULER_FACTOR,
    LOSS_SSIM_LAMBDA, RANDOM_SEED,
)
from uwb_rti.models.dual_branch_unet import DualBranchUNet
from uwb_rti.models.tikhonov_only import TikhonovOnlyUNet
from uwb_rti.models.fc_only import FCOnlyUNet


# =============================================================================
# Loss function
# =============================================================================

class CombinedLoss(nn.Module):
    """Weighted MSE + lambda * (1 - SSIM) loss for SLF change reconstruction.

    Uses pixel-weighting to combat class imbalance (97.6% zero pixels).
    Nonzero target pixels are weighted higher to prevent trivial all-zeros solution.
    """

    def __init__(self, ssim_lambda: float = LOSS_SSIM_LAMBDA,
                 object_weight: float = 20.0) -> None:
        super().__init__()
        self.ssim_lambda = ssim_lambda
        self.object_weight = object_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted SLF change, shape (B, 1, 30, 30).
            target: Ground truth SLF change, shape (B, 1, 30, 30).
        """
        # Pixel-weighted MSE: upweight nonzero (object) pixels
        weight = torch.where(target > 0, self.object_weight, 1.0)
        weighted_mse = (weight * (pred - target) ** 2).mean()
        # SSIM: window_size=7 for 30x30 images (must be odd, <= image size)
        ssim_val = ssim(pred, target, data_range=1.0, size_average=True, win_size=7)
        return weighted_mse + self.ssim_lambda * (1.0 - ssim_val)


# =============================================================================
# Data loading
# =============================================================================

def load_data(data_dir: str = "data") -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Load train/val/test datasets with z-score normalized RSS difference.

    Returns:
        train_loader, val_loader, test_loader, norm_stats.
    """
    # Compute normalization stats from training set
    train_data = np.load(f"{data_dir}/train.npz")
    dr_mean = train_data["delta_r"].mean(axis=0)  # (16,) per-link
    dr_std = train_data["delta_r"].std(axis=0)     # (16,) per-link
    norm_stats = {"delta_r_mean": dr_mean, "delta_r_std": dr_std}

    loaders = []
    for split in ["train", "val", "test"]:
        data = np.load(f"{data_dir}/{split}.npz")
        delta_r = (data["delta_r"] - dr_mean) / dr_std  # z-score normalize
        delta_r = torch.from_numpy(delta_r.astype(np.float32))
        delta_f = torch.from_numpy(data["delta_f_star"])  # (N, 900)
        delta_f = delta_f.view(-1, 1, N_PIXELS_Y, N_PIXELS_X)
        dataset = TensorDataset(delta_r, delta_f)
        shuffle = (split == "train")
        loaders.append(DataLoader(dataset, batch_size=BATCH_SIZE,
                                  shuffle=shuffle, num_workers=2,
                                  pin_memory=True))
    return loaders[0], loaders[1], loaders[2], norm_stats


# =============================================================================
# Training loop
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    device: torch.device,
    exp_id: int,
) -> dict:
    """Train a single model with early stopping and LR scheduling.

    Returns:
        Dict with training history and best metrics.
    """
    model = model.to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR)

    # Checkpointing
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "lr": []}

    # Log experiment config before training
    exp_config = {
        "exp_id": exp_id,
        "model_name": model_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "scheduler_patience": SCHEDULER_PATIENCE,
            "scheduler_factor": SCHEDULER_FACTOR,
            "loss_ssim_lambda": LOSS_SSIM_LAMBDA,
            "random_seed": RANDOM_SEED,
        },
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    exp_dir = Path("experiments")
    exp_dir.mkdir(exist_ok=True)
    exp_path = exp_dir / f"exp_{exp_id:03d}.json"
    with open(exp_path, "w") as f:
        json.dump(exp_config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training: {model_name} (exp_{exp_id:03d})")
    print(f"Params: {exp_config['trainable_params']:,}")
    print(f"{'='*60}")

    t0 = time.time()

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        train_loss_sum = 0.0
        for delta_r, delta_f in train_loader:
            delta_r, delta_f = delta_r.to(device), delta_f.to(device)
            pred = model(delta_r)
            loss = criterion(pred, delta_f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * delta_r.size(0)
        train_loss = train_loss_sum / len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for delta_r, delta_f in val_loader:
                delta_r, delta_f = delta_r.to(device), delta_f.to(device)
                pred = model(delta_r)
                loss = criterion(pred, delta_f)
                val_loss_sum += loss.item() * delta_r.size(0)
        val_loss = val_loss_sum / len(val_loader.dataset)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        scheduler.step(val_loss)

        # Early stopping (with min_delta to avoid float-tie stalling)
        min_delta = 1e-6
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / f"{model_name}_best.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
                  f"train={train_loss:.6f} val={val_loss:.6f} | "
                  f"lr={current_lr:.1e} | best={best_val_loss:.6f} | "
                  f"{elapsed:.0f}s")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    total_time = time.time() - t0
    best_epoch = int(np.argmin(history["val_loss"])) + 1

    # Update experiment log with results
    exp_config["completed_at"] = datetime.now(timezone.utc).isoformat()
    exp_config["results"] = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "training_time_sec": round(total_time, 1),
        "final_lr": history["lr"][-1],
    }
    exp_config["history"] = history
    with open(exp_path, "w") as f:
        json.dump(exp_config, f, indent=2)

    print(f"  Done: best_val={best_val_loss:.6f} @ epoch {best_epoch}, "
          f"time={total_time:.0f}s")

    return exp_config


# =============================================================================
# Main: train all 3 models
# =============================================================================

def train_all(data_dir: str = "data") -> list[dict]:
    """Train proposed + both ablation models with identical config."""
    # Seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data (with z-score normalization)
    train_loader, val_loader, test_loader, norm_stats = load_data(data_dir)
    print(f"Data: {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test")
    print(f"RSS difference normalized: mean subtracted, std divided (per-link)")

    # Save normalization stats for inference
    np.savez(f"{data_dir}/norm_stats.npz", **norm_stats)

    # Load Pi
    fm = np.load(f"{data_dir}/forward_model.npz")
    Pi_tensor = torch.from_numpy(fm["Pi"]).float()

    # Find next experiment ID
    exp_dir = Path("experiments")
    exp_dir.mkdir(exist_ok=True)
    existing = list(exp_dir.glob("exp_*.json"))
    next_id = max((int(p.stem.split("_")[1]) for p in existing), default=0) + 1

    results = []
    models_to_train = [
        ("proposed", DualBranchUNet),
        ("tikhonov_only", TikhonovOnlyUNet),
        ("fc_only", FCOnlyUNet),
    ]

    for model_name, ModelClass in models_to_train:
        # Reset seed before each model for fair comparison
        torch.manual_seed(RANDOM_SEED)
        model = ModelClass(Pi_tensor)
        result = train_model(model, train_loader, val_loader,
                             model_name, device, next_id)
        results.append(result)
        next_id += 1

    return results


if __name__ == "__main__":
    results = train_all()
    print("\n=== Training Summary ===")
    for r in results:
        name = r["model_name"]
        res = r["results"]
        print(f"  {name}: val_loss={res['best_val_loss']:.6f} "
              f"@ epoch {res['best_epoch']}, time={res['training_time_sec']}s")
