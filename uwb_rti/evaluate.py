"""Evaluation script: compute metrics for all models on the test set.

Metrics: MSE, PSNR, SSIM, RMSE per model.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_msssim import ssim

from uwb_rti.config import (
    N_PIXELS_X, N_PIXELS_Y, N_LINKS, K, BATCH_SIZE,
)
from uwb_rti.models.dual_branch_unet import DualBranchUNet
from uwb_rti.models.tikhonov_only import TikhonovOnlyUNet
from uwb_rti.models.fc_only import FCOnlyUNet


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute MSE, PSNR, SSIM, RMSE over a batch.

    Args:
        pred: Shape (N, 1, 30, 30).
        target: Shape (N, 1, 30, 30).

    Returns:
        Dict with mean metrics across all samples.
    """
    # Per-sample MSE
    mse_per = ((pred - target) ** 2).mean(dim=(1, 2, 3))  # (N,)
    mse = mse_per.mean().item()
    rmse = np.sqrt(mse)

    # PSNR: 10 * log10(max^2 / MSE), max of target
    data_max = target.max().item()
    psnr_per = 10.0 * torch.log10(data_max**2 / (mse_per + 1e-10))
    psnr = psnr_per.mean().item()

    # SSIM
    ssim_val = ssim(pred, target, data_range=data_max, size_average=True, win_size=7).item()

    return {"mse": mse, "psnr": psnr, "ssim": ssim_val, "rmse": rmse}


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate a model on the test set."""
    model = model.to(device)
    model.eval()

    all_pred = []
    all_target = []

    with torch.no_grad():
        for rss, theta in test_loader:
            rss = rss.to(device)
            pred = model(rss)
            all_pred.append(pred.cpu())
            all_target.append(theta)

    pred = torch.cat(all_pred, dim=0)
    target = torch.cat(all_target, dim=0)

    return compute_metrics(pred, target)


def evaluate_all(data_dir: str = "data", ckpt_dir: str = "checkpoints") -> dict:
    """Evaluate all 3 models and produce comparison table."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data with normalization
    data = np.load(f"{data_dir}/test.npz")
    norm = np.load(f"{data_dir}/norm_stats.npz")
    rss_raw = data["rss"]
    rss_norm = (rss_raw - norm["rss_mean"]) / norm["rss_std"]
    rss = torch.from_numpy(rss_norm.astype(np.float32))
    theta = torch.from_numpy(data["theta_star"]).view(-1, 1, N_PIXELS_Y, N_PIXELS_X)
    test_loader = DataLoader(TensorDataset(rss, theta), batch_size=BATCH_SIZE,
                             num_workers=2, pin_memory=True)

    # Load Pi
    fm = np.load(f"{data_dir}/forward_model.npz")
    Pi = torch.from_numpy(fm["Pi"]).float()

    results = {}
    models = [
        ("proposed", DualBranchUNet),
        ("tikhonov_only", TikhonovOnlyUNet),
        ("fc_only", FCOnlyUNet),
    ]

    for name, ModelClass in models:
        ckpt_path = Path(ckpt_dir) / f"{name}_best.pt"
        if not ckpt_path.exists():
            print(f"  Skipping {name}: no checkpoint at {ckpt_path}")
            continue

        model = ModelClass(Pi)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        metrics = evaluate_model(model, test_loader, device)
        results[name] = metrics
        print(f"  {name}: MSE={metrics['mse']:.6f} PSNR={metrics['psnr']:.2f}dB "
              f"SSIM={metrics['ssim']:.4f} RMSE={metrics['rmse']:.6f}")

    # Save results
    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to outputs/results/evaluation_results.json")

    # Print comparison table
    print(f"\n{'Model':<20} {'MSE':>10} {'PSNR (dB)':>10} {'SSIM':>8} {'RMSE':>10}")
    print("-" * 62)
    for name, m in results.items():
        print(f"{name:<20} {m['mse']:>10.6f} {m['psnr']:>10.2f} "
              f"{m['ssim']:>8.4f} {m['rmse']:>10.6f}")

    return results


if __name__ == "__main__":
    evaluate_all()
