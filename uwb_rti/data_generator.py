"""Training data generation for UWB RTI.

Generates 60,000 samples: SLF targets with spatially correlated noise,
RSS measurements via forward model, and Tikhonov initial estimates.
"""

import numpy as np
from pathlib import Path
from typing import Literal

from uwb_rti.config import (
    K, N_PIXELS_X, N_PIXELS_Y, PIXEL_CENTERS, PIXEL_SIZE,
    AREA_WIDTH, AREA_HEIGHT, N_LINKS,
    SLF_NOISE_STD_RANGE, SLF_SPATIAL_CORR_LENGTH, SLF_EDGE_MARGIN,
    SCALING_CONSTANT_C,
    BIAS_RANGE, PATH_LOSS_EXPONENT_RANGE, NOISE_STD_RANGE,
    DATASET_TOTAL, DATASET_TRAIN, DATASET_VAL, DATASET_TEST,
    DATA_SPLIT_SEED, RANDOM_SEED,
)
from uwb_rti.forward_model import (
    compute_weight_matrix,
    compute_tikhonov_matrix,
    compute_log_distances,
)


# =============================================================================
# Noise covariance (precomputed once)
# =============================================================================

def compute_noise_covariance() -> np.ndarray:
    """Precompute spatial noise covariance skeleton C(k,l) = exp(-D_kl / kappa).

    The actual covariance is sigma_theta^2 * C. We factor out sigma_theta
    so we can reuse C across samples with different sigma_theta values.

    Returns:
        C_skeleton: Shape (K, K), dtype float64.
    """
    # Pairwise distances between all pixel centers
    diff = PIXEL_CENTERS[:, None, :] - PIXEL_CENTERS[None, :, :]  # (K, K, 2)
    D = np.linalg.norm(diff, axis=2)  # (K, K)
    C_skeleton = np.exp(-D / SLF_SPATIAL_CORR_LENGTH)
    return C_skeleton


def compute_noise_cholesky(C_skeleton: np.ndarray) -> np.ndarray:
    """Cholesky decomposition of covariance skeleton for efficient sampling.

    Returns:
        L: Lower triangular Cholesky factor, shape (K, K).
    """
    # Add small diagonal for numerical stability
    C = C_skeleton + 1e-10 * np.eye(K, dtype=np.float64)
    return np.linalg.cholesky(C)


# =============================================================================
# SLF Target Generators
# =============================================================================

def _place_rect(theta: np.ndarray, cx: float, cy: float,
                w: float, h: float, value: float) -> None:
    """Place a rectangular object on the SLF grid (in-place)."""
    x_min, x_max = cx - w / 2, cx + w / 2
    y_min, y_max = cy - h / 2, cy + h / 2
    for k in range(K):
        px, py = PIXEL_CENTERS[k]
        if x_min <= px <= x_max and y_min <= py <= y_max:
            theta[k] = max(theta[k], value)


def _place_circle(theta: np.ndarray, cx: float, cy: float,
                  radius: float, value: float) -> None:
    """Place a circular object on the SLF grid (in-place)."""
    for k in range(K):
        px, py = PIXEL_CENTERS[k]
        if (px - cx)**2 + (py - cy)**2 <= radius**2:
            theta[k] = max(theta[k], value)


def _random_center(rng: np.random.Generator,
                   obj_w: float, obj_h: float) -> tuple[float, float]:
    """Random center ensuring object stays within margins."""
    margin = SLF_EDGE_MARGIN
    cx = rng.uniform(margin + obj_w / 2, AREA_WIDTH - margin - obj_w / 2)
    cy = rng.uniform(margin + obj_h / 2, AREA_HEIGHT - margin - obj_h / 2)
    return cx, cy


def generate_person_standing(rng: np.random.Generator) -> np.ndarray:
    """Person standing: ~0.4m x 0.4m, attenuation U(0.5, 1.0)."""
    theta = np.zeros(K, dtype=np.float64)
    w = rng.uniform(0.35, 0.45)
    h = rng.uniform(0.35, 0.45)
    value = rng.uniform(0.5, 1.0)
    cx, cy = _random_center(rng, w, h)
    _place_rect(theta, cx, cy, w, h, value)
    return theta


def generate_person_walking(rng: np.random.Generator) -> np.ndarray:
    """Person walking: ~0.3m x 0.5m, attenuation U(0.5, 1.0)."""
    theta = np.zeros(K, dtype=np.float64)
    w = rng.uniform(0.25, 0.35)
    h = rng.uniform(0.45, 0.55)
    value = rng.uniform(0.5, 1.0)
    cx, cy = _random_center(rng, w, h)
    _place_rect(theta, cx, cy, w, h, value)
    return theta


def generate_table(rng: np.random.Generator) -> np.ndarray:
    """Table/desk: ~0.8m x 0.6m, attenuation U(0.3, 0.6)."""
    theta = np.zeros(K, dtype=np.float64)
    w = rng.uniform(0.7, 0.9)
    h = rng.uniform(0.5, 0.7)
    value = rng.uniform(0.3, 0.6)
    cx, cy = _random_center(rng, w, h)
    _place_rect(theta, cx, cy, w, h, value)
    return theta


def generate_chair(rng: np.random.Generator) -> np.ndarray:
    """Chair: ~0.4m x 0.4m, attenuation U(0.3, 0.5)."""
    theta = np.zeros(K, dtype=np.float64)
    w = rng.uniform(0.35, 0.45)
    h = rng.uniform(0.35, 0.45)
    value = rng.uniform(0.3, 0.5)
    cx, cy = _random_center(rng, w, h)
    _place_rect(theta, cx, cy, w, h, value)
    return theta


def generate_cabinet(rng: np.random.Generator) -> np.ndarray:
    """Cabinet/shelf: ~0.5m x 0.3m, attenuation U(0.5, 0.8)."""
    theta = np.zeros(K, dtype=np.float64)
    w = rng.uniform(0.45, 0.55)
    h = rng.uniform(0.25, 0.35)
    value = rng.uniform(0.5, 0.8)
    cx, cy = _random_center(rng, w, h)
    _place_rect(theta, cx, cy, w, h, value)
    return theta


def generate_wall(rng: np.random.Generator) -> np.ndarray:
    """Wall segment: ~0.1m x 1.0m, attenuation U(0.6, 1.0)."""
    theta = np.zeros(K, dtype=np.float64)
    # Random orientation: horizontal or vertical
    if rng.random() < 0.5:
        w, h = rng.uniform(0.08, 0.12), rng.uniform(0.8, 1.2)
    else:
        w, h = rng.uniform(0.8, 1.2), rng.uniform(0.08, 0.12)
    value = rng.uniform(0.6, 1.0)
    cx, cy = _random_center(rng, w, h)
    _place_rect(theta, cx, cy, w, h, value)
    return theta


def generate_empty(rng: np.random.Generator) -> np.ndarray:
    """Empty room: all zeros."""
    return np.zeros(K, dtype=np.float64)


def generate_l_shaped(rng: np.random.Generator) -> np.ndarray:
    """L-shaped or T-shaped composite object."""
    theta = np.zeros(K, dtype=np.float64)
    value = rng.uniform(0.4, 0.8)
    base_w = rng.uniform(0.6, 0.9)
    base_h = rng.uniform(0.15, 0.25)
    arm_w = rng.uniform(0.15, 0.25)
    arm_h = rng.uniform(0.4, 0.7)

    cx, cy = _random_center(rng, base_w, base_h + arm_h)
    # Horizontal base
    _place_rect(theta, cx, cy - arm_h / 2, base_w, base_h, value)
    # Vertical arm (left side for L, center for T)
    if rng.random() < 0.5:
        arm_cx = cx - base_w / 2 + arm_w / 2
    else:
        arm_cx = cx
    _place_rect(theta, arm_cx, cy + base_h / 2, arm_w, arm_h, value)
    return theta


def generate_circular(rng: np.random.Generator) -> np.ndarray:
    """Circular object (pillar): radius ~0.2-0.3m."""
    theta = np.zeros(K, dtype=np.float64)
    radius = rng.uniform(0.2, 0.3)
    value = rng.uniform(0.4, 0.8)
    margin = SLF_EDGE_MARGIN
    cx = rng.uniform(margin + radius, AREA_WIDTH - margin - radius)
    cy = rng.uniform(margin + radius, AREA_HEIGHT - margin - radius)
    _place_circle(theta, cx, cy, radius, value)
    return theta


def generate_multiple(rng: np.random.Generator) -> np.ndarray:
    """Combination of 2-3 random objects."""
    theta = np.zeros(K, dtype=np.float64)
    single_generators = [
        generate_person_standing, generate_person_walking,
        generate_table, generate_chair, generate_cabinet,
        generate_wall, generate_circular,
    ]
    n_objects = rng.integers(2, 4)  # 2 or 3
    for _ in range(n_objects):
        gen = rng.choice(single_generators)
        obj = gen(rng)
        theta = np.maximum(theta, obj)
    return theta


# All target generators indexed by type
TARGET_GENERATORS = [
    generate_person_standing,
    generate_person_walking,
    generate_table,
    generate_chair,
    generate_cabinet,
    generate_wall,
    generate_multiple,
    generate_empty,
    generate_l_shaped,
    generate_circular,
]


# =============================================================================
# Full sample generation
# =============================================================================

def generate_single_sample(
    W: np.ndarray,
    Pi: np.ndarray,
    d_log: np.ndarray,
    L_noise: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Generate one complete training sample.

    Returns dict with keys: theta_star, theta, rss, tikhonov_recon, target_type.
    """
    # Pick target type
    type_idx = rng.integers(0, len(TARGET_GENERATORS))
    theta_star = TARGET_GENERATORS[type_idx](rng)

    # Add spatially correlated noise
    sigma_theta = rng.uniform(*SLF_NOISE_STD_RANGE)
    z = rng.standard_normal(K)
    theta_tilde = sigma_theta * (L_noise @ z)
    theta = theta_star + theta_tilde

    # Generate RSS: y = b - c*W*theta - alpha*d + epsilon
    b = rng.uniform(*BIAS_RANGE, size=N_LINKS)
    alpha_pl = rng.uniform(*PATH_LOSS_EXPONENT_RANGE)
    sigma_eps = rng.uniform(*NOISE_STD_RANGE)
    epsilon = rng.normal(0.0, sigma_eps, size=N_LINKS)

    shadowing = SCALING_CONSTANT_C * (W @ theta)
    rss = b - shadowing - alpha_pl * d_log + epsilon

    # Tikhonov initial estimate
    tikhonov_recon = Pi @ rss

    return {
        "theta_star": theta_star.astype(np.float32),
        "rss": rss.astype(np.float32),
        "tikhonov_recon": tikhonov_recon.astype(np.float32),
        "target_type": type_idx,
    }


def generate_dataset(
    n_samples: int = DATASET_TOTAL,
    seed: int = RANDOM_SEED,
    save_dir: str = "data",
) -> dict:
    """Generate the full dataset and save to disk.

    Args:
        n_samples: Total number of samples.
        seed: Random seed for reproducibility.
        save_dir: Directory to save dataset files.

    Returns:
        Dict with dataset statistics.
    """
    import time
    rng = np.random.default_rng(seed)

    print("Precomputing forward model components...")
    W = compute_weight_matrix()
    Pi = compute_tikhonov_matrix(W)
    d_log = compute_log_distances()

    print("Precomputing noise covariance Cholesky...")
    C_skeleton = compute_noise_covariance()
    L_noise = compute_noise_cholesky(C_skeleton)

    # Allocate arrays
    all_theta_star = np.zeros((n_samples, K), dtype=np.float32)
    all_rss = np.zeros((n_samples, N_LINKS), dtype=np.float32)
    all_tikhonov = np.zeros((n_samples, K), dtype=np.float32)
    all_types = np.zeros(n_samples, dtype=np.int32)

    print(f"Generating {n_samples} samples...")
    t0 = time.time()
    for i in range(n_samples):
        sample = generate_single_sample(W, Pi, d_log, L_noise, rng)
        all_theta_star[i] = sample["theta_star"]
        all_rss[i] = sample["rss"]
        all_tikhonov[i] = sample["tikhonov_recon"]
        all_types[i] = sample["target_type"]

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_samples} ({elapsed:.1f}s)")

    gen_time = time.time() - t0
    print(f"Generation complete: {gen_time:.1f}s")

    # Fixed split using separate seed
    split_rng = np.random.default_rng(DATA_SPLIT_SEED)
    indices = split_rng.permutation(n_samples)
    train_idx = indices[:DATASET_TRAIN]
    val_idx = indices[DATASET_TRAIN:DATASET_TRAIN + DATASET_VAL]
    test_idx = indices[DATASET_TRAIN + DATASET_VAL:]

    # Save
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        np.savez_compressed(
            save_path / f"{name}.npz",
            theta_star=all_theta_star[idx],
            rss=all_rss[idx],
            tikhonov_recon=all_tikhonov[idx],
            target_type=all_types[idx],
        )
        print(f"Saved {name}.npz: {len(idx)} samples")

    # Save forward model matrices
    np.savez(save_path / "forward_model.npz", W=W, Pi=Pi, d_log=d_log)
    print("Saved forward_model.npz")

    # Type distribution
    type_names = [
        "person_standing", "person_walking", "table", "chair", "cabinet",
        "wall", "multiple", "empty", "l_shaped", "circular",
    ]
    type_counts = {type_names[i]: int((all_types == i).sum())
                   for i in range(len(TARGET_GENERATORS))}

    stats = {
        "n_samples": n_samples,
        "generation_time_sec": round(gen_time, 1),
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "type_distribution": type_counts,
        "rss_mean": float(all_rss.mean()),
        "rss_std": float(all_rss.std()),
        "theta_star_nonzero_frac": float((all_theta_star > 0).mean()),
    }
    return stats


if __name__ == "__main__":
    stats = generate_dataset()
    print("\n=== Dataset Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
