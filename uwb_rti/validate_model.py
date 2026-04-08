"""Model validation: verify forward model correctness before data generation.

Validations:
    1. RSS vs Distance (no shadowing) — log-distance decay
    2. RSS during human crossing — attenuation dip at LOS
    3. Weight matrix W visualization — elliptical patterns
    4. Tikhonov reconstruction quality — spatial prior check
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uwb_rti.config import (
    TX_POSITIONS, RX_POSITIONS, PIXEL_CENTERS,
    N_TX, N_RX, N_LINKS, K, N_PIXELS_X, N_PIXELS_Y,
    AREA_WIDTH, AREA_HEIGHT, PIXEL_SIZE,
    SCALING_CONSTANT_C, WAVELENGTH,
)
from uwb_rti.forward_model import (
    compute_weight_matrix,
    compute_tikhonov_matrix,
)

FIGURES_DIR = "outputs/figures"


# ── Validation 1: RSS vs Distance ──────────────────────────────────────────

def validate_rss_vs_distance(save: bool = True) -> np.ndarray:
    """Theoretical RSS vs distance with no shadowing.

    y = b - alpha * 20 * log10(D), using b=95, alpha=0.95, D in [0.5, 5].

    Returns:
        Array of (distance, rss) pairs.
    """
    b = 95.0
    alpha = 0.95
    D = np.linspace(0.5, 5.0, 200)
    y = b - alpha * 20.0 * np.log10(D)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(D, y, "b-", linewidth=2)
    ax.set_xlabel("Distance D [m]")
    ax.set_ylabel("RSS [dB]")
    ax.set_title("Validation 1: RSS vs Distance (No Shadowing)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 5.0)
    fig.tight_layout()

    if save:
        path = f"{FIGURES_DIR}/val1_rss_vs_distance.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close(fig)

    return np.column_stack([D, y])


# ── Validation 2: RSS During Human Crossing ────────────────────────────────

def validate_human_crossing(save: bool = True) -> np.ndarray:
    """RSS change as a person crosses the TX0-RX0 link at the midpoint.

    TX at (0,0), RX at (0,3). Person (0.4m x 0.4m, theta*=0.7) moves
    from x=-1 to x=4 at y=1.5.

    Returns:
        Array of (x_position, rss) pairs.
    """
    # Use a single link: TX0 at (0,0), RX0 at (0,3)
    tx = TX_POSITIONS[0]
    rx = RX_POSITIONS[0]
    d_nm = np.linalg.norm(tx - rx)

    b = 95.0
    alpha = 0.95
    d_log = 20.0 * np.log10(d_nm)

    # Compute weight vector for this single link
    c_half = d_nm / 2.0
    beta_max = np.sqrt(WAVELENGTH * d_nm / 4.0)

    d1 = np.linalg.norm(PIXEL_CENTERS - tx, axis=1)
    d2 = np.linalg.norm(PIXEL_CENTERS - rx, axis=1)
    a = (d1 + d2) / 2.0
    beta_sq = np.maximum(a**2 - c_half**2, 0.0)
    beta = np.sqrt(beta_sq)
    from uwb_rti.config import BETA_MIN
    beta_clamped = np.maximum(beta, BETA_MIN)
    inside = beta < beta_max
    w = np.zeros(K, dtype=np.float64)
    pixel_area = PIXEL_SIZE ** 2
    w[inside] = pixel_area / (np.pi * a[inside] * beta_clamped[inside])

    # Sweep person position
    person_w, person_h = 0.4, 0.4
    x_positions = np.linspace(-1.0, 4.0, 300)
    y_center = 1.5
    rss_values = np.zeros(len(x_positions))

    for idx, x_center in enumerate(x_positions):
        theta = np.zeros(K, dtype=np.float64)

        # Fill pixels covered by the person
        x_min = x_center - person_w / 2
        x_max = x_center + person_w / 2
        y_min = y_center - person_h / 2
        y_max = y_center + person_h / 2

        for k in range(K):
            px, py = PIXEL_CENTERS[k]
            if x_min <= px <= x_max and y_min <= py <= y_max:
                theta[k] = 0.7

        shadowing = SCALING_CONSTANT_C * np.dot(w, theta)
        rss_values[idx] = b - shadowing - alpha * d_log

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_positions, rss_values, "r-", linewidth=2)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="LOS (x=0)")
    ax.set_xlabel("Person x-position [m]")
    ax.set_ylabel("RSS [dB]")
    ax.set_title("Validation 2: RSS During Human Crossing (TX0-RX0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        path = f"{FIGURES_DIR}/val2_human_crossing.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close(fig)

    return np.column_stack([x_positions, rss_values])


# ── Validation 3: Weight Matrix Visualization ──────────────────────────────

def validate_weight_matrix(W: np.ndarray, save: bool = True) -> None:
    """Visualize weight vectors reshaped to 30x30 for representative links."""
    links = {
        "TX0-RX0 (edge, d=3.0m)": 0,
        "TX1-RX1 (center, d=3.0m)": 5,
        "TX0-RX3 (diagonal, d=4.2m)": 3,
        "TX1-RX2 (off-center, d=3.2m)": 6,
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (label, link_idx) in zip(axes, links.items()):
        w_img = W[link_idx].reshape(N_PIXELS_Y, N_PIXELS_X)
        im = ax.imshow(w_img, origin="lower", extent=[0, AREA_WIDTH, 0, AREA_HEIGHT],
                        cmap="hot", aspect="equal")
        # Mark TX and RX
        i, j = link_idx // N_RX, link_idx % N_RX
        ax.plot(*TX_POSITIONS[i], "bv", markersize=10, label="TX")
        ax.plot(*RX_POSITIONS[j], "g^", markersize=10, label="RX")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.colorbar(im, ax=ax, fraction=0.046)

    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Validation 3: Weight Matrix W — Elliptical Patterns", y=1.02)
    fig.tight_layout()

    if save:
        path = f"{FIGURES_DIR}/val3_weight_matrix.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ── Validation 4: Tikhonov Reconstruction ──────────────────────────────────

def validate_tikhonov_reconstruction(W: np.ndarray, Pi: np.ndarray,
                                     save: bool = True) -> None:
    """Apply Pi to synthetic RSS for sample targets and visualize."""
    extent = [0, AREA_WIDTH, 0, AREA_HEIGHT]

    # Create 3 test targets
    targets = {}

    # Target 1: single person at center
    t1 = np.zeros(K, dtype=np.float64)
    for r in range(13, 17):
        for c in range(13, 17):
            t1[r * N_PIXELS_X + c] = 0.7
    targets["Person (center)"] = t1

    # Target 2: two objects
    t2 = np.zeros(K, dtype=np.float64)
    for r in range(5, 9):
        for c in range(5, 9):
            t2[r * N_PIXELS_X + c] = 0.6
    for r in range(20, 25):
        for c in range(20, 25):
            t2[r * N_PIXELS_X + c] = 0.8
    targets["Two objects"] = t2

    # Target 3: wall segment
    t3 = np.zeros(K, dtype=np.float64)
    for r in range(10, 20):
        for c in range(14, 16):
            t3[r * N_PIXELS_X + c] = 0.9
    targets["Wall segment"] = t3

    fig, axes = plt.subplots(len(targets), 3, figsize=(12, 4 * len(targets)))

    for row, (name, theta_true) in enumerate(targets.items()):
        # Generate shadowing and reconstruct
        s = SCALING_CONSTANT_C * W @ theta_true
        theta_hat = Pi @ s

        gt_img = theta_true.reshape(N_PIXELS_Y, N_PIXELS_X)
        recon_img = theta_hat.reshape(N_PIXELS_Y, N_PIXELS_X)
        err_img = np.abs(gt_img - recon_img)

        for col, (img, title) in enumerate([
            (gt_img, f"{name} — Ground Truth"),
            (recon_img, f"{name} — Tikhonov Recon"),
            (err_img, f"{name} — |Error|"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(img, origin="lower", extent=extent,
                           cmap="viridis", aspect="equal")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Validation 4: Tikhonov Reconstruction Quality", y=1.01)
    fig.tight_layout()

    if save:
        path = f"{FIGURES_DIR}/val4_tikhonov_reconstruction.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ── Run All Validations ───────────────────────────────────────────────────

def run_all_validations() -> dict:
    """Run all validations and return summary."""
    results = {}

    # Validation 1
    data1 = validate_rss_vs_distance()
    rss_decreasing = np.all(np.diff(data1[:, 1]) < 0)
    results["val1_rss_vs_distance"] = {
        "rss_at_0.5m": float(data1[0, 1]),
        "rss_at_5.0m": float(data1[-1, 1]),
        "monotonically_decreasing": bool(rss_decreasing),
    }
    print(f"Val 1: RSS {data1[0,1]:.1f} dB @ 0.5m → {data1[-1,1]:.1f} dB @ 5m, "
          f"decreasing={rss_decreasing}")

    # Validation 2
    data2 = validate_human_crossing()
    min_idx = np.argmin(data2[:, 1])
    min_x = data2[min_idx, 0]
    baseline = data2[0, 1]
    min_rss = data2[min_idx, 1]
    results["val2_human_crossing"] = {
        "min_rss_x_position": float(min_x),
        "rss_dip_dB": float(baseline - min_rss),
        "min_near_los": bool(abs(min_x) < 0.3),
    }
    print(f"Val 2: Max dip {baseline - min_rss:.2f} dB at x={min_x:.2f}m, "
          f"near LOS={abs(min_x) < 0.3}")

    # Validation 3 & 4
    W = compute_weight_matrix()
    Pi = compute_tikhonov_matrix(W)
    validate_weight_matrix(W)
    print("Val 3: Weight matrix plots saved")
    validate_tikhonov_reconstruction(W, Pi)
    print("Val 4: Tikhonov reconstruction plots saved")

    results["val3_weight_matrix"] = "elliptical_patterns_visualized"
    results["val4_tikhonov"] = "reconstruction_visualized"

    return results


if __name__ == "__main__":
    results = run_all_validations()
    print("\n=== Validation Summary ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
