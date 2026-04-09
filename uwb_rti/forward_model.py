"""UWB RTI forward model: weight matrix, Tikhonov matrix, and RSS difference generation.

References:
    - Hamilton et al. (2014): Inverse Area Elliptical Model
    - Wu et al. (2024): RSS difference model, Section 5.1
"""

import numpy as np

from uwb_rti.config import (
    TX_POSITIONS, RX_POSITIONS, PIXEL_CENTERS,
    N_TX, N_RX, N_LINKS, K,
    WAVELENGTH, BETA_MIN, SCALING_CONSTANT_C, PIXEL_SIZE,
    TIKHONOV_ALPHA_REG,
    NOISE_STD_RANGE,
)


def compute_weight_matrix() -> np.ndarray:
    """Compute the weight matrix W using the Inverse Area Elliptical Model.

    For each TX-RX link and each pixel, computes the weight based on the
    elliptical geometry with the TX and RX as foci.

    Returns:
        W: Weight matrix of shape (N_LINKS, K) = (16, 900), dtype float64.
    """
    W = np.zeros((N_LINKS, K), dtype=np.float64)

    for i in range(N_TX):
        for j in range(N_RX):
            link_idx = i * N_RX + j
            tx = TX_POSITIONS[i]   # (2,)
            rx = RX_POSITIONS[j]   # (2,)

            d_nm = np.linalg.norm(tx - rx)
            c_half = d_nm / 2.0

            # Fresnel zone cutoff
            beta_max = np.sqrt(WAVELENGTH * d_nm / 4.0)

            # Distances from each pixel to TX and RX
            d1 = np.linalg.norm(PIXEL_CENTERS - tx, axis=1)  # (K,)
            d2 = np.linalg.norm(PIXEL_CENTERS - rx, axis=1)  # (K,)

            # Semi-major axis per pixel
            a = (d1 + d2) / 2.0  # (K,)

            # Semi-minor axis: β = sqrt(a² - c_half²)
            # Clamp argument to avoid numerical issues for points near the foci
            beta_sq = a**2 - c_half**2
            beta_sq = np.maximum(beta_sq, 0.0)
            beta = np.sqrt(beta_sq)  # (K,)

            # Compute weights with clamping and cutoff
            weights = np.zeros(K, dtype=np.float64)

            # Pixels inside the Fresnel zone (β < β_max)
            inside = beta < beta_max

            # Clamp β at β_min for pixels very close to the LOS
            beta_clamped = np.maximum(beta, BETA_MIN)

            # Weight = [1 / (π * a * β_clamped)] * pixel_area for discrete summation
            # The continuous weight density (1/m²) is integrated over each pixel
            pixel_area = PIXEL_SIZE ** 2
            weights[inside] = pixel_area / (np.pi * a[inside] * beta_clamped[inside])

            W[link_idx] = weights

    return W


def compute_tikhonov_matrix(W: np.ndarray, alpha: float = TIKHONOV_ALPHA_REG) -> np.ndarray:
    """Compute the Tikhonov regularized inverse matrix Pi.

    Pi = (W^T W + alpha * I)^{-1} * W^T

    Args:
        W: Weight matrix of shape (N_LINKS, K).
        alpha: Tikhonov regularization parameter.

    Returns:
        Pi: Tikhonov matrix of shape (K, N_LINKS) = (900, 16), dtype float64.
    """
    WtW = W.T @ W                          # (K, K)
    reg = alpha * np.eye(K, dtype=np.float64)
    Pi = np.linalg.solve(WtW + reg, W.T)   # (K, N_LINKS)
    return Pi


def generate_rss_difference(W: np.ndarray, delta_f: np.ndarray,
                            rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
    """Generate RSS difference vector from the forward model.

    ΔR = c · W · Δf_A + ε

    where Δf_A = Δf*_A + f̃_A (ideal SLF change + spatially correlated noise).

    Args:
        W: Weight matrix of shape (N_LINKS, K).
        delta_f: SLF change vector of shape (K,) — includes noise (Δf*_A + f̃_A).
        rng: NumPy random generator for reproducibility.

    Returns:
        delta_r: RSS difference vector of shape (N_LINKS,).
        params: Dict of sampled parameters (noise_std).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample measurement noise
    sigma_eps = rng.uniform(*NOISE_STD_RANGE)             # scalar
    epsilon = rng.normal(0.0, sigma_eps, size=N_LINKS)    # (16,)

    # Forward model: ΔR = c · W · Δf_A + ε
    delta_r = SCALING_CONSTANT_C * (W @ delta_f) + epsilon  # (16,)

    params = {
        "noise_std": sigma_eps,
    }
    return delta_r, params
