"""Branch A: Fixed Tikhonov reconstruction (no learnable parameters).

Computes theta_tik = Pi @ y and reshapes to (1, 30, 30).
Pi is registered as a buffer — frozen during training.
"""

import torch
import torch.nn as nn

from uwb_rti.config import N_PIXELS_X, N_PIXELS_Y, N_LINKS, K


class TikhonovBranch(nn.Module):
    """Fixed linear reconstruction using precomputed Tikhonov matrix Pi."""

    def __init__(self, Pi: torch.Tensor) -> None:
        """
        Args:
            Pi: Tikhonov matrix of shape (K, N_LINKS) = (900, 16), float32.
        """
        super().__init__()
        assert Pi.shape == (K, N_LINKS), f"Expected Pi shape ({K}, {N_LINKS}), got {Pi.shape}"
        self.register_buffer("Pi", Pi)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: RSS vector of shape (batch, 16).

        Returns:
            Tikhonov reconstruction of shape (batch, 1, 30, 30).
        """
        assert y.shape[1] == N_LINKS, f"Expected input dim {N_LINKS}, got {y.shape[1]}"
        theta = y @ self.Pi.T  # (batch, K)
        return theta.view(-1, 1, N_PIXELS_Y, N_PIXELS_X)
