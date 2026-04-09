"""Branch B: Learnable fully connected layer.

FC(16 -> 900) with no activation. Weights initialized from Tikhonov
matrix Pi as transfer learning.
"""

import torch
import torch.nn as nn

from uwb_rti.config import N_PIXELS_X, N_PIXELS_Y, N_LINKS, K


class FCBranch(nn.Module):
    """Learnable linear mapping from RSS difference to SLF change image."""

    def __init__(self, Pi: torch.Tensor) -> None:
        """
        Args:
            Pi: Tikhonov matrix of shape (K, N_LINKS) = (900, 16), float32.
                Used for weight initialization (transfer learning from Tikhonov).
        """
        super().__init__()
        assert Pi.shape == (K, N_LINKS), f"Expected Pi shape ({K}, {N_LINKS}), got {Pi.shape}"
        self.fc = nn.Linear(N_LINKS, K)
        # Transfer learning: initialize W_fc = Pi
        self.fc.weight.data.copy_(Pi)

    def forward(self, delta_r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_r: RSS difference vector of shape (batch, 16).

        Returns:
            FC reconstruction of shape (batch, 1, 30, 30).
        """
        assert delta_r.shape[1] == N_LINKS, f"Expected input dim {N_LINKS}, got {delta_r.shape[1]}"
        delta_f = self.fc(delta_r)  # (batch, K)
        return delta_f.view(-1, 1, N_PIXELS_Y, N_PIXELS_X)
