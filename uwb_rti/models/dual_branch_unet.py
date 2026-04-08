"""Proposed model: Dual-Branch Physics-Informed U-Net.

Concatenates Tikhonov (fixed) and FC (learnable) branch outputs
as 2-channel input to U-Net refinement network.
"""

import torch
import torch.nn as nn

from uwb_rti.models.tikhonov_branch import TikhonovBranch
from uwb_rti.models.fc_branch import FCBranch
from uwb_rti.models.unet import UNet


class DualBranchUNet(nn.Module):
    """Proposed: Tikhonov + FC -> concat (2ch) -> U-Net -> SLF (1ch)."""

    def __init__(self, Pi: torch.Tensor) -> None:
        """
        Args:
            Pi: Tikhonov matrix of shape (900, 16), float32.
        """
        super().__init__()
        self.tikhonov = TikhonovBranch(Pi)
        self.fc = FCBranch(Pi)
        self.unet = UNet(in_channels=2)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: RSS vector of shape (batch, 16).

        Returns:
            Reconstructed SLF of shape (batch, 1, 30, 30).
        """
        tik_out = self.tikhonov(y)   # (B, 1, 30, 30)
        fc_out = self.fc(y)          # (B, 1, 30, 30)
        combined = torch.cat([tik_out, fc_out], dim=1)  # (B, 2, 30, 30)
        return self.unet(combined)   # (B, 1, 30, 30)
