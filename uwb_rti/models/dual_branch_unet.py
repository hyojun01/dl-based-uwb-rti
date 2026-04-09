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
    """Proposed: Tikhonov + FC -> concat (2ch) -> U-Net -> Δf_hat (1ch)."""

    def __init__(self, Pi: torch.Tensor) -> None:
        """
        Args:
            Pi: Tikhonov matrix of shape (900, 16), float32.
        """
        super().__init__()
        self.tikhonov = TikhonovBranch(Pi)
        self.fc = FCBranch(Pi)
        self.unet = UNet(in_channels=2)

    def forward(self, delta_r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_r: RSS difference vector of shape (batch, 16).

        Returns:
            Reconstructed SLF change of shape (batch, 1, 30, 30).
        """
        tik_out = self.tikhonov(delta_r)   # (B, 1, 30, 30)
        fc_out = self.fc(delta_r)          # (B, 1, 30, 30)
        combined = torch.cat([tik_out, fc_out], dim=1)  # (B, 2, 30, 30)
        return self.unet(combined)   # (B, 1, 30, 30)
