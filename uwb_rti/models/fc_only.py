"""Ablation model: FC-Only + U-Net.

Uses only the learnable FC branch (1-channel) as U-Net input.
Quantifies the contribution of the Tikhonov branch by its absence.
Reference: Oral et al. (2023), DeepFC.
"""

import torch
import torch.nn as nn

from uwb_rti.models.fc_branch import FCBranch
from uwb_rti.models.unet import UNet


class FCOnlyUNet(nn.Module):
    """Ablation: FC -> 1ch -> U-Net -> SLF (1ch)."""

    def __init__(self, Pi: torch.Tensor) -> None:
        """
        Args:
            Pi: Tikhonov matrix for FC weight initialization.
        """
        super().__init__()
        self.fc = FCBranch(Pi)
        self.unet = UNet(in_channels=1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        fc_out = self.fc(y)          # (B, 1, 30, 30)
        return self.unet(fc_out)     # (B, 1, 30, 30)
