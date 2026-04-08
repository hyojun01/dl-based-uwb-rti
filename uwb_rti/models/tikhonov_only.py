"""Ablation model: Tikhonov-Only + U-Net.

Uses only the fixed Tikhonov branch (1-channel) as U-Net input.
Quantifies the contribution of the FC branch by its absence.
"""

import torch
import torch.nn as nn

from uwb_rti.models.tikhonov_branch import TikhonovBranch
from uwb_rti.models.unet import UNet


class TikhonovOnlyUNet(nn.Module):
    """Ablation: Tikhonov -> 1ch -> U-Net -> SLF (1ch)."""

    def __init__(self, Pi: torch.Tensor) -> None:
        super().__init__()
        self.tikhonov = TikhonovBranch(Pi)
        self.unet = UNet(in_channels=1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        tik_out = self.tikhonov(y)   # (B, 1, 30, 30)
        return self.unet(tik_out)    # (B, 1, 30, 30)
