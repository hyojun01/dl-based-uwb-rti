"""U-Net refinement network for SLF image reconstruction.

2-level encoder + bottleneck + decoder with skip connections.
Handles 30x30 input (non-power-of-2) by padding to 32x32 internally.

Shared across proposed model and both ablation variants — only the
input channel count changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two consecutive Conv2d-BN-ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """2-level U-Net for 30x30 SLF refinement.

    Pads input from 30x30 to 32x32 so all pool/upsample operations
    produce integer sizes (32->16->8->16->32), then crops back to 30x30.

    Args:
        in_channels: Number of input channels (2 for proposed, 1 for ablations).
    """

    def __init__(self, in_channels: int = 2) -> None:
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)  # 128 (up) + 128 (skip) = 256
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)   # 64 (up) + 64 (skip) = 128

        # Output: 1x1 conv, linear activation (no sigmoid/relu)
        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, C, 30, 30).

        Returns:
            Output of shape (batch, 1, 30, 30).
        """
        # Pad 30x30 -> 32x32 (1 pixel on each side)
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")  # (B, C, 32, 32)

        # Encoder
        e1 = self.enc1(x)          # (B, 64, 32, 32)
        e2 = self.enc2(self.pool1(e1))  # (B, 128, 16, 16)

        # Bottleneck
        b = self.bottleneck(self.pool2(e2))  # (B, 256, 8, 8)

        # Decoder
        d2 = self.up2(b)           # (B, 128, 16, 16)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, 128, 16, 16)
        d1 = self.up1(d2)          # (B, 64, 32, 32)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, 64, 32, 32)

        out = self.out_conv(d1)    # (B, 1, 32, 32)

        # Crop 32x32 -> 30x30
        return out[:, :, 1:31, 1:31]
