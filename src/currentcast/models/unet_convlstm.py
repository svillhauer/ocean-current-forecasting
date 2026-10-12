from __future__ import annotations

import torch
import torch.nn as nn

from .convlstm import ConvLSTM


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNetConvLSTM(nn.Module):
    """U-Net with a ConvLSTM in the bottleneck.

    Input ``(B, T, 2, H, W)`` -> output ``(B, out_ch, H, W)``. Every input frame
    is encoded by a shared 4-level U-Net encoder; a ConvLSTM integrates the
    bottleneck features over time; the decoder upsamples the final hidden state
    using skip connections from the most recent frame. H and W must be
    divisible by 16.
    """

    def __init__(self, in_ch: int = 2, out_ch: int = 2, base: int = 32, lstm_ch: int = 256):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.enc4 = conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base * 8, lstm_ch)
        self.temporal = ConvLSTM(in_ch=lstm_ch, hid_ch=lstm_ch, k=3)

        self.up3 = nn.ConvTranspose2d(lstm_ch, base * 8, 2, stride=2)
        self.dec3 = conv_block(base * 16, base * 8)
        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec2 = conv_block(base * 8, base * 4)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec1 = conv_block(base * 4, base * 2)
        self.up0 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec0 = conv_block(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    def _encode(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        return self.bottleneck(self.pool(s4)), (s1, s2, s3, s4)

    def forward(self, x_seq):
        b, t = x_seq.shape[:2]
        # Encode all frames in one batched pass (shared weights).
        bott, skips = self._encode(x_seq.flatten(0, 1))
        h_last = self.temporal(bott.unflatten(0, (b, t)))
        s1, s2, s3, s4 = (s.unflatten(0, (b, t))[:, -1] for s in skips)

        x = self.dec3(torch.cat([self.up3(h_last), s4], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s3], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s2], dim=1))
        x = self.dec0(torch.cat([self.up0(x), s1], dim=1))
        return self.head(x)


class UNetConvLSTMUncertainty(nn.Module):
    """UNetConvLSTM with a heteroscedastic Gaussian head.

    Returns ``(mu, logvar)``, each ``(B, 2, H, W)``: the predicted mean (u, v)
    and a per-pixel log-variance (aleatoric uncertainty), clamped for stability.
    """

    LOGVAR_MIN, LOGVAR_MAX = -10.0, 5.0

    def __init__(self, base_ch: int = 32, lstm_ch: int = 256):
        super().__init__()
        self.backbone = UNetConvLSTM(in_ch=2, out_ch=4, base=base_ch, lstm_ch=lstm_ch)

    def forward(self, x_seq):
        out = self.backbone(x_seq)
        return out[:, :2], torch.clamp(out[:, 2:], self.LOGVAR_MIN, self.LOGVAR_MAX)


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log-likelihood (constant term dropped), averaged."""
    return 0.5 * torch.mean(torch.exp(-logvar) * (y - mu) ** 2 + logvar)
