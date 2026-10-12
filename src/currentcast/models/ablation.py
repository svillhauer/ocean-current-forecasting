"""Architecture ablations: which parts of U-Net + ConvLSTM actually matter?

``mode`` selects one of three variants that share the same encoder/decoder:

* ``unet_convlstm``  - skip connections + ConvLSTM memory (full model)
* ``cnn_convlstm``   - ConvLSTM memory, no skip connections
* ``standard_unet``  - skip connections, no recurrence; the input frames are
  stacked along the channel axis instead (``in_ch = 2 * seq_len``)

All variants output 4 channels: (u, v) mean and (u, v) log-variance.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _double_conv(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True),
        nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True),
    )


class _ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size=(3, 3), bias: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size,
                              padding=(kernel_size[0] // 2, kernel_size[1] // 2), bias=bias)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.split(self.conv(torch.cat([x, h], dim=1)), self.hidden_dim, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class _UpBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, use_skips: bool):
        super().__init__()
        self.use_skips = use_skips
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _double_conv(in_c + (out_c if use_skips else 0), out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if self.use_skips:
            dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
            x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AblationNet(nn.Module):
    MODES = ("unet_convlstm", "cnn_convlstm", "standard_unet")

    def __init__(self, mode: str = "unet_convlstm", seq_len: int = 3, base_ch: int = 32, lstm_ch: int = 128):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        self.recurrent = "lstm" in mode
        in_ch = 2 if self.recurrent else 2 * seq_len

        self.inc = _double_conv(in_ch, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), _double_conv(base_ch, base_ch * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), _double_conv(base_ch * 2, base_ch * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), _double_conv(base_ch * 4, base_ch * 8))
        self.bottleneck = (_ConvLSTMCell(base_ch * 8, lstm_ch) if self.recurrent
                           else _double_conv(base_ch * 8, lstm_ch))
        use_skips = "unet" in mode
        self.up1 = _UpBlock(lstm_ch, base_ch * 4, use_skips)
        self.up2 = _UpBlock(base_ch * 4, base_ch * 2, use_skips)
        self.up3 = _UpBlock(base_ch * 2, base_ch, use_skips)
        self.outc = nn.Conv2d(base_ch, 4, 1)

    def forward(self, x_seq):
        """``x_seq``: (B, T, 2, H, W) for every mode. Returns ``(mu, logvar)``."""
        b, t, c, h, w = x_seq.shape
        if self.recurrent:
            x1 = self.inc(x_seq.reshape(b * t, c, h, w))
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            seq = x4.unflatten(0, (b, t))
            hs = torch.zeros(b, self.bottleneck.hidden_dim, *x4.shape[-2:], device=x_seq.device, dtype=x_seq.dtype)
            state = (hs, torch.zeros_like(hs))
            for k in range(t):
                state = self.bottleneck(seq[:, k], state)
            bott = state[0]
            s1, s2, s3 = (s.unflatten(0, (b, t))[:, -1] for s in (x1, x2, x3))
        else:
            x1 = self.inc(x_seq.reshape(b, t * c, h, w))
            s2 = self.down1(x1)
            s3 = self.down2(s2)
            bott = self.bottleneck(self.down3(s3))
            s1 = x1
        x = self.up3(self.up2(self.up1(bott, s3), s2), s1)
        out = self.outc(x)
        return out[:, :2], out[:, 2:]
