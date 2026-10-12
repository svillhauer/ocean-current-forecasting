from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell (Shi et al., 2015): LSTM gates computed by a 2-D convolution."""

    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size=k, padding=k // 2)

    def forward(self, x, h, c):
        if h is None:
            b, _, height, width = x.shape
            h = torch.zeros(b, self.hid_ch, height, width, device=x.device, dtype=x.dtype)
            c = torch.zeros_like(h)
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], dim=1)), 4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    """Runs a ConvLSTMCell over a (B, T, C, H, W) sequence and returns the last hidden state."""

    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hid_ch, k=k)

    def forward(self, x_seq):
        h = c = None
        for t in range(x_seq.shape[1]):
            h, c = self.cell(x_seq[:, t], h, c)
        return h
