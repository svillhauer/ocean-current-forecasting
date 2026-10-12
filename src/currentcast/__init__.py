"""Probabilistic short-term forecasting of ocean surface currents.

A U-Net encoder/decoder with a ConvLSTM bottleneck predicts the next hourly
(u, v) surface-current field and a per-pixel variance, and is rolled out
autoregressively for multi-hour forecasts.
"""

__version__ = "0.1.0"
