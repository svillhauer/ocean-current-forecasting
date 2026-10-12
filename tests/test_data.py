import numpy as np
import pytest
import torch

from currentcast.data import SlidingWindowDataset, ZScoreStats, time_split


def series(t=30, h=4, w=4):
    # Frame k is filled with the value k, so indices are recoverable from contents.
    return np.arange(t, dtype=np.float32)[:, None, None, None] * np.ones((1, 2, h, w), np.float32)


def test_window_targets_are_the_frames_right_after_the_input():
    ds = SlidingWindowDataset(series(), seq_len=5, horizon=3)
    x, y = ds[7]
    assert x[:, 0, 0, 0].tolist() == [7, 8, 9, 10, 11]
    assert y[:, 0, 0, 0].tolist() == [12, 13, 14]   # first target = frame immediately after input


def test_last_window_uses_the_last_frame_exactly():
    uv = series(30)
    ds = SlidingWindowDataset(uv, seq_len=5, horizon=3)
    assert len(ds) == 30 - 5 - 3 + 1
    _, y = ds[len(ds) - 1]
    assert y[-1, 0, 0, 0] == 29


def test_too_short_series_raises():
    with pytest.raises(ValueError):
        SlidingWindowDataset(series(5), seq_len=4, horizon=2)


def test_time_split_is_chronological_and_disjoint():
    uv = series(100)
    train, val = time_split(uv, 0.7)
    assert len(train) == 70 and len(val) == 30
    assert train[-1, 0, 0, 0] < val[0, 0, 0, 0]


def test_zscore_roundtrip_numpy_and_torch():
    rng = np.random.default_rng(0)
    uv = rng.normal([0.1, -0.05], [0.2, 0.15], size=(50, 4, 4, 2)).transpose(0, 3, 1, 2).astype(np.float32)
    st = ZScoreStats.fit(uv)
    z = st.normalize(uv)
    assert abs(z[:, 0].mean()) < 1e-5 and abs(z[:, 0].std() - 1) < 1e-4
    np.testing.assert_allclose(st.denormalize(z), uv, atol=1e-5)
    zt = st.normalize(torch.from_numpy(uv))
    torch.testing.assert_close(st.denormalize(zt), torch.from_numpy(uv), atol=1e-5, rtol=0)


def test_sigma_scales_by_std_only():
    st = ZScoreStats(u_mean=5.0, u_std=2.0, v_mean=-3.0, v_std=0.5)
    s = st.denormalize_sigma(np.ones((2, 1, 1), np.float32))
    assert s[0, 0, 0] == 2.0 and s[1, 0, 0] == 0.5


def test_stats_dict_roundtrip():
    st = ZScoreStats(0.1, 0.2, 0.3, 0.4)
    assert ZScoreStats.from_dict(st.to_dict()) == st
