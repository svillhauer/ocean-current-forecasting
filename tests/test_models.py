import math

import pytest
import torch

from currentcast.checkpoint import build_model, load_checkpoint, save_checkpoint
from currentcast.data import ZScoreStats
from currentcast.models import AblationNet, UNetConvLSTMUncertainty, gaussian_nll

X = torch.randn(2, 4, 2, 32, 32)  # (B, T, C, H, W)


def test_uncertainty_model_shapes_and_logvar_clamp():
    model = UNetConvLSTMUncertainty(base_ch=4, lstm_ch=8)
    mu, logvar = model(X * 1e3)  # extreme input to push the variance head
    assert mu.shape == logvar.shape == (2, 2, 32, 32)
    assert logvar.min() >= model.LOGVAR_MIN and logvar.max() <= model.LOGVAR_MAX


@pytest.mark.parametrize("mode", AblationNet.MODES)
def test_ablation_variants_share_the_interface(mode):
    model = AblationNet(mode=mode, seq_len=4, base_ch=4, lstm_ch=8).eval()
    mu, logvar = model(X)
    assert mu.shape == logvar.shape == (2, 2, 32, 32)


def test_model_uses_temporal_context():
    model = UNetConvLSTMUncertainty(base_ch=4, lstm_ch=8).eval()
    x2 = X.clone()
    x2[:, 0] += 10.0  # change only the oldest frame
    assert (model(X)[0] - model(x2)[0]).abs().max() > 0


def test_gaussian_nll_is_minimised_at_the_true_variance():
    y, mu = torch.zeros(1000), torch.full((1000,), 0.3)
    best = min((gaussian_nll(mu, torch.full_like(mu, lv), y).item(), lv) for lv in torch.linspace(-6, 2, 801).tolist())
    assert math.isclose(best[1], math.log(0.3**2), abs_tol=0.02)


def test_checkpoint_roundtrip_reproduces_outputs(tmp_path):
    cfg = dict(base_ch=4, lstm_ch=8)
    model = build_model("unet_convlstm_unc", cfg).eval()
    stats = ZScoreStats(0.0, 1.0, 0.0, 1.0)
    save_checkpoint(tmp_path / "m.pt", model, "unet_convlstm_unc", cfg, stats, seq_len=4, note="test")
    loaded, st, seq_len, meta = load_checkpoint(tmp_path / "m.pt")
    assert seq_len == 4 and st == stats and meta == {"note": "test"}
    torch.testing.assert_close(loaded(X)[0], model(X)[0])
