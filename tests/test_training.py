import torch
from torch.utils.data import DataLoader

from currentcast.data import SlidingWindowDataset
from currentcast.models import UNetConvLSTMUncertainty
from currentcast.training import TrainConfig, rollout_loss, train


def test_curriculum_loss_backpropagates_through_rollout():
    model = UNetConvLSTMUncertainty(base_ch=4, lstm_ch=8)
    x, y = torch.randn(2, 3, 2, 16, 16), torch.randn(2, 4, 2, 16, 16)
    nll, rmse = rollout_loss(model, x, y, steps=3)
    nll.backward()
    assert all(p.grad is not None for p in model.parameters())
    assert rmse.item() > 0


def test_training_runs_every_stage_and_writes_checkpoints(tmp_path):
    uv = torch.randn(40, 2, 16, 16).numpy()
    ds = SlidingWindowDataset(uv, seq_len=3, horizon=4)
    loader = DataLoader(ds, batch_size=8)
    saved = []
    cfg = TrainConfig(schedule=[(1, 1), (2, 1)], eval_steps=4, out_dir=str(tmp_path))
    summary = train(UNetConvLSTMUncertainty(base_ch=4, lstm_ch=8), loader, loader, cfg, torch.device("cpu"),
                    on_checkpoint=lambda path, m, info: saved.append(path.name))
    assert summary["epochs"] == 2
    assert {"best.pt", "stage0_ar1.pt", "stage1_ar2.pt", "last.pt"} <= set(saved)
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 2
