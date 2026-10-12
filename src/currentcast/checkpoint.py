"""Self-describing checkpoints: architecture + config + normalisation + weights in one file."""
from __future__ import annotations

from pathlib import Path

import torch

from .data import ZScoreStats
from .models import AblationNet, UNetConvLSTMUncertainty

ARCHITECTURES = {
    "unet_convlstm_unc": UNetConvLSTMUncertainty,
    "ablation": AblationNet,
}


def build_model(arch: str, config: dict) -> torch.nn.Module:
    return ARCHITECTURES[arch](**config)


def save_checkpoint(path, model, arch: str, config: dict, stats: ZScoreStats, seq_len: int, **meta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": arch,
        "config": config,
        "seq_len": seq_len,
        "stats": stats.to_dict(),
        "state_dict": model.state_dict(),
        "meta": meta,
    }, path)


def load_checkpoint(path, device="cpu"):
    """Returns ``(model, stats, seq_len, meta)`` with the model in eval mode."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ck["arch"], ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ZScoreStats.from_dict(ck["stats"]), int(ck["seq_len"]), ck.get("meta", {})
