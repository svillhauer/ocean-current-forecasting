"""Training loop shared by one-step, curriculum and ablation runs."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .forecast import rollout
from .models import gaussian_nll


def rollout_loss(model, x, y, steps: int):
    """Mean Gaussian NLL and RMSE over an ``steps``-long autoregressive rollout.

    Gradients flow through every step, so the model is penalised for errors that
    arise from consuming its own earlier predictions (the core idea of the
    curriculum). ``y`` is (B, >=steps, 2, H, W) in normalised units.
    """
    mu, logvar = rollout(model, x, steps, grad=torch.is_grad_enabled())
    target = y[:, :steps]
    nll = gaussian_nll(mu, logvar, target)
    rmse = torch.sqrt(torch.mean((mu - target) ** 2))
    return nll, rmse


@torch.no_grad()
def evaluate(model, loader: DataLoader, steps: int, device) -> dict:
    model.eval()
    tot_nll = tot_mse = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        nll, rmse = rollout_loss(model, x, y, steps)
        tot_nll += nll.item() * len(x)
        tot_mse += rmse.item() ** 2 * len(x)
        n += len(x)
    return {"val_nll": tot_nll / max(n, 1), "val_rmse": (tot_mse / max(n, 1)) ** 0.5}


@dataclass
class TrainConfig:
    schedule: list[tuple[int, int]]          # [(ar_steps, epochs), ...]; [(1, E)] = one-step training
    eval_steps: int = 12                      # validation always rolls out the full horizon
    lr: float = 1e-3
    lr_decay_per_stage: float = 0.5
    grad_clip: float = 1.0
    val_every: int = 5
    amp: bool = False
    out_dir: str = "runs/run"
    extra: dict = field(default_factory=dict)


def train(model, train_loader, val_loader, cfg: TrainConfig, device, on_checkpoint=None) -> dict:
    """Run every curriculum stage in ``cfg.schedule``.

    The checkpoint with the lowest validation RMSE over the full ``eval_steps``
    horizon is kept as best, so checkpoints from different stages are compared on
    the same task. ``on_checkpoint(path, model, info)`` writes the file.
    """
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history_path = out / "history.jsonl"
    history_path.unlink(missing_ok=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = float("inf")
    epoch = 0

    for stage, (ar_steps, n_epochs) in enumerate(cfg.schedule):
        if stage > 0:
            for g in opt.param_groups:
                g["lr"] *= cfg.lr_decay_per_stage
        print(f"Stage {stage}: {ar_steps}-step rollout loss, {n_epochs} epochs, lr={opt.param_groups[0]['lr']:.1e}")

        for local in range(n_epochs):
            model.train()
            tic, tot, n = time.time(), 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device.type, enabled=use_amp):
                    loss, _ = rollout_loss(model, x, y, ar_steps)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(x)
                n += len(x)

            row = {"epoch": epoch, "stage": stage, "ar_steps": ar_steps, "train_nll": tot / max(n, 1),
                   "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - tic, 1)}
            if local % cfg.val_every == 0 or local == n_epochs - 1:
                row |= evaluate(model, val_loader, cfg.eval_steps, device)
                if row["val_rmse"] < best:
                    best = row["val_rmse"]
                    on_checkpoint(out / "best.pt", model, row)
            print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
            with history_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            epoch += 1

        on_checkpoint(out / f"stage{stage}_ar{ar_steps}.pt", model, {"stage": stage, "ar_steps": ar_steps})

    on_checkpoint(out / "last.pt", model, {"epoch": epoch})
    return {"best_val_rmse": best, "epochs": epoch}
