#!/usr/bin/env python3
"""Train a probabilistic surface-current forecaster.

One-step training (the model only ever sees ground-truth inputs):
    python scripts/train.py --data data/cmems_iberia_uv_2024_2025.nc --out runs/onestep \
        --curriculum 1 --epochs-per-stage 100

Autoregressive curriculum (rollout length grows 1 -> 2 -> 4 -> 8 -> 12 hours):
    python scripts/train.py --data data/cmems_iberia_uv_2024_2025.nc --out runs/curriculum \
        --curriculum 1,2,4,8,12 --epochs-per-stage 100 --amp

Architecture ablations (3-hour context):
    python scripts/train.py --data ... --out runs/abl_unet --arch ablation --mode standard_unet --seq-len 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from currentcast.checkpoint import build_model, save_checkpoint
from currentcast.data import SlidingWindowDataset, ZScoreStats, load_cmems_uv, time_split
from currentcast.training import TrainConfig, train


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="CMEMS NetCDF with utotal/vtotal")
    p.add_argument("--out", required=True, help="Run directory")
    # Data
    p.add_argument("--regrid", type=int, default=64, help="Regrid to N x N (must be divisible by 16)")
    p.add_argument("--train-frac", type=float, default=0.7, help="Chronological train fraction")
    p.add_argument("--seq-len", type=int, default=12, help="Hours of input history")
    p.add_argument("--max-train-samples", type=int, default=None, help="Subsample for quick smoke runs")
    # Model
    p.add_argument("--arch", choices=["unet_convlstm_unc", "ablation"], default="unet_convlstm_unc")
    p.add_argument("--mode", choices=["unet_convlstm", "cnn_convlstm", "standard_unet"], default="unet_convlstm",
                   help="Ablation variant (only with --arch ablation)")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--lstm-ch", type=int, default=256)
    # Optimisation
    p.add_argument("--curriculum", default="1", help="Comma-separated rollout lengths, one stage each")
    p.add_argument("--epochs-per-stage", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=12, help="Validation rollout length")
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-decay-per-stage", type=float, default=0.5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--amp", action="store_true", help="Mixed precision on CUDA")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    uv, _, _, _ = load_cmems_uv(args.data, regrid_hw=(args.regrid, args.regrid))
    train_uv, val_uv = time_split(uv, args.train_frac)
    stats = ZScoreStats.fit(train_uv)                      # training period only: no leakage
    schedule_steps = [int(s) for s in args.curriculum.split(",")]
    horizon = max(max(schedule_steps), args.eval_steps)

    train_ds = SlidingWindowDataset(stats.normalize(train_uv), args.seq_len, horizon)
    val_ds = SlidingWindowDataset(stats.normalize(val_uv), args.seq_len, horizon)
    if args.max_train_samples:
        train_ds = Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
        val_ds = Subset(val_ds, range(min(args.max_train_samples, len(val_ds))))
    loader_kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    print(f"data {uv.shape}  train {len(train_ds)}  val {len(val_ds)}  device {device}")

    if args.arch == "ablation":
        config = dict(mode=args.mode, seq_len=args.seq_len, base_ch=args.base_ch, lstm_ch=args.lstm_ch)
    else:
        config = dict(base_ch=args.base_ch, lstm_ch=args.lstm_ch)
    model = build_model(args.arch, config).to(device)
    print(f"{args.arch} {config}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    def on_checkpoint(path, m, info):
        save_checkpoint(path, m, args.arch, config, stats, args.seq_len,
                        train_frac=args.train_frac, curriculum=schedule_steps, **info)

    cfg = TrainConfig(
        schedule=[(s, args.epochs_per_stage) for s in schedule_steps],
        eval_steps=args.eval_steps, lr=args.lr, lr_decay_per_stage=args.lr_decay_per_stage,
        grad_clip=args.grad_clip, val_every=args.val_every, amp=args.amp, out_dir=args.out,
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "config.json").write_text(json.dumps(vars(args) | {"stats": stats.to_dict()}, indent=2))
    summary = train(model, train_loader, val_loader, cfg, device, on_checkpoint)
    print(f"Done. Best {args.eval_steps}-step val RMSE (normalised): {summary['best_val_rmse']:.4f}")


if __name__ == "__main__":
    main()
