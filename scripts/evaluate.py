#!/usr/bin/env python3
"""Benchmark every model against persistence on a common held-out test period.

Each model is rolled out autoregressively for ``--horizon`` hours from every
start time in the test period (the final ``--test-frac`` of the series, which
no model was trained on). Metrics are computed in physical units (m/s) over
ocean pixels only, per lead time.

Usage:
    python scripts/evaluate.py --data data/cmems_iberia_uv_2024_2025.nc --weights weights/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from currentcast.checkpoint import load_checkpoint
from currentcast.data import load_cmems_uv
from currentcast.forecast import persistence, rollout
from currentcast.metrics import LeadTimeMetrics, skill

MODELS = {
    # name in results      checkpoint file                   display label
    "curriculum": ("unet_convlstm_curriculum.pt", "U-Net + ConvLSTM, curriculum (12 h context)"),
    "onestep": ("unet_convlstm_onestep.pt", "U-Net + ConvLSTM, one-step (12 h context)"),
    "ablation_unet_convlstm": ("ablation_unet_convlstm.pt", "Ablation: U-Net + ConvLSTM (3 h context)"),
    "ablation_cnn_convlstm": ("ablation_cnn_convlstm.pt", "Ablation: no skip connections (3 h)"),
    "ablation_standard_unet": ("ablation_standard_unet.pt", "Ablation: no recurrence (3 h)"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="CMEMS NetCDF with utotal/vtotal")
    p.add_argument("--weights", default="weights", help="Directory of exported checkpoints")
    p.add_argument("--out", default="results/benchmark.json")
    p.add_argument("--test-frac", type=float, default=0.2, help="Final fraction of the series held out")
    p.add_argument("--horizon", type=int, default=12, help="Forecast lead times (hours)")
    p.add_argument("--stride", type=int, default=1, help="Hours between forecast start times")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--regrid", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    uv_raw, times, lat, lon = load_cmems_uv(args.data, regrid_hw=(args.regrid, args.regrid), fillna=None)
    # Ocean mask: pixels that are valid at every time step (land/coast-interpolated cells are NaN)
    mask = torch.from_numpy(~np.isnan(uv_raw).any(axis=(0, 1)))
    uv = np.nan_to_num(uv_raw, nan=0.0).astype(np.float32)
    del uv_raw

    T = len(uv)
    test_start = int(T * (1 - args.test_frac))
    # Forecasts are issued at every time t whose first target t lies in the test period and whose
    # full horizon fits; each model reads whatever history it needs before t.
    issue_times = np.arange(test_start, T - args.horizon + 1, args.stride)
    print(f"Data {uv.shape}, ocean pixels {int(mask.sum())}/{mask.numel()}, device {device}")
    print(f"Test period {str(times[test_start])[:16]} -> {str(times[-1])[:16]}: "
          f"{len(issue_times)} forecasts x {args.horizon} h")

    uv_t = torch.from_numpy(uv)
    targets_idx = issue_times[:, None] + np.arange(args.horizon)[None]   # (N, L)

    def batches():
        for i in range(0, len(issue_times), args.batch):
            yield issue_times[i:i + args.batch], targets_idx[i:i + args.batch]

    results = {"config": vars(args) | {"n_forecasts": len(issue_times),
                                       "test_start": str(times[test_start]), "test_end": str(times[-1])},
               "models": {}}

    # --- Persistence baseline -------------------------------------------------
    m = LeadTimeMetrics(args.horizon, mask)
    for t0, tgt in batches():
        last = uv_t[t0 - 1][:, None]                                   # (B, 1, 2, H, W)
        m.update(persistence(last, args.horizon), uv_t[tgt])
    results["models"]["persistence"] = {"label": "Persistence (last observed field)", **m.result()}

    # --- Learned models ---------------------------------------------------------
    for name, (fname, label) in MODELS.items():
        path = Path(args.weights) / fname
        if not path.exists():
            print(f"  skip {name}: {path} not found")
            continue
        model, stats, seq_len, meta = load_checkpoint(path, device)
        m = LeadTimeMetrics(args.horizon, mask)
        tic = time.time()
        with torch.inference_mode():
            for t0, tgt in batches():
                hist_idx = t0[:, None] + np.arange(-seq_len, 0)[None]          # (B, seq_len)
                x = stats.normalize(uv_t[hist_idx]).to(device)
                mu_n, logvar_n = rollout(model, x, args.horizon)
                mu = stats.denormalize(mu_n).cpu()
                sigma = stats.denormalize_sigma(torch.exp(0.5 * logvar_n)).cpu()
                m.update(mu, uv_t[tgt], sigma)
        res = m.result()
        res["seconds_per_forecast"] = (time.time() - tic) / len(issue_times)
        results["models"][name] = {"label": label, "seq_len": seq_len, **meta, **res}
        print(f"  {name:26s} RMSE t+1={res['rmse_vector'][0]:.4f}  t+6={res['rmse_vector'][5]:.4f}  "
              f"t+12={res['rmse_vector'][-1]:.4f} m/s")

    ref = results["models"]["persistence"]["rmse_vector"]
    for r in results["models"].values():
        r["skill_vs_persistence"] = skill(r["rmse_vector"], ref)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
