#!/usr/bin/env python3
"""Train a PyTorch denoising autoencoder on held-out perturbation identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from perturbatlas.benchmark import row_pearson
from perturbatlas.models import DenoisingAutoencoder
from perturbatlas.reproducibility import run_metadata, save_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("half_a", type=Path)
    p.add_argument("half_b", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/models/replogle_k562_pytorch"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=20261013)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(32, torch.get_num_threads()))

    a, b = ad.read_h5ad(args.half_a), ad.read_h5ad(args.half_b)
    xa, xb = np.asarray(a.X, dtype=np.float32), np.asarray(b.X, dtype=np.float32)
    split = a.obs["split"].astype(str).to_numpy()
    tr, va, te = split == "train", split == "val", split == "test"

    mean = xa[tr].mean(axis=0, keepdims=True)
    std = xa[tr].std(axis=0, keepdims=True)
    std[std < 1e-4] = 1.0
    xtr = (xa[tr] - mean) / std
    ytr = (xb[tr] - mean) / std
    xva = (xa[va] - mean) / std
    yva = (xb[va] - mean) / std
    xte = (xa[te] - mean) / std

    train_loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)), batch_size=args.batch_size, shuffle=True)
    model = DenoisingAutoencoder(xa.shape[1], hidden=args.hidden, latent=args.latent)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(torch.from_numpy(xva)), torch.from_numpy(yva)))
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_std = model(torch.from_numpy(xte)).numpy()
    pred = pred_std * std + mean
    target = xb[te]
    r = row_pearson(pred, target)
    rmse = np.sqrt(np.mean((pred - target) ** 2, axis=1))

    result = {
        "model": "pytorch_denoising_autoencoder",
        "n_test": int(te.sum()),
        "median_pearson": float(np.nanmedian(r)),
        "mean_pearson": float(np.nanmean(r)),
        "median_rmse": float(np.median(rmse)),
        "best_val_loss": float(best),
        "epochs": int(args.epochs),
        "hidden": int(args.hidden),
        "latent": int(args.latent),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    pd.DataFrame({"perturbation": a.obs_names[te], "pearson": r, "rmse": rmse}).to_csv(args.output_dir / "per_perturbation.csv", index=False)
    torch.save(best_state, args.output_dir / "model.pt")
    save_metadata(run_metadata(vars(args)), args.output_dir / "run_metadata.json")
    print("\n=== PYTORCH TEST RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
