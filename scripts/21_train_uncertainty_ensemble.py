#!/usr/bin/env python3
"""Train a small deep ensemble and evaluate uncertainty on held-out perturbations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from perturbatlas.benchmark import row_pearson
from perturbatlas.models import DenoisingAutoencoder
from perturbatlas.reproducibility import run_metadata, save_metadata
from perturbatlas.uncertainty import perturbation_uncertainty, uncertainty_diagnostics


def train_member(xtr, ytr, xva, yva, n_genes, seed, hidden, latent, batch_size, lr, epochs, patience):
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
        batch_size=batch_size,
        shuffle=True,
    )
    model = DenoisingAutoencoder(n_genes, hidden=hidden, latent=latent)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(torch.from_numpy(xva)), torch.from_numpy(yva)))
        if val < best - 1e-6:
            best = val
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint found.")
    model.load_state_dict(best_state)
    return model, best, best_epoch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("half_a", type=Path)
    p.add_argument("half_b", type=Path)
    p.add_argument("reliability", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/uncertainty/replogle_k562"))
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=20261013)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(32, torch.get_num_threads()))

    a, b = ad.read_h5ad(args.half_a), ad.read_h5ad(args.half_b)
    xa, xb = np.asarray(a.X, dtype=np.float32), np.asarray(b.X, dtype=np.float32)
    split = a.obs["split"].astype(str).to_numpy()
    tr, va, te = split == "train", split == "val", split == "test"

    mean = xa[tr].mean(axis=0, keepdims=True)
    std = xa[tr].std(axis=0, keepdims=True)
    std[std < 1e-4] = 1.0
    xtr, ytr = (xa[tr] - mean) / std, (xb[tr] - mean) / std
    xva, yva = (xa[va] - mean) / std, (xb[va] - mean) / std
    xte = (xa[te] - mean) / std
    target = xb[te]

    preds = []
    member_rows = []
    for m in range(args.members):
        seed = args.seed + m
        model, best_val, best_epoch = train_member(
            xtr, ytr, xva, yva, xa.shape[1], seed,
            args.hidden, args.latent, args.batch_size, args.lr,
            args.epochs, args.patience,
        )
        model.eval()
        with torch.no_grad():
            pred_std = model(torch.from_numpy(xte)).numpy()
        pred = pred_std * std + mean
        preds.append(pred.astype(np.float32))
        r = row_pearson(pred, target)
        member_rows.append({
            "member": m,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "median_pearson": float(np.nanmedian(r)),
        })
        print(f"member={m} seed={seed} best_epoch={best_epoch} val={best_val:.6f} median_r={np.nanmedian(r):.6f}")

    stack = np.stack(preds, axis=0)
    ensemble_pred = stack.mean(axis=0)
    unc = perturbation_uncertainty(stack)
    r = row_pearson(ensemble_pred, target)
    rmse = np.sqrt(np.mean((ensemble_pred - target) ** 2, axis=1))

    table = pd.DataFrame({
        "perturbation": a.obs_names[te],
        "pearson": r,
        "rmse": rmse,
        "ensemble_variance": unc,
    }).set_index("perturbation")

    rel = pd.read_csv(args.reliability, index_col=0)
    table = table.join(rel[["pearson_median"]].rename(columns={"pearson_median": "intrinsic_reproducibility"}), how="left")
    diagnostics = uncertainty_diagnostics(table)
    diagnostics.update({
        "members": int(args.members),
        "ensemble_median_pearson": float(np.nanmedian(r)),
        "ensemble_mean_pearson": float(np.nanmean(r)),
        "ensemble_median_rmse": float(np.median(rmse)),
    })

    table.to_csv(args.output_dir / "per_perturbation.csv")
    pd.DataFrame(member_rows).to_csv(args.output_dir / "members.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    save_metadata(run_metadata(vars(args)), args.output_dir / "run_metadata.json")

    print("\n=== ENSEMBLE UNCERTAINTY ===")
    for k, v in diagnostics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
