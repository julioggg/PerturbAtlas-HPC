"""PyTorch models used by PerturbAtlas-HPC."""

from __future__ import annotations

import torch
from torch import nn


class DenoisingAutoencoder(nn.Module):
    """Small MLP autoencoder for perturbation-response signatures."""

    def __init__(self, n_genes: int, hidden: int = 512, latent: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_genes),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ResidualDenoiser(nn.Module):
    """Identity-preserving denoiser that learns a residual correction."""

    def __init__(
        self,
        n_genes: int,
        hidden: int = 256,
        latent: int = 32,
        dropout: float = 0.1,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
            nn.GELU(),
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_genes),
        )

    def forward(self, x):
        return x + self.residual_scale * self.net(x)
