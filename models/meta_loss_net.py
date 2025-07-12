"""Meta Loss Net – outputs dynamic weights for each base loss component."""
from __future__ import annotations

import torch
import torch.nn as nn


class MetaLossNet(nn.Module):
    def __init__(self, in_dim: int = 4096 + 16, n_losses: int = 10, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_losses),
            nn.Softplus(),  # ensure non-negative weights
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return positive weights (B, n_losses)."""
        x = torch.cat([state, goal], dim=-1)
        return self.net(x) 