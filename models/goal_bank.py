"""Goal Latent Bank (GLB)

A differentiable slot-based memory that stores multiple goal vectors g_i and
learns their importance via a gate γ_i.
"""
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalLatentBank(nn.Module):
    def __init__(self, num_slots: int = 16, goal_dim: int = 16):
        super().__init__()
        self.num_slots = num_slots
        self.goal_dim = goal_dim
        # slot matrix G (M, D), initialised to zeros
        self.G = nn.Parameter(torch.zeros(num_slots, goal_dim))
        # importance gate γ_i in (0,1)
        self.gamma = nn.Parameter(torch.zeros(num_slots))

    @property
    def normalized_gamma(self):
        # softmax for stability
        return F.softmax(self.gamma, dim=0)  # (M,)

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Content-based read. query: (B,D) -> g_t (B,D)"""
        # cosine similarity
        G_norm = F.normalize(self.G, p=2, dim=-1)
        q_norm = F.normalize(query, p=2, dim=-1)
        sim = q_norm @ G_norm.T  # (B,M)
        weights = F.softmax(sim, dim=-1)  # (B,M)
        g = weights @ self.G  # (B,D)
        return g

    def absorb(self, z: torch.Tensor, score: torch.Tensor):
        """Write new goal candidates z into the least-used slot.

        Args:
            z: (B,D) latent vector to store
            score: (B,1) affordance score in [0,1]
        """
        B, D = z.shape
        gamma = self.normalized_gamma  # (M,)
        min_id = torch.argmin(gamma)  # scalar
        # Only absorb if score > threshold (handled by caller)
        with torch.no_grad():
            self.G[min_id] = z.mean(0)
            self.gamma[min_id] = torch.tensor(0.1, device=z.device)

    def decay_gates(self, decay_factor: float = 0.99):
        """Apply exponential decay to the importance gates."""
        with torch.no_grad():
            self.gamma.data *= decay_factor

    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.read(query)
        return g, self.normalized_gamma 