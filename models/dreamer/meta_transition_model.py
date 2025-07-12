"""Meta-Transition Model for the Meta-Cognitive Dreamer architecture.

This model predicts the next latent state (z_t+1) AND the next latent value
state (u_t+1), conditioned on the current states, action, and goal.

p(z_t+1, u_t+1 | z_t, u_t, a_t, g_t)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class MetaTransitionModel(nn.Module):
    def __init__(
        self,
        latent_state_dim: int = 4096,
        latent_action_dim: int = 256,
        value_latent_dim: int = 256,
        goal_dim: int = 16,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.input_dim = (
            latent_state_dim + latent_action_dim + value_latent_dim + goal_dim
        )
        self.output_dim = latent_state_dim + value_latent_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )
        self.latent_state_dim = latent_state_dim
        self.value_latent_dim = value_latent_dim

    def forward(
        self, z: torch.Tensor, u: torch.Tensor, a: torch.Tensor, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the next latent state (z_t+1) and value state (u_t+1).

        Args:
            z: current latent state (B, D_z)
            u: current value latent state (B, D_u)
            a: latent action (B, D_a)
            g: goal vector (B, D_g)

        Returns:
            A tuple (z_next, u_next).
        """
        # Concatenate all inputs
        x = torch.cat([z, u, a, g], dim=-1)

        # Predict the change (delta) for stability
        delta = self.net(x)
        delta_z = delta[:, : self.latent_state_dim]
        delta_u = delta[:, self.latent_state_dim :]

        # Residual connection: add delta to current state
        z_next = z + delta_z
        u_next = u + delta_u

        return z_next, u_next 