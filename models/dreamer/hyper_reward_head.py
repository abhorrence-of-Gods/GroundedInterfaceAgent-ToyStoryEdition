"""HyperNetwork-based Reward Head for Meta-Cognitive Dreamer.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class HyperRewardHead(nn.Module):
    """
    Generates the weights of a reward predictor dynamically from a latent
    value state `u_t` (HyperNetwork). It then predicts the reward from `z_t`.

    r_t = Linear(z_t; W_r, b_r) where W_r, b_r = HyperNet(u_t)
    """

    def __init__(
        self,
        latent_state_dim: int = 4096,
        value_latent_dim: int = 256,
        hyper_hidden_dim: int = 512,
    ):
        super().__init__()
        self.latent_state_dim = latent_state_dim
        self.value_latent_dim = value_latent_dim

        # HyperNetwork: u_t -> (W_r, b_r)
        self.hyper_net = nn.Sequential(
            nn.Linear(value_latent_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, latent_state_dim + 1),  # W_r weights + b_r bias
        )

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent state from world model, shape (B, latent_state_dim)
            u: Latent value state from value-RSSM, shape (B, value_latent_dim)

        Returns:
            Predicted scalar reward, shape (B, 1)
        """
        # 1. Generate dynamic weights from u_t
        # (B, D_z + 1)
        params = self.hyper_net(u)

        # 2. Split into weights W_r and bias b_r
        # W_r: (B, D_z), b_r: (B, 1)
        w_r = params[:, : self.latent_state_dim]
        b_r = params[:, self.latent_state_dim :]

        # 3. Apply dynamic linear layer: r = z * W_r + b_r
        # Unsqueeze w_r to (B, D_z, 1) for batch matmul
        # z (B, D_z) -> (B, 1, D_z)
        # r = (B, 1, D_z) @ (B, D_z, 1) -> (B, 1, 1)
        # Squeeze back to (B, 1)
        r = torch.bmm(z.unsqueeze(1), w_r.unsqueeze(2)).squeeze(-1) + b_r

        return r 