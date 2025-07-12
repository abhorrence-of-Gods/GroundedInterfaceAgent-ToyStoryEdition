"""
Generates a sequence of discrete subgoals (macro-actions) using Gumbel-Softmax,
enabling high-level, hierarchical planning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MacroActionTower(nn.Module):
    def __init__(
        self,
        latent_state_dim: int = 4096,
        goal_dim: int = 16,
        num_subgoals: int = 64,
        subgoal_dim: int = 128,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.num_subgoals = num_subgoals
        self.subgoal_dim = subgoal_dim
        
        self.subgoal_vocab = nn.Embedding(num_subgoals, subgoal_dim)

        self.planner_net = nn.Sequential(
            nn.Linear(latent_state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_subgoals)
        )

    def forward(
        self, 
        z: torch.Tensor, 
        g: torch.Tensor, 
        tau: float = 1.0, 
        hard: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Selects a subgoal and returns its embedding.

        Args:
            z: Current latent state (B, D_z)
            g: Main goal vector (B, D_g)
            tau: Temperature for Gumbel-Softmax.
            hard: If True, returns one-hot vectors.

        Returns:
            A tuple of:
            - g_sub (Tensor): The embedding of the chosen subgoal. (B, D_subgoal)
            - logits (Tensor): The pre-softmax logits. (B, num_subgoals)
        """
        x = torch.cat([z, g], dim=-1)
        logits = self.planner_net(x)
        y_soft = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        g_sub = y_soft @ self.subgoal_vocab.weight

        return g_sub, logits 