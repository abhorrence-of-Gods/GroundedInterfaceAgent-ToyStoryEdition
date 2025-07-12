"""Utility functions for Meta-Cognitive Dreamer loss calculations."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def kl_divergence_gaussian(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor | None = None, logvar_p: torch.Tensor | None = None) -> torch.Tensor:
    """KL( q || p ) for factorised Gaussians.

    If *p* params are None, the prior is assumed to be N(0,I).
    Returns per-sample KL (batch vector).
    """
    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar_q)

    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * (
        (var_q / var_p)
        + ((mu_p - mu_q) ** 2) / var_p
        + logvar_p
        - logvar_q
        - 1.0
    ).sum(dim=-1)
    return kl  # (B,)


def lambda_return(reward: torch.Tensor, value: torch.Tensor, gamma: float = 0.99, lmbda: float = 0.95) -> torch.Tensor:
    """Compute TD(λ) returns given reward sequence and bootstrap value.

    reward: (T,B)
    value:  (T+1,B)  bootstrap includes V_{T}
    returns: (T,B)
    """
    T, B = reward.shape
    returns = torch.zeros_like(reward)
    next_return = value[-1]
    for t in reversed(range(T)):
        delta = reward[t] + gamma * value[t + 1] - value[t]
        next_return = value[t] + delta * lmbda + (1 - lmbda) * (reward[t] + gamma * value[t + 1])
        returns[t] = next_return
    return returns 