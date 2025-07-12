from __future__ import annotations

import torch
import torch.nn.functional as F


def self_consistency_loss(current_vec: torch.Tensor, persona_vec: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Compute self-consistency loss as 1 - cosine similarity.

    Args:
        current_vec: (B, D) tensor representing current latent/personality.
        persona_vec: (B, D) reference persona vectors (broadcastable).
    Returns:
        Scalar loss tensor.
    """
    current_norm = F.normalize(current_vec, dim=-1)
    persona_norm = F.normalize(persona_vec, dim=-1)
    loss = 1.0 - (current_norm * persona_norm).sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(reduction) 