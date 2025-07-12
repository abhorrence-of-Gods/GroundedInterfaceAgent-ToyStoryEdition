from __future__ import annotations

"""Utility for building observation+action token sequences for ULWT.

This is a *stub* implementation that projects arbitrary modality embeddings to
ULWT embedding dimension and concatenates them in the prescribed order:

    [ vision_embed , language_embed , action_prev_embed , reward_prev_embed ]

The module can be extended to include additional modalities (audio, sensor, etc.)
by adding corresponding projection layers.
"""

import torch
import torch.nn as nn

__all__ = ["SequenceBuilder"]


class SequenceBuilder(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        action_dim: int,
        ulwt_embed_dim: int,
    ) -> None:
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, ulwt_embed_dim)
        self.lang_proj = nn.Linear(language_dim, ulwt_embed_dim)
        self.action_proj = nn.Linear(action_dim, ulwt_embed_dim)

    def forward(
        self,
        vision_embed: torch.Tensor,  # (B, D_v)
        lang_embed: torch.Tensor,    # (B, D_l)
        action_prev_embed: torch.Tensor,  # (B, D_a)
    ) -> torch.Tensor:
        """Return sequence tensor suitable for ULWT.

        Output shape: (B, 3, ulwt_embed_dim)
        """
        v_tok = self.vision_proj(vision_embed).unsqueeze(1)
        l_tok = self.lang_proj(lang_embed).unsqueeze(1)
        a_tok = self.action_proj(action_prev_embed).unsqueeze(1)
        # order: obs vision, obs language, act_prev
        seq = torch.cat([v_tok, l_tok, a_tok], dim=1)
        return seq 