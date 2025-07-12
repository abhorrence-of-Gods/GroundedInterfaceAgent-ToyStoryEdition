from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from models.embeddings.reward_token import RewardTokenEmbedding
from models.heads.human_feedback_head import HumanFeedbackHead


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        return x + self.pe[: x.size(1)].unsqueeze(0)


class UnifiedLatentWorldTransformer(nn.Module):
    """Transformer that jointly models world state, latent reward, and human feedback.

    The sequence per timestep is expected to be:
        [obs_tokens..., act_prev_token, reward_prev_token]
    Where reward_prev_token is produced by :class:`RewardTokenEmbedding`.
    The CLS token is appended at the end of the sequence and its hidden state is
    used for downstream heads (e.g. value, reward, policy, human-feedback).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 128,
        reward_bins: int = 15,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.reward_embed = RewardTokenEmbedding(
            embed_dim=embed_dim, num_bins=reward_bins
        )

        # projection for observation/action tokens assumed to be same dim already
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max_len + 1)

        # heads
        self.hfp_head = HumanFeedbackHead(hidden_dim=embed_dim)
        # value, reward, policy heads are assumed to live elsewhere and take CLS hidden.

        # Parameter init
        nn.init.normal_(self.cls_token, std=0.02)

    def build_sequence(
        self,
        obs_act_tokens: torch.Tensor,  # (B, L, D)
        reward_prev: torch.Tensor,  # (B, 1) scalar
    ) -> torch.Tensor:
        """Concatenate reward embedding and CLS to token sequence."""
        B = obs_act_tokens.size(0)
        r_emb = self.reward_embed(reward_prev)  # (B, D)
        r_emb = r_emb.unsqueeze(1)  # (B,1,D)
        cls = self.cls_token.repeat(B, 1, 1)  # (B,1,D)
        seq = torch.cat([obs_act_tokens, r_emb, cls], dim=1)
        return seq

    def forward(
        self,
        obs_act_tokens: torch.Tensor,  # (B, L, D)
        reward_prev: torch.Tensor,  # (B,1)
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Returns dict with keys: sequence_output, cls_hidden, hfp_prob
        """
        x = self.build_sequence(obs_act_tokens, reward_prev)
        x = self.pos_enc(x)
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)

        cls_hidden = out[:, -1]  # last token
        hfp_prob = self.hfp_head(cls_hidden)

        return {
            "sequence_output": out,
            "cls_hidden": cls_hidden,
            "hfp_prob": hfp_prob,
        } 