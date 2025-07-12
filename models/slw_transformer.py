"""Shared Latent Workspace (SLW) Transformer.

This module ingests a mixed sequence of *REAL* and *IMAG* latent tokens coming
from multiple robots and produces fused context vectors that are consumed by
policy / value / reward heads.

Design goals:
1. Support variable-length sequences (padding with mask).
2. Embed token-type (REAL vs IMAG) and optional timestamp Δτ.
3. Fast forward pass for 25–30 Hz real-time loop.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding with a learnable linear projection."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, embed_dim)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, delta_tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_tau: Time difference tensor, shape (B, L, 1)
        Returns:
            Time embedding, shape (B, L, D)
        """
        pe = torch.zeros(*delta_tau.shape[:-1], self.embed_dim, device=delta_tau.device)
        pe[..., 0::2] = torch.sin(delta_tau * self.div_term)
        pe[..., 1::2] = torch.cos(delta_tau * self.div_term)
        return self.linear(pe)


class SLWTransformer(nn.Module):
    """Transformer encoder tailored for Toy-Story central brain."""

    def __init__(
        self,
        embed_dim: int = 4096,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embed = nn.Linear(embed_dim, embed_dim)

        # token-type (REAL=0, IMAG=1, maybe more) + Δτ scalar bucketed to 16 dims
        self.type_embed = nn.Embedding(4, embed_dim)
        self.time_embed = TimeEmbedding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # sinusoidal positional encoding (non-learnable)
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pos_buffer", pe, persistent=False)

    def forward(
        self,
        latent_seq: torch.Tensor,  # (B, L, D)
        token_type: torch.LongTensor,  # (B, L) 0=REAL,1=IMAG
        delta_tau: Optional[torch.Tensor] = None,  # (B, L, 1) time diff
        key_padding_mask: Optional[torch.BoolTensor] = None,  # (B, L)
    ) -> torch.Tensor:
        B, L, D = latent_seq.shape
        assert D == self.embed_dim

        x = self.token_embed(latent_seq)
        type_emb = self.type_embed(token_type)
        x = x + type_emb

        if delta_tau is not None:
            x = x + self.time_embed(delta_tau)

        # positional encoding
        x = x + self.pos_buffer[:L].unsqueeze(0)

        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return out 