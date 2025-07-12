from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (unchanged across calls)."""

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
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


class SlowStateTransformer(nn.Module):
    """Slow-State Transformer (SST) for long-range latent context.

    The SST consumes a sequence of latent state vectors (e.g. Dreamer z_t or
    fused h_t) over a long temporal horizon (hundreds–thousands of steps) and
    outputs an aggregated "slow" state vector s_t that can serve as a
    persistent context embedding for policy or memory modules.
    """

    def __init__(
        self,
        embed_dim: int = 4096,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 512,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        # Optional learnable CLS token for global representation
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)

        # Input projection (in case upstream latent dim != embed_dim). If the
        # dimensions match, the projection acts as identity.
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=max_len + 1)

        # Final layer-norm for the aggregated slow state
        self.final_ln = nn.LayerNorm(embed_dim)

    def forward(
        self,
        latent_seq: torch.Tensor,  # (B, L, D)
        key_padding_mask: Optional[torch.BoolTensor] = None,  # (B, L)
    ) -> dict[str, torch.Tensor]:
        B, L, D = latent_seq.size()
        assert D == self.embed_dim, (
            f"Input embedding dim {D} does not match SST embed_dim {self.embed_dim}."
        )

        x = self.input_proj(latent_seq)

        # Append CLS token if requested
        if self.use_cls_token:
            cls = self.cls_token.repeat(B, 1, 1)
            x = torch.cat([x, cls], dim=1)  # (B, L+1, D)
            if key_padding_mask is not None:
                pad_mask = torch.zeros(B, 1, dtype=key_padding_mask.dtype, device=key_padding_mask.device)
                key_padding_mask = torch.cat([key_padding_mask, pad_mask], dim=1)

        x = self.pos_enc(x)
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Aggregate
        if self.use_cls_token:
            s_t = out[:, -1]  # (B, D) CLS hidden
        else:
            # Mean pool over temporal dimension
            if key_padding_mask is not None:
                # Masked mean pooling
                valid = (~key_padding_mask).float().unsqueeze(-1)
                s_t = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1e-6)
            else:
                s_t = out.mean(dim=1)

        s_t = self.final_ln(s_t)
        return {"sequence_output": out, "slow_state": s_t} 