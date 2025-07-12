from __future__ import annotations

import torch
import torch.nn as nn


class RewardTokenEmbedding(nn.Module):
    """Embed previous-step scalar reward as a discrete token.

    The scalar reward *r* is first clipped to a range ``[min_value, max_value]`` and then
    quantised into ``num_bins`` equally-spaced buckets. The bucket index is looked up in an
    ``nn.Embedding`` table to yield a fixed-dimensional vector that can be concatenated to
    the input token sequence of the Unified Latent-World Transformer (ULWT).
    """

    def __init__(
        self,
        embed_dim: int,
        num_bins: int = 15,
        min_value: float = -3.0,
        max_value: float = 3.0,
    ) -> None:
        """Args
        -----
        embed_dim: int
            Dimensionality of the reward embedding vector.
        num_bins: int
            Number of discrete bins to quantise the reward range into.
        min_value: float
            Minimum reward value expected (clipped otherwise).
        max_value: float
            Maximum reward value expected (clipped otherwise).
        """
        super().__init__()
        assert num_bins > 1, "num_bins must be at least 2."
        self.num_bins = num_bins
        self.min_value = min_value
        self.max_value = max_value
        self.embedding = nn.Embedding(num_bins, embed_dim)

    def _discretise(self, r: torch.Tensor) -> torch.Tensor:
        """Map reward tensor to integer bucket indices.

        Parameters
        ----------
        r : torch.Tensor
            Tensor of scalar rewards ``(batch,)`` or ``(batch, 1)``.
        Returns
        -------
        torch.Tensor
            Long tensor of bucket indices in ``[0, num_bins-1]`` with shape ``(batch,)``.
        """
        r = r.squeeze(-1)
        r_clamped = torch.clamp(r, self.min_value, self.max_value)
        # Scale to [0, num_bins - 1]
        scale = (self.num_bins - 1) / (self.max_value - self.min_value)
        indices = torch.round((r_clamped - self.min_value) * scale).long()
        return indices

    def forward(self, r: torch.Tensor) -> torch.Tensor:  # (B,) or (B,1) -> (B, E)
        """Return reward embedding vector.
        """
        idx = self._discretise(r)
        return self.embedding(idx) 