from __future__ import annotations

import torch
import torch.nn as nn


class LEDController(nn.Module):
    """Maps emotion latent vector *e_t* to RGB color and brightness.

    The controller is a simple MLP followed by `tanh` to bound outputs in
    ``[-1, 1]`` which can be rescaled to physical LED PWM values.
    """

    def __init__(self, emotion_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(emotion_dim),
            nn.Linear(emotion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),  # R,G,B,brightness
            nn.Tanh(),
        )

    def forward(self, e_t: torch.Tensor) -> torch.Tensor:  # (B, D) -> (B, 4)
        """Return normalized LED control signal.

        Output order: ``(R, G, B, brightness)`` each in ``[-1, 1]``.
        """
        return self.net(e_t) 