from __future__ import annotations

import torch
import torch.nn as nn


class VoiceController(nn.Module):
    """Maps emotion latent *e_t* to prosody parameters and optional text prefix.

    For a full TTS pipeline we would feed the prefix and prosody into an external
    synthesiser (e.g. VITS). This controller focuses on generating:

    1. ``prosody``: 3-dim tensor (pitch_shift, speaking_rate, volume_gain) in
       range ``[-1, 1]``.
    2. ``tone_id``: integer token (0=neutral,1=joy,2=sad,3=angry,4=calm).

    The mapping remains differentiable so that gradients can flow from a speech
    loss (e.g. style classifier) if desired.
    """

    def __init__(self, emotion_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.prosody_head = nn.Sequential(
            nn.LayerNorm(emotion_dim),
            nn.Linear(emotion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),  # bound prosody params
        )
        # Tone logits head
        self.tone_head = nn.Sequential(
            nn.LayerNorm(emotion_dim),
            nn.Linear(emotion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),  # 5 tone classes
        )

    def forward(self, e_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # (B,D)->((B,3),(B,))
        prosody = self.prosody_head(e_t)
        tone_logits = self.tone_head(e_t)
        tone_id = torch.argmax(tone_logits, dim=-1)
        return prosody, tone_id 