from __future__ import annotations
import torch
import torch.nn as nn
import hydra
from typing import Tuple
from PIL import Image

class PerceptionTower(nn.Module):
    """
    The Perception Tower serves as the digital eye of the GIA.
    It takes raw sensor data (i.e., screenshots) and encodes it into
    meaningful feature representations (perceptual primitives) of UI elements.
    
    This is a placeholder for a sophisticated model like UI-TARS.
    """
    def __init__(self, model_name: str | None = None, is_frozen: bool = False, latent_dim: int = 4096):
        super().__init__()
        self.model_name = model_name
        self.is_frozen = is_frozen

        # In a real implementation, this would load the UI-TARS vision encoder.
        # As a placeholder, we'll use a simple convolutional network.
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        
        # Placeholder for output size; this needs to be calculated based on input image size.
        # Assuming a 256x256 input for calculation.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)
            cnn_out_dim = self.vision_encoder(dummy_input).shape[1]

        # Output mu and logvar
        self.fc = nn.Linear(cnn_out_dim, latent_dim * 2)
        self.latent_dim = latent_dim

        if self.is_frozen:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()

    def forward(self, screenshot_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a batch of screenshots into a set of perceptual primitives.

        Args:
            screenshot_batch: A tensor of screenshots with shape (B, C, H, W).

        Returns:
            A tuple of (mean, log_variance) of the latent state distribution.
        """
        # This is a highly simplified placeholder. A real implementation would:
        # 1. Use the UI-TARS model to perform object detection/segmentation.
        # 2. Extract feature vectors for each detected UI element per image.
        # For now, we'll just get a single global feature vector for each image.

        # Ensure input is on same device/dtype as the vision encoder parameters
        target_device = next(self.vision_encoder.parameters()).device
        screenshot_batch = screenshot_batch.to(target_device, dtype=torch.float32)

        with torch.no_grad() if self.is_frozen else torch.enable_grad():
            cnn_features = self.vision_encoder(screenshot_batch)
            params = self.fc(cnn_features)
        
        mu, logvar = torch.chunk(params, 2, dim=-1)
        return mu, logvar 