import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, TransformedDistribution, AffineTransform
from torch.distributions.independent import Independent

class ActionTower(nn.Module):
    """
    The Action Tower has three roles in the Dreamer architecture:
    1.  Actor: Proposes latent actions in the "dream".
    2.  ActionEncoder: Encodes real actions into the latent space for pre-training.
    3.  ActionDecoder: Decodes a chosen latent action back into a concrete motor command.
    """
    def __init__(
        self,
        latent_state_dim: int,
        latent_action_dim: int,
        concrete_action_dim: int,
        warp_output_dim: int,
        action_limits: list[list[float]] | None = None,
    ):
        super().__init__()
        self.latent_state_dim = latent_state_dim
        self.latent_action_dim = latent_action_dim
        self.concrete_action_dim = concrete_action_dim
        hidden_dim = 512

        # 1. Actor Network: h_t -> a_t
        # Takes a state and decides on a latent action.
        self.actor_network = nn.Sequential(
            nn.Linear(latent_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_action_dim * 2) # mean and std dev
        )

        # 2. Action Encoder: concrete_action -> a_t
        self.action_encoder = nn.Sequential(
            nn.Linear(concrete_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_action_dim)
        )
        
        # 3. Action Decoder: a_t -> concrete_action
        self.action_decoder = nn.Sequential(
            nn.Linear(latent_action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, concrete_action_dim),
            nn.Tanh(),  # Squash output to [-1, 1] range
        )

        # 4. Warp Decoder: a_t -> warp params (speed, precision, etc.)
        self.warp_decoder = nn.Sequential(
            nn.Linear(latent_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, warp_output_dim),
            nn.Sigmoid()
        )

        # 5. Action space rescaling
        self.action_limits = action_limits
        if self.action_limits:
            limits = torch.tensor(self.action_limits, dtype=torch.float32)
            self.register_buffer("action_min", limits[:, 0])
            self.register_buffer("action_max", limits[:, 1])
        else:
            self.action_min = None
            self.action_max = None

    def get_action_dist(self, latent_state: torch.Tensor) -> TransformedDistribution:
        """
        Returns a squashed Gaussian distribution, rescaled to action_limits.
        """
        mu, log_std = self.actor_network(latent_state).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        std = torch.exp(log_std)
        
        base_dist = Independent(Normal(loc=mu, scale=std), 1)
        
        # Squash to [-1, 1] then rescale to [min, max]
        transforms = [TanhTransform()]
        if self.action_limits is not None:
            loc = (self.action_max + self.action_min) / 2
            scale = (self.action_max - self.action_min) / 2
            transforms.append(AffineTransform(loc=loc, scale=scale))
            
        return TransformedDistribution(base_dist, transforms)

    def encode_action(self, concrete_action: torch.Tensor) -> torch.Tensor:
        """Encodes a real action into the latent space."""
        return self.action_encoder(concrete_action)
    
    def decode_action(self, latent_action: torch.Tensor) -> torch.Tensor:
        """Decodes a latent action into a real motor command."""
        return self.action_decoder(latent_action)

    def decode_warp(self, latent_action: torch.Tensor) -> torch.Tensor:
        """Decodes latent action into spacetime warp parameters."""
        return self.warp_decoder(latent_action)

    # -------------------------------------------------------------
    # Default forward: Used by GiaAgent's generative pathways
    # -------------------------------------------------------------
    def forward(self, embedding: torch.Tensor, deterministic: bool = True) -> torch.Tensor:  # type: ignore
        """Maps an input embedding (typically a latent *state* representation) to a concrete action.

        This unifies usage across:
        1. Pre-training and generative loss paths where we want a deterministic
           prediction of the action given a fused/context embedding.
        2. Inference where we may still call the tower directly for a quick
           action proposal outside the Dreamer rollout.

        Args:
            embedding: Either a latent *state* (dim = latent_state_dim) or an
                       already latent *action* (dim = latent_action_dim).
            deterministic: If True, uses the mean of the latent action
                       distribution. If False, samples using reparameterization.

        Returns:
            A concrete action tensor (Batch, concrete_action_dim).
        """
        if embedding.shape[-1] == self.latent_action_dim:
            # Already a latent action – directly decode.
            latent_action = embedding
        elif embedding.shape[-1] == self.latent_state_dim:
            # Treat as state embedding – derive an action distribution.
            dist = self.get_action_dist(embedding)
            if deterministic:
                latent_action = dist.mean
            else:
                latent_action = dist.rsample()
        else:
            raise ValueError(
                f"Unsupported embedding dimension {embedding.shape[-1]} for ActionTower.forward"
            )

        # Decode to concrete actions and scale/clip
        concrete_action_raw = self.action_decoder(latent_action)
        
        if self.action_min is not None:
            # Scale from [-1, 1] to [min, max] range
            action_range = (self.action_max - self.action_min) / 2.0
            action_center = (self.action_max + self.action_min) / 2.0
            concrete_action = concrete_action_raw * action_range + action_center
        else:
            concrete_action = concrete_action_raw

        return concrete_action 