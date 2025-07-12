import torch
import torch.nn as nn
import torch.nn.functional as F


class HumanFeedbackHead(nn.Module):
    """Predicts the probability that a human evaluator will give positive feedback.

    This head is intended to be attached to the CLS (or equivalent pooled) hidden
    representation *h_t* produced by the Unified Latent-World Transformer (ULWT).

    The output is a scalar in the range ``[0, 1]`` representing the *like* probability
    \hat{y}_t. The same module can be used in an ensemble (multiple instances with
    different random seeds) to obtain an epistemic uncertainty estimate via the
    output variance.
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int = 128,
    ) -> None:
        """Args
        -----
        hidden_dim: int
            Dimensionality of the transformer hidden state fed into the head.
        mlp_hidden_dim: int, optional
            Size of the intermediate MLP layer. Defaults to ``128``.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:  # (B, hidden_dim) -> (B, 1)
        """Return the like-probability.

        Parameters
        ----------
        h_t : torch.Tensor
            Hidden representation of shape ``(batch, hidden_dim)``.

        Returns
        -------
        torch.Tensor
            Like probability in range ``[0, 1]`` with shape ``(batch, 1)``.
        """
        logits = self.net(h_t)
        probs = torch.sigmoid(logits)
        return probs

    @staticmethod
    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary-cross-entropy loss helper.

        Both ``pred`` and ``target`` should be of shape ``(batch, 1)``. ``target`` is
        expected to contain ``0`` or ``1`` values. The mean BCE loss is returned.
        """
        return F.binary_cross_entropy(pred, target) 