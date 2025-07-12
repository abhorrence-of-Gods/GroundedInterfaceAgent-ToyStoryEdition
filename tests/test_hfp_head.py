import torch
from models.heads import HumanFeedbackHead

def test_hfp_head_bce():
    head = HumanFeedbackHead(hidden_dim=32)
    h = torch.randn(4, 32)
    pred = head(h)
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    loss = head.loss(pred, target)
    assert loss.item() >= 0.0 