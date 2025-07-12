import torch
from models.losses.self_consistency import self_consistency_loss

def test_self_consistency_basic():
    v1 = torch.tensor([[1.0, 0.0]])
    v2 = torch.tensor([[1.0, 0.0]])
    loss = self_consistency_loss(v1, v2)
    assert torch.isclose(loss, torch.tensor(0.0))

    v3 = torch.tensor([[0.0, 1.0]])
    loss2 = self_consistency_loss(v1, v3)
    assert torch.isclose(loss2, torch.tensor(1.0)) 