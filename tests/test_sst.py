import torch
from models.sst_transformer import SlowStateTransformer

def test_sst_forward():
    model = SlowStateTransformer(embed_dim=64, n_layers=2, n_heads=4, max_len=32)
    seq = torch.randn(2, 10, 64)
    out = model(seq)
    assert out["sequence_output"].shape == (2, 11, 64)  # CLS appended
    assert out["slow_state"].shape == (2, 64) 