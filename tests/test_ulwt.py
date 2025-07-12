import torch
from models.transformers import UnifiedLatentWorldTransformer

def test_ulwt_forward():
    B, L, D = 2, 5, 512
    tokens = torch.randn(B, L, D)
    reward_prev = torch.zeros(B, 1)
    ulwt = UnifiedLatentWorldTransformer(embed_dim=D, n_layers=2, n_heads=8, max_len=L+2)
    out = ulwt(tokens, reward_prev)
    assert out["sequence_output"].shape == (B, L + 2, D)
    assert out["cls_hidden"].shape == (B, D)
    assert out["hfp_prob"].shape == (B, 1)
    assert (out["hfp_prob"] >= 0).all() and (out["hfp_prob"] <= 1).all() 