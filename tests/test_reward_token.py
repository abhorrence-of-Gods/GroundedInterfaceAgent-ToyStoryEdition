import torch
from models.embeddings import RewardTokenEmbedding
 
def test_reward_token_embedding():
    emb = RewardTokenEmbedding(embed_dim=16, num_bins=5, min_value=-1.0, max_value=1.0)
    r = torch.tensor([[-1.5], [-0.5], [0.0], [0.5], [2.0]])  # clamp extremes
    e = emb(r)
    assert e.shape == (5, 16) 