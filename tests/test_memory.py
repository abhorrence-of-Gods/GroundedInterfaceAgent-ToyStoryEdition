import torch
from models.hierarchical_memory import HierarchicalMemory

def test_hierarchical_memory_search():
    mem = HierarchicalMemory(embed_dim=3, episodic_max=4, flush_size=2, semantic_db_path="/tmp/semantic_mem_test.pt")
    keys = torch.randn(8, 3)
    values = [{"h": k.clone()} for k in keys]
    mem.write(keys[:4], values[:4])  # within episodic max
    mem.write(keys[4:], values[4:])  # triggers flush
    q = torch.zeros(3)
    res = mem.search(q, topk=5)
    assert len(res) <= 5
    # ensure each result has 'h'
    for v, d in res:
        assert "h" in v 