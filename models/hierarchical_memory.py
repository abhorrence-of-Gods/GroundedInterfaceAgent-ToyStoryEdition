from __future__ import annotations

import torch
from typing import List, Dict, Any, Tuple

from .memory_store import MemoryStore


class HierarchicalMemory:
    """Two-level memory: fast episodic buffer + large persistent semantic store.

    Design:
    1. *Episodic level* keeps the most recent `episodic_max` records purely in RAM
       for rapid look-up.
    2. When the episodic level exceeds this threshold it *flushes* the **oldest**
       half into the semantic store (on-disk) to free up space.
    3. During retrieval we query **both** levels, concatenate results, and return
       the global top-k by distance.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        episodic_max: int = 10_000,
        semantic_db_path: str = "semantic_memory.pt",
        flush_size: int = 1024,
    ) -> None:
        self.embed_dim = embed_dim
        self.episodic_max = episodic_max
        self.episodic = MemoryStore(embed_dim=embed_dim, db_path=None)  # RAM only
        self.semantic = MemoryStore(embed_dim=embed_dim, db_path=semantic_db_path, flush_size=flush_size)

    # ---------------------------------------------------------------------
    # Public API mirrors MemoryStore
    # ---------------------------------------------------------------------
    def write(self, keys: torch.Tensor, values: List[Dict[str, Any]]) -> None:
        """Write to episodic level; trigger flush if needed."""
        self.episodic.write(keys, values)

        # If episodic memory too large, move the oldest half to semantic memory
        if len(self.episodic._values) > self.episodic_max:
            flush_n = len(self.episodic._values) // 2
            with self.episodic._lock:
                move_keys = self.episodic._keys[:flush_n]
                move_vals = self.episodic._values[:flush_n]
                # Trim episodic buffers
                self.episodic._keys = self.episodic._keys[flush_n:]
                self.episodic._values = self.episodic._values[flush_n:]
            # Write to semantic level (will flush to disk as per its policy)
            self.semantic.write(move_keys, move_vals)

    def search(self, query: torch.Tensor, topk: int = 16) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieve nearest neighbours across both memory levels."""
        results_epi = self.episodic.search(query, topk=topk)
        results_sem = self.semantic.search(query, topk=topk)
        # Merge while preserving distance order
        merged = results_epi + results_sem
        merged.sort(key=lambda tpl: tpl[1])  # ascending by distance
        return merged[:topk]

    def consolidate(self):
        """Manually consolidate semantic store (pass-through)."""
        self.semantic.consolidate() 