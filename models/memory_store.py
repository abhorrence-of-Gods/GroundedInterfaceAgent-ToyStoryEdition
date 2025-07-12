import os
import time
import threading
from typing import List, Dict, Any, Tuple

import torch

__all__ = ["MemoryStore"]


class MemoryStore:
    """A light-weight persistent vector memory with k-NN search.

    Each *record* is a Python ``dict`` that must contain at least a key ``"h"``
    (the high-dimensional latent tensor) and may include an optional ``"e"``
    field (emotion latent).  The user supplies an *embedding vector* that will
    be used as the ANN search key.  For small (<10M) record
    counts we simply keep everything in RAM and perform a brute-force L2 search
    which is fast enough (≤1 ms for 100 k keys on modern CPUs).

    If `db_path` is given the store will periodically flush itself to disk via
    ``torch.save`` so that the content survives crashes / restarts.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        db_path: str | None = None,
        flush_size: int = 1024,
    ) -> None:
        self.embed_dim = embed_dim
        self.db_path = db_path
        self.flush_size = flush_size

        # Internal buffers.  Keep everything on CPU to avoid GPU<>CPU hops during
        # search.  Use float32 for compatibility with faiss (if we later switch).
        self._keys: torch.Tensor = torch.empty(0, embed_dim, dtype=torch.float32)
        self._values: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        if self.db_path and os.path.exists(self.db_path):
            self._load()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def write(self, keys: torch.Tensor, values: List[Dict[str, Any]]) -> None:
        """Append *keys* (``B×D``) and their *values* to the store.

        Args:
            keys: Tensor of shape (B, ``embed_dim``) **on any device**.  Will be
                  detached, converted to ``float32`` and moved to CPU.
            values: ``len(values) == B`` list of Python dicts (arbitrary payload).
        """
        assert keys.dim() == 2 and keys.size(1) == self.embed_dim, (
            f"Expected keys of shape (B,{self.embed_dim}), got {tuple(keys.shape)}"
        )
        assert len(values) == keys.size(0), "Number of keys and values must match"

        keys_cpu = keys.detach().to(torch.float32).cpu()

        with self._lock:
            self._keys = torch.cat([self._keys, keys_cpu], dim=0)
            self._values.extend(values)
            if self.db_path and (len(self._values) % self.flush_size == 0):
                self._save()

    def search(self, query: torch.Tensor, topk: int = 16) -> List[Tuple[Dict[str, Any], float]]:
        """Return ``topk`` nearest records to *query*.

        Args:
            query: (D,) tensor on any device.
            topk:  number of neighbours to return.
        Returns:
            List of ``(value_dict, distance)`` tuples sorted **closest first**.
        """
        if self._keys.numel() == 0:
            return []

        query_cpu = query.detach().to(torch.float32).cpu()
        # Unsqueeze to broadcast: (N,D) - (1,D) -> (N,D)
        dists = torch.norm(self._keys - query_cpu, dim=-1)  # L2 distance
        k = min(topk, self._keys.size(0))
        dist_vals, idxs = torch.topk(dists, k=k, largest=False)
        return [(self._values[int(i)], float(dist_vals[j])) for j, i in enumerate(idxs)]

    def consolidate(self, max_records: int = 1_000_000) -> None:
        """Keep only the most recent *max_records* entries to bound memory.
        """
        with self._lock:
            if len(self._values) <= max_records:
                return
            keep_from = len(self._values) - max_records
            self._keys = self._keys[keep_from:]
            self._values = self._values[keep_from:]
            if self.db_path:
                self._save()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _save(self) -> None:
        tmp_path = self.db_path + ".tmp"
        torch.save({"keys": self._keys, "values": self._values}, tmp_path)
        os.replace(tmp_path, self.db_path)

    def _load(self) -> None:
        data = torch.load(self.db_path, map_location="cpu")
        self._keys = data.get("keys", torch.empty(0, self.embed_dim))
        self._values = data.get("values", [])


# ---------------------------------------------------------------------------
# Simple CLI test (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    store = MemoryStore(embed_dim=3, db_path="memory_test.pt", flush_size=5)

    for i in range(20):
        k = torch.randn(3)
        v = {"h": k.clone(), "ts": time.time()}
        store.write(k.unsqueeze(0), [v])

    q = torch.tensor([0.0, 0.0, 0.0])
    results = store.search(q, topk=3)
    for v, d in results:
        print("dist", d, "val_h_norm", v["h"].norm().item()) 