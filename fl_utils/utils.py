from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class NeighborSampler:

    num_nodes: int
    seed: Optional[int] = 0
    sample_neighbor_strategy: str = "recent"  # kept for compatibility

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed if self.seed is not None else 0)
        self._adj: List[List[Tuple[float, int, int]]] = [[] for _ in range(int(self.num_nodes) + 1)]

    def reset_random_state(self):
        self.rng = np.random.default_rng(self.seed if self.seed is not None else 0)

    def insert_edge(self, src: int, dst: int, t: float, edge_id: int) -> None:
        src = int(src); dst = int(dst); edge_id = int(edge_id)
        tt = float(t)

        # undirected storage for neighbor lookup
        self._adj[src].append((tt, dst, edge_id))
        self._adj[dst].append((tt, src, edge_id))

    def _query_one(self, node: int, t: float, k: int):
        hist = self._adj[int(node)]
        if not hist:
            return [], [], []

        # ensure hist sorted by time (amortized cheap; lists grow)
        hist.sort(key=lambda x: x[0])

        times = [x[0] for x in hist]
        cut = bisect_left(times, float(t))  # strictly < t
        past = hist[:cut]
        if not past:
            return [], [], []

        # take most recent k
        take = past[-k:]
        nbs = [x[1] for x in take]
        eids = [x[2] for x in take]
        ts = [x[0] for x in take]
        return nbs, eids, ts

    def get_historical_neighbors(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        node_interact_times = np.asarray(node_interact_times, dtype=np.float32).reshape(-1)
        B = node_ids.shape[0]
        k = int(num_neighbors)

        neigh_nodes = np.zeros((B, k), dtype=np.int64)
        neigh_eids = np.zeros((B, k), dtype=np.int64)
        neigh_ts = np.zeros((B, k), dtype=np.float32)

        for i in range(B):
            nbs, eids, ts = self._query_one(int(node_ids[i]), float(node_interact_times[i]), k)
            if len(nbs) == 0:
                continue
            m = len(nbs)
            # right-align recent neighbors (so padding at left)
            neigh_nodes[i, -m:] = np.asarray(nbs, dtype=np.int64)
            neigh_eids[i, -m:] = np.asarray(eids, dtype=np.int64)
            neigh_ts[i, -m:] = np.asarray(ts, dtype=np.float32)

        return neigh_nodes, neigh_eids, neigh_ts

    # compatibility alias (your old test used this name)
    def get_temporal_neighbor(self, node_ids, node_interact_times, num_neighbors):
        return self.get_historical_neighbors(node_ids=np.asarray(node_ids),
                                            node_interact_times=np.asarray(node_interact_times),
                                            num_neighbors=int(num_neighbors))