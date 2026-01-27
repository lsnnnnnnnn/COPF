from typing import Dict, Set, Tuple, Any, Optional
from collections import defaultdict, deque
import math
import numpy as np

class LocalGraph:
    def __init__(self, change_window: int = 100):
        self.neigh: Dict[int, Set[int]] = defaultdict(set)
        self.deg: Dict[int, int] = defaultdict(int)
        
        # For neighborhood change rate tracking
        self.change_window = change_window
        self.edge_history: deque = deque(maxlen=change_window)
        self.node_last_change: Dict[int, int] = {}
        self.current_t = 0
        
        # Count-Min sketch for high-degree nodes (Sec 14.2)
        self.use_sketch_above_deg = 1000
        self.cm_sketch: Optional[Any] = None  # Would use actual Count-Min implementation

    def add_edge(self, u: int, v: int, t: Optional[int] = None):
        """Add edge with optional timestamp for change tracking"""
        if v not in self.neigh[u]:
            self.neigh[u].add(v)
            self.deg[u] += 1
            if t is not None:
                self.node_last_change[u] = t
                
        if u not in self.neigh[v]:
            self.neigh[v].add(u)
            self.deg[v] += 1
            if t is not None:
                self.node_last_change[v] = t
                
        # Track edge in history
        if t is not None:
            self.edge_history.append((u, v, t))
            self.current_t = max(self.current_t, t)

    def has_edge(self, u: int, v: int) -> bool:
        """Check if edge exists in G<=t"""
        return (v in self.neigh[u])

    def neighborhood_change_rate(self, u: int, v: int) -> float:
        """
        Compute neighborhood change rate for (u,v) as per paper Section 5.
        Used for locality weights w_t(i,j) = exp(-γ * rate)
        """
        if len(self.edge_history) < 2:
            return 0.0
            
        # Count changes in 1-hop neighborhoods of u and v
        changes_u = 0
        changes_v = 0
        
        for edge in self.edge_history:
            src, dst, _ = edge
            if src == u or dst == u:
                changes_u += 1
            if src == v or dst == v:
                changes_v += 1
                
        # Normalized change rate
        rate = (changes_u + changes_v) / (2.0 * len(self.edge_history))
        return float(rate)

    def common_neighbors(self, u: int, v: int) -> int:
        """Common neighbors |N(u) ∩ N(v)|"""
        return len(self.neigh[u] & self.neigh[v])

    def adamic_adar(self, u: int, v: int) -> float:
        """Adamic-Adar: Σ_{w∈N(u)∩N(v)} 1/log(deg(w))"""
        inter = self.neigh[u] & self.neigh[v]
        s = 0.0
        for w in inter:
            dw = max(1, self.deg[w])
            s += 1.0 / math.log(dw + 1.0)  # add +1 to avoid log(1)=0
        return s

    def resource_allocation(self, u: int, v: int) -> float:
        """Resource Allocation: Σ_{w∈N(u)∩N(v)} 1/deg(w)"""
        inter = self.neigh[u] & self.neigh[v]
        return sum(1.0 / max(1, self.deg[w]) for w in inter)

    def jaccard(self, u: int, v: int) -> float:
        """Jaccard coefficient: |N(u)∩N(v)| / |N(u)∪N(v)|"""
        Au, Av = self.neigh[u], self.neigh[v]
        if not Au and not Av:
            return 0.0
        inter = len(Au & Av)
        union = len(Au | Av)
        return float(inter) / float(union) if union > 0 else 0.0

    def preferential_attachment(self, u: int, v: int) -> float:
        """Preferential attachment: deg(u) * deg(v)"""
        return float(self.deg[u] * self.deg[v])

    def katz_centrality_approx(self, u: int, v: int, beta: float = 0.001) -> float:
        """
        Approximate Katz index using local paths (2-hop only for efficiency)
        Katz = β * |paths_1| + β² * |paths_2|
        """
        paths_1 = 1 if self.has_edge(u, v) else 0
        paths_2 = self.common_neighbors(u, v)
        return beta * paths_1 + beta * beta * paths_2

    def features(self, u: int, v: int, t: Optional[int] = None) -> Dict[str, Any]:
        """
        Comprehensive feature extraction for link (u,v) at time t
        All features use ONLY G<=t information
        """
        feat = {
            "deg_u": self.deg[u],
            "deg_v": self.deg[v],
            "cn": self.common_neighbors(u, v),
            "aa": self.adamic_adar(u, v),
            "ra": self.resource_allocation(u, v),
            "jaccard": self.jaccard(u, v),
            "pa": self.preferential_attachment(u, v),
            "katz": self.katz_centrality_approx(u, v),
        }
        
        # Add temporal features if timestamp provided
        if t is not None:
            feat["t"] = t
            feat["time_since_u_change"] = t - self.node_last_change.get(u, 0)
            feat["time_since_v_change"] = t - self.node_last_change.get(v, 0)
            
        # Locality weight for GA-DR (Paper Section 5)
        feat["w_local"] = math.exp(-0.1 * self.neighborhood_change_rate(u, v))
        
        return feat

    def get_motif_counts(self, u: int, v: int) -> Dict[str, int]:
        """
        Count local motifs/graphlets for structural roles (Paper Sec 6: ψ_m(G^local))
        For efficiency, only count 3-node motifs
        """
        motifs = {
            "triangle": 0,      # closed triangles through (u,v)
            "wedge": 0,         # open wedges centered at u or v
            "3path": 0,         # 3-paths through (u,v)
        }
        
        # Triangles: common neighbors
        motifs["triangle"] = self.common_neighbors(u, v)
        
        # Wedges centered at u (u-w-x where w≠v, x≠v, w≠x)
        for w in self.neigh[u]:
            if w != v:
                motifs["wedge"] += len(self.neigh[w] - {u, v})
                
        # 3-paths: paths u-w-x where w is neighbor of u but not v
        for w in (self.neigh[u] - self.neigh[v]):
            motifs["3path"] += len(self.neigh[w] - {u})
            
        return motifs

    def clear_old_edges(self, cutoff_t: int):
        """
        Optional: Remove edges older than cutoff_t for memory efficiency
        Only for streaming scenarios with bounded memory
        """
        # This would require tracking edge timestamps more carefully
        pass
