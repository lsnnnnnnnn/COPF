from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd

import importlib.util
import inspect
import types

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_TGB_ROOT = os.environ.get("TGB_BASELINES", os.path.join(REPO_ROOT, "tgb_baselines"))


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _load_py_module_from_path(mod_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {mod_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_class_by_keyword(mod, keyword: str):
    if hasattr(mod, keyword) and inspect.isclass(getattr(mod, keyword)):
        return getattr(mod, keyword)
    for name, obj in mod.__dict__.items():
        if inspect.isclass(obj) and keyword.lower() in name.lower():
            return obj
    raise ImportError(f"Cannot find class with keyword='{keyword}' in module {mod.__name__}")


def _purge_modules(prefix: str) -> None:
    """Remove already-imported modules with a given prefix from sys.modules."""
    keys = [k for k in list(sys.modules.keys()) if k == prefix or k.startswith(prefix + ".")]
    for k in keys:
        del sys.modules[k]


def _push_sys_path_after_cwd(path0: str) -> None:
    """
    Put tgb_root AFTER '' (cwd) so fairlink's local packages win by default.
    But for GraphMixer we will patch fl_utils/fl_models anyway.
    """
    if path0 in sys.path:
        sys.path.remove(path0)
    insert_pos = 1 if (len(sys.path) > 0 and sys.path[0] == "") else 0
    sys.path.insert(insert_pos, path0)


# ---------------------------------------------------------------------
# Compat modules for GraphMixer (fix fl_utils.utils.NeighborSampler and fl_models.modules.TimeEncoder)
# ---------------------------------------------------------------------
class CompatTimeEncoder(nn.Module):
    """
    Minimal compatible TimeEncoder:
      - accepts parameter_requires_grad (GraphMixer passes it)
      - forward(timestamps=...) returns cos(W*t + b) with shape (..., time_dim)
    """
    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        super().__init__()
        self.time_dim = int(time_dim)
        self.lin = nn.Linear(1, self.time_dim, bias=True)
        if not parameter_requires_grad:
            for p in self.lin.parameters():
                p.requires_grad = False

    def forward(self, timestamps: torch.Tensor, **kwargs) -> torch.Tensor:
        if timestamps.dim() == 0:
            timestamps = timestamps.view(1)
        if timestamps.dim() == 1:
            x = timestamps.unsqueeze(-1)
        else:
            x = timestamps.unsqueeze(-1) if timestamps.size(-1) != 1 else timestamps
        out = self.lin(x)
        return torch.cos(out)


class CompatNeighborSampler:
    """
    A robust neighbor sampler that implements:
      - get_historical_neighbors(node_ids, node_interact_times, num_neighbors) -> (nbr_nodes, nbr_eidx, nbr_ts)
    Using a per-node time-sorted adjacency.

    IMPORTANT: node_id=0 is reserved for padding. So this sampler assumes you SHIFTED real node ids by +1.
    Edge ids also start from 1; 0 is padding.

    FIX: must be initialized with a GLOBAL num_nodes (from meta), because negatives may include isolated nodes
    not present in edges; otherwise IndexError when nid > max observed id in edges.
    """
    def __init__(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        ts: np.ndarray,
        eidx: Optional[np.ndarray] = None,
        seed: Optional[int] = 0,
        sample_neighbor_strategy: str = "recent",
        num_nodes: Optional[int] = None,
    ):
        self.seed = seed if seed is not None else 0
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self._rng = np.random.default_rng(self.seed)

        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        ts = np.asarray(ts, dtype=np.float64)
        if eidx is None:
            eidx = np.arange(len(src), dtype=np.int64)
        eidx = np.asarray(eidx, dtype=np.int64)

        # Determine total nodes (include padding 0)
        if num_nodes is None:
            max_nid = int(max(src.max(initial=0), dst.max(initial=0)))
            self.num_nodes = max_nid + 1
        else:
            self.num_nodes = int(num_nodes)

        self.adj: List[List[Tuple[float, int, int]]] = [[] for _ in range(self.num_nodes)]

        # Add edges (undirected temporal adjacency)
        for s, d, t, ei in zip(src, dst, ts, eidx):
            s = int(s); d = int(d)
            if 0 <= s < self.num_nodes:
                self.adj[s].append((float(t), d, int(ei)))
            if 0 <= d < self.num_nodes:
                self.adj[d].append((float(t), s, int(ei)))

        # sort by time
        for nid in range(self.num_nodes):
            self.adj[nid].sort(key=lambda x: x[0])

    def reset_random_state(self):
        self._rng = np.random.default_rng(self.seed)

    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        cut_times = np.asarray(node_interact_times, dtype=np.float64).reshape(-1)
        B = len(node_ids)
        K = int(num_neighbors)

        nbr_nodes = np.zeros((B, K), dtype=np.int64)
        nbr_eidx = np.zeros((B, K), dtype=np.int64)
        nbr_ts = np.zeros((B, K), dtype=np.float64)

        for i, (nid, ct) in enumerate(zip(node_ids, cut_times)):
            nid = int(nid)

            # SAFE: isolated or out-of-range nodes just have no neighbors
            if nid <= 0 or nid >= self.num_nodes:
                continue

            hist = self.adj[nid]
            if not hist:
                continue

            picked: List[Tuple[float, int, int]] = []
            for t, nb, ei in reversed(hist):
                if t < ct:
                    picked.append((t, nb, ei))
                    if len(picked) >= K:
                        break
            picked.reverse()

            for j, (t, nb, ei) in enumerate(picked[-K:]):
                nbr_ts[i, j] = float(t)
                # clamp neighbor id into range; out-of-range -> 0 padding
                nb = int(nb)
                nbr_nodes[i, j] = nb if (0 <= nb < self.num_nodes) else 0
                nbr_eidx[i, j] = int(ei)

        return nbr_nodes, nbr_eidx, nbr_ts

    # Backward-compat alias
    def get_temporal_neighbor(self, node_ids, node_interact_times, num_neighbors):
        return self.get_historical_neighbors(np.array(node_ids), np.array(node_interact_times), int(num_neighbors))


def _install_graphmixer_compat_modules() -> None:
    """
    Ensure that when GraphMixer.py does:
      from fl_models.modules import TimeEncoder
      from fl_utils.utils import NeighborSampler
    it sees compatible implementations.
    """
    # patch fl_models.modules.TimeEncoder
    try:
        import fl_models.modules as fm  # type: ignore
        fm.TimeEncoder = CompatTimeEncoder  # type: ignore[attr-defined]
    except Exception:
        fl_models_mod = sys.modules.get("fl_models")
        if fl_models_mod is None:
            fl_models_mod = types.ModuleType("fl_models")
            sys.modules["fl_models"] = fl_models_mod
        modules_mod = types.ModuleType("fl_models.modules")
        modules_mod.TimeEncoder = CompatTimeEncoder
        sys.modules["fl_models.modules"] = modules_mod
        setattr(fl_models_mod, "modules", modules_mod)

    # patch fl_utils.utils.NeighborSampler
    try:
        import fl_utils.utils as fu  # type: ignore
        fu.NeighborSampler = CompatNeighborSampler  # type: ignore[attr-defined]
    except Exception:
        fl_utils_mod = sys.modules.get("fl_utils")
        if fl_utils_mod is None:
            fl_utils_mod = types.ModuleType("fl_utils")
            sys.modules["fl_utils"] = fl_utils_mod
        utils_mod = types.ModuleType("fl_utils.utils")
        utils_mod.NeighborSampler = CompatNeighborSampler
        sys.modules["fl_utils.utils"] = utils_mod
        setattr(fl_utils_mod, "utils", utils_mod)


# ---------------------------------------------------------------------
# TGB EdgeBank Online Adapter
# ---------------------------------------------------------------------
class TGBEdgeBankOnlineAdapter:
    def __init__(self, tgb_root: str):
        self.tgb_root = tgb_root
        self._seen = set()

        edgebank_py = os.path.join(tgb_root, "models", "EdgeBank.py")
        if not os.path.exists(edgebank_py):
            raise FileNotFoundError(f"EdgeBank.py not found at {edgebank_py}")

        _push_sys_path_after_cwd(tgb_root)
        _purge_modules("models")

        try:
            mod = _load_py_module_from_path("tgb_edgebank_mod", edgebank_py)
            EdgeBankCls = _find_class_by_keyword(mod, "EdgeBank")
            try:
                self.model = EdgeBankCls()
            except TypeError:
                self.model = EdgeBankCls(None)
            self._has_official = True
        except Exception as e:
            print(f"[WARN] Failed to load official TGB EdgeBank, fallback to seen-edge memory. Error: {e}")
            self.model = None
            self._has_official = False

    def score(self, u: int, v: int, t: int) -> float:
        u = int(u); v = int(v); t = int(t)
        if self._has_official and self.model is not None:
            for fn_name in ["predict", "predict_proba", "predict_link_prob", "compute_edge_prob", "forward"]:
                if hasattr(self.model, fn_name):
                    fn = getattr(self.model, fn_name)
                    try:
                        out = fn(np.array([u]), np.array([v]), np.array([t]))
                        return float(np.asarray(out).reshape(-1)[0])
                    except Exception:
                        pass
        return 1.0 if (u, v) in self._seen else 0.0

    def update(self, u: int, v: int, t: int) -> None:
        u = int(u); v = int(v); t = int(t)
        self._seen.add((u, v))
        if self._has_official and self.model is not None:
            for fn_name in ["update", "insert", "add_edge", "add_edges", "update_memory"]:
                if hasattr(self.model, fn_name):
                    fn = getattr(self.model, fn_name)
                    try:
                        fn(u, v, t)
                        return
                    except Exception:
                        try:
                            fn(np.array([u]), np.array([v]), np.array([t]))
                            return
                        except Exception:
                            pass

    def diagnostics(self) -> Dict[str, Any]:
        return {"model": "edgebank"}


# ---------------------------------------------------------------------
# GraphMixer Config + Wrapper
# ---------------------------------------------------------------------
@dataclass
class GraphMixerConfig:
    num_nodes: int
    lr: float
    weight_decay: float
    device: str
    seed: int
    tgb_root: str

    gm_num_tokens: int = 20
    gm_time_feat_dim: int = 128
    gm_num_layers: int = 2
    gm_token_dim_expansion_factor: float = 0.5
    gm_channel_dim_expansion_factor: float = 4.0
    gm_dropout: float = 0.1

    gm_num_neighbors: int = 20
    gm_time_gap: int = 2000


class GraphMixerWrapper:
    def __init__(self, cfg: GraphMixerConfig,
                 node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 neighbor_sampler: Any):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        _install_graphmixer_compat_modules()

        _push_sys_path_after_cwd(cfg.tgb_root)
        _purge_modules("models")

        gm_py = os.path.join(cfg.tgb_root, "models", "GraphMixer.py")
        if not os.path.exists(gm_py):
            raise ImportError(f"GraphMixer.py not found at: {gm_py}")

        try:
            from models.GraphMixer import GraphMixer  # type: ignore
            self._GraphMixer = GraphMixer
        except Exception as e:
            raise ImportError(f"Failed to import GraphMixer from {gm_py}. Error: {e}")

        self.model = self._GraphMixer(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=int(cfg.gm_time_feat_dim),
            num_tokens=int(cfg.gm_num_tokens),
            num_layers=int(cfg.gm_num_layers),
            token_dim_expansion_factor=float(cfg.gm_token_dim_expansion_factor),
            channel_dim_expansion_factor=float(cfg.gm_channel_dim_expansion_factor),
            dropout=float(cfg.gm_dropout),
            device=str(self.device),
        ).to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.last_loss: float = float("nan")

    def _edge_logits(self, u: np.ndarray, v: np.ndarray, t: np.ndarray) -> torch.Tensor:
        src_emb, dst_emb = self.model.compute_src_dst_node_temporal_embeddings(
            src_node_ids=u,
            dst_node_ids=v,
            node_interact_times=t,
            num_neighbors=int(self.cfg.gm_num_neighbors),
            time_gap=int(self.cfg.gm_time_gap),
        )
        logits = torch.sum(src_emb * dst_emb, dim=1)
        return logits

    @torch.no_grad()
    def score(self, u: int, v_list: List[int], t: int, src_group: int = 0) -> np.ndarray:
        self.model.eval()
        u_np = np.asarray([int(u)] * len(v_list), dtype=np.int64)
        v_np = np.asarray([int(v) for v in v_list], dtype=np.int64)
        t_np = np.asarray([float(t)] * len(v_list), dtype=np.float64)

        logits = self._edge_logits(u_np, v_np, t_np)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        return probs

    def update(self, u: int, pos_v: int, neg_vs: List[int], t: int, src_group: int = 0) -> None:
        self.model.train()

        v_list = [int(pos_v)] + [int(x) for x in neg_vs]
        y = torch.tensor([1.0] + [0.0] * len(neg_vs), device=self.device, dtype=torch.float32)

        u_np = np.asarray([int(u)] * len(v_list), dtype=np.int64)
        v_np = np.asarray(v_list, dtype=np.int64)
        t_np = np.asarray([float(t)] * len(v_list), dtype=np.float64)

        logits = self._edge_logits(u_np, v_np, t_np)
        loss = self.bce_logits(logits.view(-1), y)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        self.last_loss = float(loss.detach().cpu().item())

    def diagnostics(self) -> Dict[str, Any]:
        return {"model": "graphmixer", "last_loss": self.last_loss}


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
def _load_synth(data_dir: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict]:
    edges_path = os.path.join(data_dir, "edges.csv")
    nodes_path = os.path.join(data_dir, "nodes.csv")
    meta_path = os.path.join(data_dir, "meta.json")

    edges = pd.read_csv(edges_path)
    nodes = pd.read_csv(nodes_path) if os.path.exists(nodes_path) else None
    meta = json.load(open(meta_path, "r")) if os.path.exists(meta_path) else {}

    if "src" not in edges.columns or "dst" not in edges.columns:
        raise ValueError(f"edges.csv must contain columns src,dst,t. Found: {list(edges.columns)}")
    if "t" not in edges.columns:
        if "time" in edges.columns:
            edges = edges.rename(columns={"time": "t"})
        else:
            edges["t"] = np.arange(len(edges), dtype=np.int64)

    return edges, nodes, meta


def _load_tgb_csv(edgelist_csv: str) -> pd.DataFrame:
    edges = pd.read_csv(edgelist_csv)
    cols = set(edges.columns)

    # --------------------------------------------------
    # Case 1: minimal edgelist: src, dst, t
    # --------------------------------------------------
    if {"src", "dst"}.issubset(cols):
        if "t" not in cols:
            if "ts" in cols:
                edges = edges.rename(columns={"ts": "t"})
            elif "time" in cols:
                edges = edges.rename(columns={"time": "t"})
            else:
                edges["t"] = np.arange(len(edges), dtype=np.int64)
        return edges[["src", "dst", "t"]]

    # --------------------------------------------------
    # Case 2: alternative minimal: u, i, ts
    # --------------------------------------------------
    if {"u", "i"}.issubset(cols):
        if "ts" in cols:
            edges = edges.rename(columns={"u": "src", "i": "dst", "ts": "t"})
        elif "t" in cols:
            edges = edges.rename(columns={"u": "src", "i": "dst"})
        else:
            edges = edges.rename(columns={"u": "src", "i": "dst"})
            edges["t"] = np.arange(len(edges), dtype=np.int64)
        return edges[["src", "dst", "t"]]

    # --------------------------------------------------
    # Case 3: TGB wiki / review CSV
    #   user_id, item_id, timestamp, (others ignored)
    # --------------------------------------------------
    if {"user_id", "item_id", "timestamp"}.issubset(cols):
        edges = edges.rename(columns={
            "user_id": "src",
            "item_id": "dst",
            "timestamp": "t",
        })
        return edges[["src", "dst", "t"]]

# --------------------------------------------------
# Case 4: DTDG-style TGB CSV (review, citation, etc.)
#   source, target, ts, (weight ignored)
# --------------------------------------------------
    if {"source", "target", "ts"}.issubset(cols):
        edges = edges.rename(columns={
            "source": "src",
            "target": "dst",
            "ts": "t",
        })
        return edges[["src", "dst", "t"]]
    # --------------------------------------------------
    # Otherwise: unsupported format
    # --------------------------------------------------
    raise ValueError(f"Unrecognized edgelist columns: {list(edges.columns)}")


def _infer_num_nodes(edges: pd.DataFrame) -> int:
    m = int(max(edges["src"].max(), edges["dst"].max()))
    return m + 1


def _build_group_map_from_nodes(nodes: Optional[pd.DataFrame]) -> Tuple[Dict[int, int], int]:
    if nodes is None:
        return {}, 0
    if "node" not in nodes.columns or "group" not in nodes.columns:
        return {}, 0

    gm: Dict[int, int] = {}
    groups = set()
    for r in nodes.itertuples(index=False):
        n = int(getattr(r, "node"))
        g = int(getattr(r, "group"))
        gm[n] = g
        if g >= 0:
            groups.add(g)
    ng = len(groups)
    return gm, ng


def _build_tgb_group_map(
    edges: pd.DataFrame,
    num_nodes: int,
    mode: str = "none",
    n_groups: int = 2,
    warmup: int = 20000,
) -> Tuple[Dict[int, int], int, List[int]]:
    """Build a *synthetic* protected-group map for TGB datasets.

    TGB datasets do not ship with a protected attribute in this repo. For experiments that need
    groups (e.g., TGN adversarial debiasing / penalty baselines, and OPP-COPF fairness metrics),
    we optionally create groups from node IDs or simple structural stats.

    Modes:
      - none: every node in group 0
      - src_mod: group(u) = u % n_groups
      - src_degree: degree bucket of src node based on a warmup prefix

    Returns: (group_map, n_groups_found, groups_list)
    """
    mode = str(mode).lower().strip()
    if mode == "none":
        return {}, 0, [0]

    n_groups = max(2, int(n_groups))

    # Default: deterministic mapping by src id
    if mode == "src_mod":
        gm = {int(u): int(u) % n_groups for u in range(int(num_nodes))}
        return gm, n_groups, list(range(n_groups))

    # Degree-bucket mapping using a warmup prefix
    if mode == "src_degree":
        warmup = max(1, int(warmup))
        prefix = edges.iloc[: min(warmup, len(edges))]
        deg = np.zeros(int(num_nodes), dtype=np.int64)
        for r in prefix.itertuples(index=False):
            u = int(getattr(r, "src"))
            if 0 <= u < num_nodes:
                deg[u] += 1

        # bucket by quantiles into n_groups
        qs = np.quantile(deg, np.linspace(0, 1, n_groups + 1))
        # make monotone / unique edges
        qs = np.unique(qs)
        if len(qs) <= 2:
            gm = {int(u): int(u) % n_groups for u in range(int(num_nodes))}
            return gm, n_groups, list(range(n_groups))

        def bucket(d: int) -> int:
            # find rightmost q <= d
            b = int(np.searchsorted(qs[1:], d, side="right"))
            return int(np.clip(b, 0, n_groups - 1))

        gm = {int(u): bucket(int(deg[u])) for u in range(int(num_nodes))}
        return gm, n_groups, list(range(n_groups))

    raise ValueError(f"Unknown --tgb_group_mode '{mode}'.")

def _rank_of_positive(v_list: np.ndarray, scores: np.ndarray, v_pos: int) -> int:
    order = np.lexsort((v_list, -scores))
    sorted_vs = v_list[order]
    idx = int(np.where(sorted_vs == v_pos)[0][0])
    return idx + 1


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synth", "tgb"], required=True)
    ap.add_argument("--data_dir", type=str, default="data/synth/bipartite_v1")
    ap.add_argument("--tgb_edgelist", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model", choices=["edgebank", "tgn", "graphmixer"], required=True)

    ap.add_argument("--T", type=int, default=50000)
    ap.add_argument("--neg", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--hits_k", type=int, default=10)

    ap.add_argument("--bipartite", action="store_true")
    ap.add_argument("--n_users", type=int, default=-1)

    ap.add_argument("--tgb_root", type=str, default=DEFAULT_TGB_ROOT)
    # synthetic group definition for TGB (needed for adv/penalty fairness baselines)
    ap.add_argument("--tgb_group_mode", type=str, default="none", choices=["none", "src_mod", "src_degree"],
                    help="How to create protected groups on TGB when none are provided.")
    ap.add_argument("--tgb_group_n", type=int, default=2, help="Number of groups for TGB synthetic grouping.")
    ap.add_argument("--tgb_group_warmup", type=int, default=20000,
                    help="Warmup prefix length for src_degree grouping (quantile buckets).")


    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda")

    # TGN-specific (keep as-is)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--msg_dim", type=int, default=128)
    ap.add_argument("--train_every", type=int, default=1)
    ap.add_argument("--adv_lambda", type=float, default=0.0)
    ap.add_argument("--n_groups", type=int, default=0)
    ap.add_argument("--reweight", type=str, default="none", choices=["none", "inv_freq"])
    ap.add_argument("--fair_penalty_lambda", type=float, default=0.0)

    # GraphMixer-specific
    ap.add_argument("--gm_num_tokens", type=int, default=20)
    ap.add_argument("--gm_time_feat_dim", type=int, default=128)
    ap.add_argument("--gm_num_layers", type=int, default=2)
    ap.add_argument("--gm_dropout", type=float, default=0.1)
    ap.add_argument("--gm_token_dim_expansion_factor", type=float, default=0.5)
    ap.add_argument("--gm_channel_dim_expansion_factor", type=float, default=4.0)
    ap.add_argument("--gm_num_neighbors", type=int, default=20)
    ap.add_argument("--gm_time_gap", type=int, default=2000)
    ap.add_argument("--gm_node_feat_dim", type=int, default=-1)
    ap.add_argument("--gm_edge_feat_dim", type=int, default=-1)

    args = ap.parse_args()
    _ensure_dir(args.out_dir)

    rng = np.random.default_rng(args.seed)

    # Load data
    if args.dataset == "synth":
        edges, nodes, meta = _load_synth(args.data_dir)
        group_map, n_groups_found = _build_group_map_from_nodes(nodes)

        # IMPORTANT: trust meta.num_nodes if present (can be > max id in edges)
        num_nodes = int(meta.get("num_nodes", _infer_num_nodes(edges)))

        n_users = int(meta.get("n_users", args.n_users if args.n_users > 0 else -1))
        if args.bipartite and n_users <= 0:
            raise ValueError("For --bipartite synth, need meta.json with n_users or pass --n_users.")
    else:
        if not args.tgb_edgelist:
            raise ValueError("--tgb_edgelist is required when --dataset tgb")
        edges = _load_tgb_csv(args.tgb_edgelist)
        num_nodes = _infer_num_nodes(edges)
        group_map, n_groups_found, _ = _build_tgb_group_map(
            edges, num_nodes, mode=args.tgb_group_mode, n_groups=args.tgb_group_n, warmup=args.tgb_group_warmup
        )
        n_users = -1

    T = min(int(args.T), len(edges))

    # Initialize model
    if args.model == "edgebank":
        model = TGBEdgeBankOnlineAdapter(tgb_root=args.tgb_root)
        model_name = "EdgeBank"
        edges_use = edges
        item_pool = np.arange(num_nodes, dtype=np.int64)

    elif args.model == "graphmixer":
        # reserve node 0 for padding -> shift ids
        edges_use = edges.copy()
        edges_use["src"] = edges_use["src"].astype(np.int64) + 1
        edges_use["dst"] = edges_use["dst"].astype(np.int64) + 1
        num_nodes_gm = int(num_nodes) + 1  # include padding 0 and all meta nodes

        if args.dataset == "synth" and args.bipartite:
            item_pool = np.arange(n_users + 1, num_nodes_gm, dtype=np.int64)
        else:
            item_pool = np.arange(1, num_nodes_gm, dtype=np.int64)

        src_arr = edges_use["src"].to_numpy(dtype=np.int64)
        dst_arr = edges_use["dst"].to_numpy(dtype=np.int64)
        t_arr = edges_use["t"].to_numpy(dtype=np.float64)
        eidx_arr = np.arange(1, len(edges_use) + 1, dtype=np.int64)

        # IMPORTANT FIX: pass num_nodes=num_nodes_gm to avoid IndexError on isolated high-id negatives
        neighbor_sampler = CompatNeighborSampler(
            src_arr, dst_arr, t_arr, eidx_arr,
            seed=args.seed,
            num_nodes=num_nodes_gm,
        )

        node_feat_dim = args.gm_node_feat_dim if args.gm_node_feat_dim > 0 else args.emb_dim
        edge_feat_dim = args.gm_edge_feat_dim if args.gm_edge_feat_dim > 0 else args.emb_dim

        node_raw_features = np.zeros((num_nodes_gm, node_feat_dim), dtype=np.float32)
        edge_raw_features = np.zeros((len(edges_use) + 1, edge_feat_dim), dtype=np.float32)

        cfg = GraphMixerConfig(
            num_nodes=num_nodes_gm,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed,
            tgb_root=args.tgb_root,
            gm_num_tokens=int(args.gm_num_tokens),
            gm_time_feat_dim=int(args.gm_time_feat_dim),
            gm_num_layers=int(args.gm_num_layers),
            gm_token_dim_expansion_factor=float(args.gm_token_dim_expansion_factor),
            gm_channel_dim_expansion_factor=float(args.gm_channel_dim_expansion_factor),
            gm_dropout=float(args.gm_dropout),
            gm_num_neighbors=int(args.gm_num_neighbors),
            gm_time_gap=int(args.gm_time_gap),
        )

        model = GraphMixerWrapper(
            cfg,
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=neighbor_sampler,
        )
        model_name = "GraphMixer"

        with open(os.path.join(args.out_dir, "graphmixer_config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)

    else:
        from fl_models.wrappers.tgn_wrapper import TGNWrapper, TGNConfig  # noqa

        edges_use = edges
        item_pool = np.arange(num_nodes, dtype=np.int64)

        ng = args.n_groups if args.n_groups > 0 else n_groups_found
        cfg = TGNConfig(
            num_nodes=num_nodes,
            emb_dim=args.emb_dim,
            msg_dim=args.msg_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            train_every=args.train_every,
            adv_lambda=args.adv_lambda,
            n_groups=max(0, int(ng)),
            reweight=args.reweight,
            fair_penalty_lambda=args.fair_penalty_lambda,
            seed=args.seed,
        )
        model = TGNWrapper(cfg)
        model_name = f"TGN(adv={args.adv_lambda}, reweight={args.reweight}, pen={args.fair_penalty_lambda})"
        with open(os.path.join(args.out_dir, "tgn_config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)

    # -----------------------------------------------------------------
    # Online loop
    # -----------------------------------------------------------------
    rows = []
    w_mrr = 0.0
    w_ap = 0.0
    w_hits = 0.0
    w_n = 0

    for i in range(T):
        src = int(edges_use.iloc[i]["src"])
        dst = int(edges_use.iloc[i]["dst"])
        t = int(edges_use.iloc[i]["t"])

        g = 0
        if args.model == "tgn":
            g = int(group_map.get(src, 0)) if group_map else 0

        k = min(int(args.neg), len(item_pool))
        neg_vs = rng.choice(item_pool, size=k, replace=False)
        neg_vs = neg_vs[neg_vs != dst]
        cand_vs = np.unique(np.concatenate([[dst], neg_vs])).astype(np.int64)

        if args.model == "edgebank":
            scores = np.array([float(model.score(src, int(v), t)) for v in cand_vs], dtype=float)
        else:
            scores = model.score(src, cand_vs.tolist(), t, src_group=g)

        rank = _rank_of_positive(cand_vs, scores, dst)
        mrr = 1.0 / float(rank)
        apv = mrr
        hitsk = 1.0 if rank <= int(args.hits_k) else 0.0

        if args.model == "edgebank":
            model.update(src, dst, t)
        else:
            model.update(src, dst, neg_vs.tolist(), t, src_group=g)

        w_mrr += mrr
        w_ap += apv
        w_hits += hitsk
        w_n += 1

        if (i + 1) % int(args.log_every) == 0:
            row = {
                "step": i + 1,
                "mrr": w_mrr / max(1, w_n),
                "ap": w_ap / max(1, w_n),
                f"hits@{args.hits_k}": w_hits / max(1, w_n),
            }
            if hasattr(model, "diagnostics"):
                try:
                    row.update(model.diagnostics())
                except Exception:
                    pass

            rows.append(row)

            extra = ""
            if "last_loss" in row:
                extra = f"loss={row.get('last_loss', np.nan):.4f} "
            print(
                f"[{i+1}/{T}] "
                f"mrr={row['mrr']:.4f} ap={row['ap']:.4f} hits@{args.hits_k}={row[f'hits@{args.hits_k}']:.4f} "
                f"{extra}"
            )

            w_mrr = w_ap = w_hits = 0.0
            w_n = 0

    out_csv = os.path.join(args.out_dir, "base_metrics.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    with open(os.path.join(args.out_dir, "run_info.json"), "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "model": model_name,
                "T": T,
                "neg": args.neg,
                "hits_k": args.hits_k,
                "seed": args.seed,
                "tgb_root": args.tgb_root,
            },
            f,
            indent=2,
        )

    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()