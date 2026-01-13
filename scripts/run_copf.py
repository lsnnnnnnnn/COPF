# scripts/run_copf_v2_opp.py
from __future__ import annotations

"""A more paper-aligned OPP runner for COPF.

What this version fixes/implements compared to your current run_copf_v1.py:
  1) **OPP phases**: pre → deploy → post (Section 13) with per-phase policies.
  2) **Correct score semantics**: always convert model outputs to probabilities in (0,1)
     before calibration/decision/auditing.
  3) **Correct fairness calls**: gTE is computed from DR; gCal returns a dict; we log max.
  4) **Residual-OI auditing + noisy transfer certificate**: logs eps/beta/p_min bounds.
  5) **GraphMixer group-map shift**: keeps group labels consistent after +1 node-id hack.

This file is meant to be a drop-in replacement for scripts/run_copf_v1.py.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import importlib.util
import inspect
import types

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Make repo root importable (so `import copf.*` works)
# ---------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_TGB_ROOT = os.path.join(REPO_ROOT, "tgb_baselines")


# ---------------------------------------------------------------------
# COPF imports
# ---------------------------------------------------------------------
from copf.decision import decide_with_exploration
from copf.propensity import log_propensities
from copf.cross_fit import OnlineCrossFitter
from copf.dr import GraphAwareDR, DRConfig
from copf.audit import MultiCalibrator, AuditorConfig
from copf.eval import (
    mrr as eval_mrr,
    average_precision as eval_ap,
    hits_at_k as eval_hits,
    ndcg_at_k as eval_ndcg,
    export_phase_results,
)
from copf.fairness import gCal, gTE, gMin, gRisk

# Residual-OI auditing utilities (new file: copf/oi_audit.py)
from copf.oi_audit import OIAuditConfig, ResidualOIAuditor


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
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
    keys = [k for k in list(sys.modules.keys()) if k == prefix or k.startswith(prefix + ".")]
    for k in keys:
        del sys.modules[k]


def _push_sys_path_after_cwd(path0: str) -> None:
    if path0 in sys.path:
        sys.path.remove(path0)
    insert_pos = 1 if (len(sys.path) > 0 and sys.path[0] == "") else 0
    sys.path.insert(insert_pos, path0)


def _as_prob(scores: np.ndarray) -> np.ndarray:
    """Convert arbitrary model scores to probabilities in (0,1).

    - If scores look like logits (outside [0,1]), apply sigmoid.
    - Else treat as probabilities and clip.
    """
    s = np.asarray(scores, dtype=float)
    if np.any(s < 0.0) or np.any(s > 1.0):
        s = 1.0 / (1.0 + np.exp(-s))
    return np.clip(s, 1e-4, 1.0 - 1e-4)


# ---------------------------------------------------------------------
# Compat modules for GraphMixer (same as your run_base)
# ---------------------------------------------------------------------
class CompatTimeEncoder(nn.Module):
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

        if num_nodes is None:
            max_nid = int(max(src.max(initial=0), dst.max(initial=0)))
            self.num_nodes = max_nid + 1
        else:
            self.num_nodes = int(num_nodes)

        self.adj: List[List[Tuple[float, int, int]]] = [[] for _ in range(self.num_nodes)]
        for s, d, t, ei in zip(src, dst, ts, eidx):
            s = int(s)
            d = int(d)
            if 0 <= s < self.num_nodes:
                self.adj[s].append((float(t), d, int(ei)))
            if 0 <= d < self.num_nodes:
                self.adj[d].append((float(t), s, int(ei)))

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
                nb = int(nb)
                nbr_nodes[i, j] = nb if (0 <= nb < self.num_nodes) else 0
                nbr_eidx[i, j] = int(ei)

        return nbr_nodes, nbr_eidx, nbr_ts

    def get_temporal_neighbor(self, node_ids, node_interact_times, num_neighbors):
        return self.get_historical_neighbors(np.array(node_ids), np.array(node_interact_times), int(num_neighbors))


def _install_graphmixer_compat_modules() -> None:
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
        u = int(u)
        v = int(v)
        t = int(t)
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
        u = int(u)
        v = int(v)
        t = int(t)
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
# GraphMixer Config + Wrapper (same as your run_copf_v1)
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
    def __init__(self, cfg: GraphMixerConfig, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: Any):
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


def _load_tgb_csv_robust(edgelist_csv: str) -> pd.DataFrame:
    """Robust loader for:
      - standard csv with headers (src,dst,t) / (u,i,ts) / etc.
      - whitespace-separated DTDG timestamp files
      - headerless comma-separated triples
    """
    edges = pd.read_csv(edgelist_csv)
    cols = list(edges.columns)

    # If it looks like a single text column, try alternative parsing.
    if len(cols) == 1 and str(cols[0]) in {"0", "Unnamed: 0"}:
        # try whitespace
        for sep, engine in [(r"\s+", "python"), ("\t", None), (",", "python")]:
            try:
                edges2 = pd.read_csv(edgelist_csv, sep=sep, header=None, engine=engine)
                if edges2.shape[1] >= 3:
                    edges2 = edges2.iloc[:, :3]
                    edges2.columns = ["src", "dst", "t"]
                    return edges2
            except Exception:
                pass
        raise ValueError(f"Could not parse '{edgelist_csv}'. It reads as a single column: {cols}.")

    cols_set = set(edges.columns)

    if {"src", "dst"}.issubset(cols_set):
        if "t" not in cols_set:
            if "ts" in cols_set:
                edges = edges.rename(columns={"ts": "t"})
            elif "time" in cols_set:
                edges = edges.rename(columns={"time": "t"})
            else:
                edges["t"] = np.arange(len(edges), dtype=np.int64)
        return edges[["src", "dst", "t"]]

    if {"u", "i"}.issubset(cols_set):
        if "ts" in cols_set:
            edges = edges.rename(columns={"u": "src", "i": "dst", "ts": "t"})
        elif "t" in cols_set:
            edges = edges.rename(columns={"u": "src", "i": "dst"})
        else:
            edges = edges.rename(columns={"u": "src", "i": "dst"})
            edges["t"] = np.arange(len(edges), dtype=np.int64)
        return edges[["src", "dst", "t"]]

    if {"user_id", "item_id", "timestamp"}.issubset(cols_set):
        edges = edges.rename(columns={"user_id": "src", "item_id": "dst", "timestamp": "t"})
        return edges[["src", "dst", "t"]]

    if {"source", "target", "ts"}.issubset(cols_set):
        edges = edges.rename(columns={"source": "src", "target": "dst", "ts": "t"})
        return edges[["src", "dst", "t"]]

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
    mode: str,
    n_groups: int,
    warmup: int,
) -> Tuple[Dict[int, int], int, List[int]]:
    """Construct a static protected-attribute map for TGB datasets.

    TGB datasets typically don't ship demographic group labels. For OPP/COPF
    experiments we often still need *some* notion of protected attribute A on
    the *source* side. This helper builds a deterministic grouping based on
    node IDs or (warm-start) degrees.

    Args:
        edges: DataFrame with columns [src, dst, t]
        num_nodes: total number of nodes in the dataset
        mode:
            - 'none'      : a=0 for all sources
            - 'src_mod'   : a = src % n_groups
            - 'src_degree': a = quantile(deg(src)) using first `warmup` edges
        n_groups: number of groups (>=1)
        warmup: number of initial edges to estimate degrees for 'src_degree'

    Returns:
        group_map: dict {node_id -> group_id} (only for nodes that appear as src)
        n_groups_found: effective number of groups (may be < n_groups if
                        quantile cuts collapse)
        groups_list: list of group IDs to use downstream
    """
    mode = str(mode).strip().lower()
    n_groups = max(1, int(n_groups))
    warmup = max(1, int(warmup))

    if mode in {"none", ""}:
        return {}, 0, [0]

    src_nodes = np.unique(edges["src"].to_numpy(dtype=np.int64))
    if src_nodes.size == 0:
        return {}, 0, [0]

    if mode == "src_mod":
        gm = {int(u): int(u) % n_groups for u in src_nodes.tolist()}
        return gm, n_groups, list(range(n_groups))

    if mode == "src_degree":
        warm = min(len(edges), warmup)
        src_w = edges.iloc[:warm]["src"].to_numpy(dtype=np.int64)
        deg = np.bincount(src_w, minlength=int(num_nodes)).astype(np.float64)

        # Build quantile cut points; if they collapse (e.g., all degrees equal),
        # fall back to fewer effective groups.
        if n_groups == 1:
            cuts = np.array([], dtype=np.float64)
        else:
            qs = np.linspace(0.0, 1.0, n_groups + 1, dtype=np.float64)[1:-1]
            cuts = np.quantile(deg, qs)
            cuts = np.unique(cuts)

        n_eff = max(1, int(len(cuts) + 1))
        grp = np.digitize(deg, bins=cuts, right=False).astype(np.int64)

        gm = {int(u): int(grp[int(u)]) for u in src_nodes.tolist()}
        return gm, n_eff, list(range(n_eff))

    raise ValueError(f"Unknown --tgb_group_mode='{mode}'.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    # dataset
    ap.add_argument("--dataset", choices=["synth", "tgb"], required=True)
    ap.add_argument("--data_dir", type=str, default="data/synth/bipartite_v1")
    ap.add_argument("--tgb_edgelist", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)

    # model
    ap.add_argument("--model", choices=["edgebank", "tgn", "graphmixer"], required=True)

    # online protocol
    ap.add_argument("--T", type=int, default=50000)
    ap.add_argument("--neg", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--hits_k", type=int, default=10)

    ap.add_argument("--bipartite", action="store_true")
    ap.add_argument("--n_users", type=int, default=-1)

    # Protected attribute construction for TGB (since TGB datasets do not ship with demographic groups)
    ap.add_argument(
        "--tgb_group_mode",
        type=str,
        default="none",
        choices=["none", "src_mod", "src_degree"],
        help=(
            "How to construct protected group A for --dataset tgb. "
            "none: all sources in group 0. "
            "src_mod: a = src_id % tgb_group_n. "
            "src_degree: a = quantile(degree(src)) computed from the first tgb_group_warmup edges."
        ),
    )
    ap.add_argument(
        "--tgb_group_n",
        type=int,
        default=2,
        help="Number of groups for --tgb_group_mode=src_mod or src_degree (default: 2).",
    )
    ap.add_argument(
        "--tgb_group_warmup",
        type=int,
        default=20000,
        help="How many first edges to estimate degrees for --tgb_group_mode=src_degree (default: 20000).",
    )

    ap.add_argument("--tgb_root", type=str, default=DEFAULT_TGB_ROOT)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda")

    # TGN-specific
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

    # Outcome simulation (how we define observed y)
    ap.add_argument(
        "--outcome_mode",
        type=str,
        default="bandit",
        choices=["bandit", "full"],
        help=(
            "Observed outcome y in the OPP simulation. "
            "bandit: y = y_true if d==1 else 0 (only exposed items can yield reward). "
            "full: y = y_true for all candidates (FULL-information, not OPP)."
        ),
    )

    # ---------------- COPF: decision policy ----------------
    ap.add_argument("--policy", type=str, default="epsilon_greedy", choices=["epsilon_greedy", "topk_stochastic"])
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=1.0)

    # OPP phases (Section 13): pre → deploy → post
    ap.add_argument("--pre_T", type=int, default=0, help="If >0, use pre/deploy/post protocol")
    ap.add_argument("--deploy_T", type=int, default=0)
    ap.add_argument("--post_T", type=int, default=0)
    ap.add_argument("--deploy_topk", type=int, default=-1)
    ap.add_argument("--deploy_epsilon", type=float, default=-1.0)
    ap.add_argument("--deploy_temperature", type=float, default=-1.0)
    ap.add_argument("--post_topk", type=int, default=-1)
    ap.add_argument("--post_epsilon", type=float, default=-1.0)
    ap.add_argument("--post_temperature", type=float, default=-1.0)

    # auditing/calibration knobs
    ap.add_argument("--audit_every", type=int, default=200)
    ap.add_argument("--pre_apply_calibrator", action="store_true")
    ap.add_argument("--aud_buckets_per_group", type=int, default=10)
    ap.add_argument("--aud_budget_B", type=int, default=64)
    ap.add_argument("--aud_step", type=float, default=0.25)
    ap.add_argument("--aud_min_mass", type=float, default=0.02)
    ap.add_argument("--aud_isotonic", action="store_true")

    # DR knobs
    ap.add_argument("--dr_clip", type=float, default=0.05)
    ap.add_argument("--dr_decay_gamma", type=float, default=0.1)
    ap.add_argument("--dr_max_buffer", type=int, default=200000)
    ap.add_argument("--dr_winsor_eps", type=float, default=0.02)
    ap.add_argument("--dr_min_eff", type=int, default=100)
    ap.add_argument("--dr_no_sn", action="store_true")
    ap.add_argument("--dr_no_ratio_stab", action="store_true")

    # cross-fit
    ap.add_argument("--cf_folds", type=int, default=5)
    ap.add_argument("--cf_max_buffer", type=int, default=20000)
    ap.add_argument(
        "--cf_update_every",
        type=int,
        default=200,
        help="How often to refit cross-fitting nuisance models (in rounds). "
        "Set to 1 to refit every round (expensive).",
    )
    ap.add_argument("--no_crossfit", action="store_true")

    # Residual-OI auditing (certificates)
    ap.add_argument("--oi_window", type=int, default=50000)
    ap.add_argument("--oi_include_struct", action="store_true")
    ap.add_argument("--oi_struct_bins", type=int, default=5)
    ap.add_argument("--oi_budget_B", type=int, default=128)
    ap.add_argument("--oi_rff_dim", type=int, default=0)
    ap.add_argument("--oi_rff_gamma", type=float, default=1.0)

    # gMin threshold (if <0, we set it from pre phase 10th percentile of tau)
    ap.add_argument("--tau_min", type=float, default=-1.0)

    args = ap.parse_args()
    _ensure_dir(args.out_dir)

    rng = np.random.default_rng(args.seed)

    # ============================================================
    # Load data
    # ============================================================
    if args.dataset == "synth":
        edges, nodes, meta = _load_synth(args.data_dir)
        group_map, n_groups_found = _build_group_map_from_nodes(nodes)
        num_nodes = int(meta.get("num_nodes", _infer_num_nodes(edges)))

        n_users = int(meta.get("n_users", args.n_users if args.n_users > 0 else -1))
        if args.bipartite and n_users <= 0:
            raise ValueError("For --bipartite synth, need meta.json with n_users or pass --n_users.")

        # Only keep valid (>=0) groups; in bipartite synth, item nodes may have group=-1.
        groups_list = sorted({int(g) for g in group_map.values() if g is not None and int(g) >= 0})
        if not groups_list:
            groups_list = [0]
    else:
        if not args.tgb_edgelist:
            raise ValueError("--tgb_edgelist is required when --dataset tgb")
        edges = _load_tgb_csv_robust(args.tgb_edgelist)
        num_nodes = _infer_num_nodes(edges)
        n_users = -1

        # Build synthetic protected groups for TGB if requested.
        group_map, n_groups_found, groups_list = _build_tgb_group_map(
            edges=edges,
            num_nodes=num_nodes,
            mode=str(args.tgb_group_mode),
            n_groups=int(args.tgb_group_n),
            warmup=int(args.tgb_group_warmup),
        )

    T_total = min(int(args.T), len(edges))

    # ============================================================
    # OPP phase split
    # ============================================================
    if int(args.pre_T) > 0:
        pre_T = int(args.pre_T)
        deploy_T = int(args.deploy_T)
        post_T = int(args.post_T)
        if pre_T + deploy_T + post_T <= 0:
            raise ValueError("pre_T+deploy_T+post_T must be >0")
        T = min(T_total, pre_T + deploy_T + post_T)
    else:
        pre_T, deploy_T, post_T = T_total, 0, 0
        T = T_total

    def _phase(i: int) -> str:
        if i < pre_T:
            return "pre"
        if i < pre_T + deploy_T:
            return "deploy"
        return "post"

    def _policy_params(phase: str) -> Tuple[int, float, float]:
        if phase == "deploy":
            topk = int(args.deploy_topk) if args.deploy_topk > 0 else int(args.topk)
            eps = float(args.deploy_epsilon) if args.deploy_epsilon >= 0.0 else float(args.epsilon)
            temp = float(args.deploy_temperature) if args.deploy_temperature >= 0.0 else float(args.temperature)
            return topk, eps, temp
        if phase == "post":
            topk = int(args.post_topk) if args.post_topk > 0 else int(args.topk)
            eps = float(args.post_epsilon) if args.post_epsilon >= 0.0 else float(args.epsilon)
            temp = float(args.post_temperature) if args.post_temperature >= 0.0 else float(args.temperature)
            return topk, eps, temp
        return int(args.topk), float(args.epsilon), float(args.temperature)

    # ============================================================
    # Initialize model
    # ============================================================
    if args.model == "edgebank":
        model = TGBEdgeBankOnlineAdapter(tgb_root=args.tgb_root)
        model_name = "EdgeBank"
        edges_use = edges
        # Candidate pool for negative sampling.
        # For bipartite synth: restrict candidates to the item side.
        if args.dataset == "synth" and args.bipartite and n_users > 0:
            item_pool = np.arange(n_users, num_nodes, dtype=np.int64)
        else:
            item_pool = np.arange(num_nodes, dtype=np.int64)

    elif args.model == "graphmixer":
        edges_use = edges.copy()
        # GraphMixer baseline reserves node id 0
        edges_use["src"] = edges_use["src"].astype(np.int64) + 1
        edges_use["dst"] = edges_use["dst"].astype(np.int64) + 1
        num_nodes_gm = int(num_nodes) + 1

        # IMPORTANT: keep group_map consistent
        if group_map:
            group_map = {int(k) + 1: int(v) for k, v in group_map.items()}
            groups_list = (
                sorted({int(g) for g in group_map.values() if g is not None and int(g) >= 0}) or [0]
            )

        if args.dataset == "synth" and args.bipartite:
            item_pool = np.arange(n_users + 1, num_nodes_gm, dtype=np.int64)
        else:
            item_pool = np.arange(1, num_nodes_gm, dtype=np.int64)

        src_arr = edges_use["src"].to_numpy(dtype=np.int64)
        dst_arr = edges_use["dst"].to_numpy(dtype=np.int64)
        t_arr = edges_use["t"].to_numpy(dtype=np.float64)
        eidx_arr = np.arange(1, len(edges_use) + 1, dtype=np.int64)

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

        model = GraphMixerWrapper(cfg, node_raw_features, edge_raw_features, neighbor_sampler)
        model_name = "GraphMixer"
        with open(os.path.join(args.out_dir, "graphmixer_config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)

    else:
        # keep your existing TGN wrapper path
        from fl_models.wrappers.tgn_wrapper import TGNWrapper, TGNConfig  # noqa

        edges_use = edges

        # Candidate pool for negative sampling.
        # For bipartite synth: restrict candidates to the item side.
        if args.dataset == "synth" and bool(args.bipartite):
            if n_users <= 0:
                raise ValueError("--bipartite requires --n_users or meta.json to provide n_users")
            item_pool = np.arange(int(n_users), int(num_nodes), dtype=np.int64)
        else:
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

    # ============================================================
    # COPF components: Cross-fit + GA-DR + Residual-OI calibrator
    # ============================================================
    cross_fitter = None
    if not args.no_crossfit:
        cross_fitter = OnlineCrossFitter(
            n_folds=int(args.cf_folds),
            seed=int(args.seed),
            max_buffer=int(args.cf_max_buffer),
        )

    dr_cfg = DRConfig(
        clip=float(args.dr_clip),
        self_normalized=(not args.dr_no_sn),
        decay_gamma=float(args.dr_decay_gamma),
        max_buffer=int(args.dr_max_buffer),
        ratio_stab=(not args.dr_no_ratio_stab),
        winsor_eps=float(args.dr_winsor_eps),
        min_eff_samples=int(args.dr_min_eff),
    )

    # NOTE: use positional arg to avoid signature mismatches (cfg vs config)
    dr_pre = GraphAwareDR(dr_cfg, cross_fitter=cross_fitter)
    dr_deploy = GraphAwareDR(dr_cfg, cross_fitter=cross_fitter)
    dr_post = GraphAwareDR(dr_cfg, cross_fitter=cross_fitter)

    def _dr_for_phase(ph: str) -> GraphAwareDR:
        return dr_pre if ph == "pre" else (dr_deploy if ph == "deploy" else dr_post)

    aud_cfg = AuditorConfig(
        buckets_per_group=int(args.aud_buckets_per_group),
        isotonic=bool(args.aud_isotonic),
        budget_B=int(args.aud_budget_B),
        step=float(args.aud_step),
        min_mass=float(args.aud_min_mass),
    )
    calibrator = MultiCalibrator(cfg=aud_cfg)

    oi_cfg = OIAuditConfig(
        buckets_per_group=int(args.aud_buckets_per_group),
        min_mass=float(args.aud_min_mass),
        include_group_only=True,
        include_group_bucket=True,
        include_struct=bool(args.oi_include_struct),
        struct_bins=int(args.oi_struct_bins),
        budget_B=int(args.oi_budget_B),
        rff_dim=int(args.oi_rff_dim),
        rff_gamma=float(args.oi_rff_gamma),
        seed=int(args.seed),
        window=int(args.oi_window),
    )
    oi_auditor = ResidualOIAuditor(cfg=oi_cfg)

    # ============================================================
    # Simple online structural stats for x (features for nuisance models / kernel auditing)
    # ============================================================
    degree: Dict[int, int] = {}
    last_time: Dict[int, float] = {}
    t0: Optional[float] = None

    def _make_x(u: int, v: int, t: float) -> Dict[str, Any]:
        du = float(degree.get(u, 0))
        dv = float(degree.get(v, 0))

        ltu = float(last_time.get(u, t))
        ltv = float(last_time.get(v, t))

        return {
            # names aligned with cross_fit.extract_features
            "degree_u": du,
            "degree_v": dv,
            "common_neighbors": 0.0,
            "jaccard": 0.0,
            "aa_score": 0.0,
            "shortest_path": 0.0,
            "time_since_last_u": float(max(0.0, t - ltu)),
            "time_since_last_v": float(max(0.0, t - ltv)),
            # a single aggregate (some auditors want this name)
            "time_since_last": float(min(max(0.0, t - ltu), max(0.0, t - ltv))),
            "time_since_start": float(0.0 if t0 is None else max(0.0, t - t0)),
        }

    # ============================================================
    # Online loop
    # ============================================================
    rows: List[Dict[str, Any]] = []
    phase_results: Dict[str, Dict[str, Any]] = {}

    # phase accumulators
    util_sum = {"mrr": 0.0, "ap": 0.0, "hits": 0.0, "ndcg": 0.0, "deploy_hit": 0.0}
    util_n = 0
    phase_cur = "pre"
    tau_min_value: Optional[float] = None

    def _flush_phase(phase_name: str) -> None:
        nonlocal util_sum, util_n, tau_min_value

        dr_phase = _dr_for_phase(phase_name)
        # utility averages
        util = {
            "mrr": float(util_sum["mrr"] / max(1, util_n)),
            "ap": float(util_sum["ap"] / max(1, util_n)),
            f"hits@{args.hits_k}": float(util_sum["hits"] / max(1, util_n)),
            f"ndcg@{args.hits_k}": float(util_sum["ndcg"] / max(1, util_n)),
            "deploy_hit@topk": float(util_sum["deploy_hit"] / max(1, util_n)),
            "n": int(util_n),
        }

        # fairness
        gte_gap = float(gTE(dr_phase, groups_list))

        te_by_group = dr_phase.estimate_TE_by_group(group_key="a")
        tau_by_group = {
            str(g): {"tau": float(v), "n_eff": int(n)} for g, (v, n) in te_by_group.items()
        }
        tau_vals = [float(v) for (v, n) in te_by_group.values() if n > 0]
        tau_avg = float(np.mean(np.asarray(tau_vals, float))) if tau_vals else 0.0
        tau_min_obs = float(np.min(np.asarray(tau_vals, float))) if tau_vals else 0.0
        gcal_dict = gCal(
            batch=list(dr_phase.buffer),
            dr=dr_phase,
            groups=groups_list,
            buckets_per_group=int(args.aud_buckets_per_group),
            isotonic=bool(args.aud_isotonic),
            min_mass=float(args.aud_min_mass),
        )
        gcal_max = float(max(gcal_dict.values())) if gcal_dict else 0.0

        # tau_min from pre phase if requested
        if tau_min_value is None:
            if float(args.tau_min) >= 0.0:
                tau_min_value = float(args.tau_min)
            else:
                # 10th percentile of per-group tau (paper protocol)
                te = dr_phase.estimate_TE_by_group(group_key="a")
                vals = [float(v) for (v, n) in te.values() if n > 0]
                tau_min_value = float(np.percentile(np.asarray(vals, float), 10)) if vals else 0.0

        gmin = float(gMin(dr_phase, tau_min=float(tau_min_value or 0.0)))
        grisk = float(gRisk(dr_phase, groups_list))

        # residual-OI + transfer certificate
        oi = oi_auditor.audit(dr_phase.buffer, groups=groups_list)

        fairness = {
            "gTE_gap": gte_gap,
            "gCal_max": gcal_max,
            "gMin": gmin,
            "gRisk": grisk,
            "tau_avg": tau_avg,
            "tau_min_obs": tau_min_obs,
            "tau_by_group": tau_by_group,
            "tau_min": float(tau_min_value or 0.0),
            "oi": oi,
        }
        phase_results[phase_name] = {"utility": util, "fairness": fairness}

        # export detailed gCal per bucket for the phase
        gcal_path = os.path.join(args.out_dir, f"{phase_name}_gcal_detail.json")
        with open(gcal_path, "w") as f:
            json.dump({str(k): float(v) for k, v in gcal_dict.items()}, f, indent=2)

        # reset accumulators
        util_sum = {"mrr": 0.0, "ap": 0.0, "hits": 0.0, "ndcg": 0.0, "deploy_hit": 0.0}
        util_n = 0

    # ---------------- main streaming loop ----------------
    for i in range(T):
        phase = _phase(i)
        if phase != phase_cur:
            _flush_phase(phase_cur)
            phase_cur = phase

        dr_phase = _dr_for_phase(phase)

        src = int(edges_use.iloc[i]["src"])
        dst_true = int(edges_use.iloc[i]["dst"])
        t = float(edges_use.iloc[i]["t"])
        if t0 is None:
            t0 = t

        # group id a (protected attribute)
        a = int(group_map.get(src, 0)) if group_map else 0

        # candidate pool
        k = min(int(args.neg), len(item_pool))
        neg_vs = rng.choice(item_pool, size=k, replace=False)
        neg_vs = neg_vs[neg_vs != dst_true]
        cand_vs = np.unique(np.concatenate([[dst_true], neg_vs])).astype(np.int64)

        # model scores
        if args.model == "edgebank":
            raw = np.array([float(model.score(src, int(v), int(t))) for v in cand_vs], dtype=float)
        else:
            raw = model.score(src, cand_vs.tolist(), int(t), src_group=a)
        scores = _as_prob(raw)

        # build candidates
        cands: List[Dict[str, Any]] = []
        for v, p in zip(cand_vs.tolist(), scores.tolist()):
            cands.append(
                {
                    "u": int(src),
                    "v": int(v),
                    "t": int(t),
                    "a": int(a),
                    "y_true": 1.0 if int(v) == int(dst_true) else 0.0,
                    "x": _make_x(int(src), int(v), float(t)),
                    "p_hat": float(p),
                }
            )

        # apply r0-calibrator offsets BEFORE decision
        if phase != "pre" or bool(args.pre_apply_calibrator):
            calibrator.apply(cands)

        # decision + propensities
        topk, eps, temp = _policy_params(phase)
        d_list, e_list = decide_with_exploration(
            cands=cands,
            policy=str(args.policy),
            topk=int(topk),
            epsilon=float(eps),
            temperature=float(temp),
            rng=rng,
        )
        log_propensities(cands, d_list, e_list)

        # ---------------------------------------------------------------------
        # Observed outcomes used by the OPP / DR estimator.
        #
        # - outcome_mode=bandit (recommended for OPP): only exposed candidates
        #   generate feedback, i.e. y = y_true if d==1 else 0.
        # - outcome_mode=full: offline full-information logging, y = y_true for
        #   every candidate (useful only for debugging; TE will collapse).
        # ---------------------------------------------------------------------
        outcome_mode = str(args.outcome_mode).strip().lower()
        for c in cands:
            y_true = float(c.get("y_true", 0.0))
            d = int(c.get("d", 0))
            if outcome_mode == "bandit":
                c["y"] = y_true if d == 1 else 0.0
            else:
                c["y"] = y_true
            c["w_local"] = 1.0

        # Cross-fit nuisance update:
        # We always ingest new samples into the buffer, but only refit
        # the nuisance models every cf_update_every rounds (refit is expensive).
        if cross_fitter is not None:
            cf_every = max(1, int(args.cf_update_every))
            do_refit = ((i + 1) % cf_every == 0)
            cross_fitter.update(cands, train=do_refit)

        # DR ingest (phase-specific buffer)
        dr_phase.ingest(cands)

        # calibration update from DR periodically (stable)
        if (i + 1) % int(args.audit_every) == 0:
            calibrator.update_from_dr(dr_phase, max_items=5000, dr_clip=float(args.dr_clip))

        # evaluate ranking metrics on current candidate set
        step_mrr = float(eval_mrr(cands))
        step_ap = float(eval_ap(cands))
        step_hits = float(eval_hits(cands, int(args.hits_k)))
        step_ndcg = float(eval_ndcg(cands, int(args.hits_k)))

        deployed_hit = 1.0 if any((c.get("y_true", 0.0) == 1.0 and c.get("d", 0) == 1) for c in cands) else 0.0

        # model update (same supervision as baselines)
        if args.model == "edgebank":
            model.update(src, dst_true, int(t))
        else:
            model.update(src, dst_true, neg_vs.tolist(), int(t), src_group=a)

        # Update simple structural stats used as covariates (degree/recency).
        #
        # In bandit outcome mode, we only "observe" the positive (u, dst_true)
        # interaction when we actually expose the true dst (deployed_hit==1).
        degree[src] = degree.get(src, 0) + 1
        last_time[src] = float(t)

        if str(args.outcome_mode).lower() == "full" or deployed_hit > 0.0:
            degree[dst_true] = degree.get(dst_true, 0) + 1
            last_time[dst_true] = float(t)

        # accumulators
        util_sum["mrr"] += step_mrr
        util_sum["ap"] += step_ap
        util_sum["hits"] += step_hits
        util_sum["ndcg"] += step_ndcg
        util_sum["deploy_hit"] += deployed_hit
        util_n += 1

        # log row every log_every
        if (i + 1) % int(args.log_every) == 0:
            # fairness (priority: gTE)
            gte_gap = float(gTE(dr_phase, groups_list))
            gcal_dict = gCal(
                batch=list(dr_phase.buffer[-min(len(dr_phase.buffer), 20000):]),
                dr=dr_phase,
                groups=groups_list,
                buckets_per_group=int(args.aud_buckets_per_group),
                isotonic=bool(args.aud_isotonic),
                min_mass=float(args.aud_min_mass),
            )
            gcal_max = float(max(gcal_dict.values())) if gcal_dict else 0.0

            oi = oi_auditor.audit(dr_phase.buffer, groups=groups_list)

            aud_diag = calibrator.get_diagnostics()

            row: Dict[str, Any] = {
                "step": i + 1,
                "phase": phase,
                "mrr": util_sum["mrr"] / max(1, util_n),
                "ap": util_sum["ap"] / max(1, util_n),
                f"hits@{args.hits_k}": util_sum["hits"] / max(1, util_n),
                f"ndcg@{args.hits_k}": util_sum["ndcg"] / max(1, util_n),
                "deploy_hit@topk": util_sum["deploy_hit"] / max(1, util_n),

                # fairness + certificates
                "gTE_gap": gte_gap,
                "gCal_max": gcal_max,
                "bound_gTE_gap": float(oi.get("bound_gTE_gap", float("nan"))),
                "bound_gCal_max": float(oi.get("bound_gCal_max", float("nan"))),

                # raw OI stats (discrete)
                "eps_r0": float(oi.get("disc_r0", {}).get("eps_uncond", float("nan"))),
                "pmin_gb": float(oi.get("disc_r0", {}).get("pmin_group_bucket", float("nan"))),
                "beta_r0": float(oi.get("disc_r0", {}).get("beta", float("nan"))),
                "eps_r_delta": float(oi.get("disc_r_delta", {}).get("eps_uncond_group", float("nan"))),
                "pmin_g": float(oi.get("disc_r_delta", {}).get("pmin_group", float("nan"))),
                "beta_r_delta": float(oi.get("disc_r_delta", {}).get("beta_group", float("nan"))),

                # kernel OI diagnostics (Any-kernel auditing)
                "eps_r0_kernel": float(oi.get("kernel_r0", {}).get("eps", float("nan"))),
                "eps_r_delta_kernel": float(oi.get("kernel_r_delta", {}).get("eps", float("nan"))),

                # calibrator diagnostics
                "audit_n_active": aud_diag.get("n_active", 0),
                "audit_beta_t": aud_diag.get("beta_t", np.nan),
                "audit_total_samples": aud_diag.get("total_samples", 0),
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
                f"[OPP-COPF {i+1}/{T} {phase}] "
                f"mrr={row['mrr']:.4f} ap={row['ap']:.4f} hits@{args.hits_k}={row[f'hits@{args.hits_k}']:.4f} "
                f"gTE={row['gTE_gap']:.4f} gCal_max={row['gCal_max']:.4f} "
                f"boundTE={row['bound_gTE_gap']:.4f} boundCal={row['bound_gCal_max']:.4f} "
                f"n_active={row['audit_n_active']} beta={row['audit_beta_t']:.4f} "
                f"{extra}"
            )

    # flush last phase
    _flush_phase(phase_cur)

    out_csv = os.path.join(args.out_dir, "opp_copf_metrics.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # phase summary
    # Writes out_dir/summary.csv (and a "summary" row with drift) per copf/eval.py
    export_phase_results(phase_results, out_dir=args.out_dir)

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
                "policy": args.policy,
                "topk": args.topk,
                "epsilon": args.epsilon,
                "temperature": args.temperature,
                "phase": {"pre_T": pre_T, "deploy_T": deploy_T, "post_T": post_T},
                "audit_every": args.audit_every,
                "dr_cfg": asdict(dr_cfg),
                "aud_cfg": asdict(aud_cfg),
                "crossfit": (not args.no_crossfit),
                "cf_folds": args.cf_folds,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()