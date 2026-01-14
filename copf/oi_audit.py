"""copf/oi_audit.py

Residual-OI auditing for COPF.

This module provides:
  - OIAuditConfig: configuration for discrete (bucketed) + optional RFF (RKHS) auditors.
  - ResidualOIAuditor: the main interface used by scripts/run_copf.py.

Paper-alignment notes (COPF / Residual-OI):
  * We audit residual opportunity imbalance over *slices* S ⊆ X × A × R:
      - A: protected attribute / group
      - score buckets (within group)
      - optional structural roles / coarse structure features
  * When GA (graph-aware) weights are used (w_local + time decay),
    ALL expectations in the auditor are computed as GA-weighted self-normalized means:
        E_GA[f] = (Σ w_i f_i) / (Σ w_i)

Important runner compatibility detail:
-------------------------------------
`scripts/run_copf.py` calls:
    oi_auditor.audit(dr_phase.buffer, groups=groups_list)

where `groups_list` is typically the *set of group IDs* (e.g., [0,1]),
NOT a per-item group vector.

This auditor therefore supports BOTH:
  (a) groups=None                     -> infer group of each item from item['a']
  (b) groups=[0,1,...] (small list)   -> treat as allowed group IDs (filter)
  (c) groups=per_item_vector (len==len(buffer) or len==len(window)) -> override per-item groups

Implementation details:
  * equal_mass_buckets() in this repo returns bucket *intervals*: [(lo, hi), ...].
    Do NOT cast them to floats. Use bucket_index(intervals, value).

Expected fields in each DR buffer item (dict-like):
  - 'a'      : group id (int)
  - 'p_hat'  : base score/prob (float)
  - 'r0'     : residual for calibration gap (float)
  - 'r_delta': residual for treatment-effect gap (float)
  - 'w_local': optional locality weight (float, default 1.0)
  - 'ga_gamma' or 'decay_gamma': optional time-decay gamma (float, default 0.0)
  - 't_round' or 't': time index used for decay (int/float)

This file is meant to live at:
  fairlink/copf/oi_audit.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
from collections import defaultdict

import numpy as np

from fl_utils.buckets import bucket_index, equal_mass_buckets


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class OIAuditConfig:
    # Sliding window length over the DR buffer (0 or None => use whole buffer)
    window: int = 40000

    # Discrete auditors: score buckets per group
    buckets_per_group: int = 10
    min_mass: float = 1e-3  # min slice mass; slices smaller than this are ignored

    # Which discrete slice families to include
    include_group_only: bool = True         # slices keyed by group only
    include_group_bucket: bool = True       # slices keyed by (group, score-bucket)

    # Optional structural / feature bucket auditors (augment gb slices)
    include_struct: bool = False
    struct_bins: int = 5

    # Keep only the top-B violating slices in diagnostics (does NOT affect eps calculation)
    budget_B: int = 64

    # Any-kernel auditor via Random Fourier Features (RBF)
    rff_dim: int = 0
    rff_gamma: float = 1.0

    # Confidence level for (simple) concentration bounds
    delta: float = 0.05

    # Internal random seed
    seed: int = 123

    # Whether to use GA weights (w_local + time-decay). Keep True for paper alignment.
    use_ga_weights: bool = True


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _effective_n(w: np.ndarray) -> float:
    """Kish effective sample size for nonnegative weights."""
    w = np.asarray(w, dtype=float)
    s1 = float(w.sum())
    s2 = float(np.sum(w * w))
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


def _get_gamma(items: Sequence[Dict[str, Any]]) -> float:
    if not items:
        return 0.0
    g = items[0].get("ga_gamma", items[0].get("decay_gamma", 0.0))
    try:
        g = float(g)
    except Exception:
        g = 0.0
    if not np.isfinite(g) or g <= 0.0:
        return 0.0
    return g


def _get_time_key(items: Sequence[Dict[str, Any]]) -> str:
    if not items:
        return "t"
    if "t_round" in items[0]:
        return "t_round"
    return "t"


# ---------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------
class ResidualOIAuditor:
    """Residual-OI auditor used by scripts/run_copf.py."""

    def __init__(self, cfg: OIAuditConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # RFF parameters (initialized lazily once we know feature dimension)
        self._rff_W: Optional[np.ndarray] = None
        self._rff_b: Optional[np.ndarray] = None
        self._rff_in_dim: Optional[int] = None

    # -----------------
    # Weighting helpers
    # -----------------
    def _ga_weights(self, items: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, float, float]:
        """Return (w, sum_w, n_eff). Uses GA weights if configured & available."""
        n = len(items)
        if n == 0:
            return np.zeros(0, dtype=float), 0.0, 0.0

        if not bool(self.cfg.use_ga_weights):
            w = np.ones(n, dtype=float)
            return w, float(w.sum()), float(n)

        w_local = np.array([float(it.get("w_local", 1.0)) for it in items], dtype=float)
        w_local = np.where(np.isfinite(w_local), w_local, 0.0)
        w_local = np.maximum(w_local, 0.0)

        gamma = _get_gamma(items)
        if gamma > 0.0:
            t_key = _get_time_key(items)
            t = np.array([float(it.get(t_key, 0.0)) for it in items], dtype=float)
            t = np.where(np.isfinite(t), t, 0.0)
            t_max = float(np.max(t)) if t.size > 0 else 0.0
            dt = np.maximum(0.0, t_max - t)
            w_time = np.exp(-gamma * dt)
        else:
            w_time = np.ones(n, dtype=float)

        w = w_local * w_time
        w = np.where(np.isfinite(w), w, 0.0)
        w = np.maximum(w, 0.0)

        sum_w = float(w.sum())
        if sum_w <= 0.0:
            # fall back to uniform if weights are degenerate
            w = np.ones(n, dtype=float)
            sum_w = float(w.sum())

        n_eff = _effective_n(w)
        if n_eff <= 0.0:
            n_eff = float(n)

        return w, sum_w, float(n_eff)

    # -----------------------
    # Group parsing
    # -----------------------
    def _parse_groups_arg(
        self,
        items: Sequence[Dict[str, Any]],
        groups: Optional[Sequence[int]],
        full_len: int,
    ) -> Tuple[np.ndarray, Optional[List[int]]]:
        """
        Returns:
          - g_vec: per-item group vector (len == len(items))
          - group_ids: optional list of allowed group IDs (or None)

        Supported `groups` inputs:
          - None
          - per-item vector (len == full_len or len == len(items))
          - small list of group ids (e.g. [0,1])
        """
        n = len(items)
        # default: read from items
        g_vec = np.array([int(it.get("a", 0)) for it in items], dtype=int)
        group_ids: Optional[List[int]] = None

        if groups is None:
            return g_vec, group_ids

        try:
            g_list = [int(x) for x in list(groups)]
        except Exception:
            g_list = []
        if len(g_list) == 0:
            return g_vec, group_ids

        # Case: per-item group labels for full buffer -> slice to window
        if len(g_list) == full_len:
            g_tail = g_list[-n:]
            g_vec = np.array(g_tail, dtype=int)
            return g_vec, group_ids

        # Case: per-item labels already windowed
        if len(g_list) == n:
            g_vec = np.array(g_list, dtype=int)
            return g_vec, group_ids

        # Otherwise: treat as group-id list
        group_ids = g_list
        return g_vec, group_ids

    # -----------------------
    # Discrete (bucket) audit
    # -----------------------
    def _audit_discrete(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        g_vec: np.ndarray,
        group_ids: Optional[List[int]],
        residual_key: str,
        w: np.ndarray,
        sum_w: float,
        n_eff: float,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        n = len(items)
        if n == 0 or sum_w <= 0.0:
            return {
                "eps_uncond": 0.0,
                "eps_cond_group_bucket": 0.0,
                "pmin_group_bucket": 0.0,
                "eps_uncond_group": 0.0,
                "eps_cond_group": 0.0,
                "pmin_group": 0.0,
                "beta": 0.0,
                "beta_group": 0.0,
                "top_violations": [],
                "n_eff": float(n_eff),
                "sum_w": float(sum_w),
                "total_samples": 0,
            }

        scores = np.array([float(it.get("p_hat", 0.5)) for it in items], dtype=float)
        r = np.array([float(it.get(residual_key, 0.0)) for it in items], dtype=float)

        # Decide which groups to include
        obs_groups = sorted(set(int(x) for x in g_vec.tolist()))
        if group_ids is not None:
            uniq_groups = [int(g) for g in group_ids if int(g) in obs_groups]
            if not uniq_groups:
                uniq_groups = obs_groups
        else:
            uniq_groups = obs_groups

        n_groups = len(uniq_groups)

        # Bucket intervals per group (equal-mass buckets on scores within each group)
        n_b = max(1, int(cfg.buckets_per_group))
        bucket_edges: Dict[int, List[Tuple[float, float]]] = {}
        for g in uniq_groups:
            mask = (g_vec == int(g))
            vals = scores[mask].tolist()
            if len(vals) >= 2:
                edges = equal_mass_buckets(vals, n_b)  # list[(lo,hi)]
                if len(edges) == 0:
                    edges = [(float(min(vals)), float(max(vals)))]
                bucket_edges[g] = [(float(lo), float(hi)) for (lo, hi) in edges]
            elif len(vals) == 1:
                v = float(vals[0])
                bucket_edges[g] = [(v, v)]
            else:
                # group has no samples in window; skip by setting a dummy interval
                bucket_edges[g] = [(0.0, 1.0)]

        # Optional structural feature bins (global)
        struct_edges: Dict[str, List[Tuple[float, float]]] = {}
        if cfg.include_struct:
            n_sb = max(1, int(cfg.struct_bins))
            struct_feats = ["degree_u", "degree_v", "time_since_last"]
            for feat in struct_feats:
                vals = []
                for it in items:
                    x = it.get("x", {}) or {}
                    vals.append(float(x.get(feat, 0.0)))
                if len(vals) >= 2:
                    e = equal_mass_buckets(vals, n_sb)
                    struct_edges[feat] = [(float(lo), float(hi)) for (lo, hi) in e]
                elif len(vals) == 1:
                    v = float(vals[0])
                    struct_edges[feat] = [(v, v)]
                else:
                    struct_edges[feat] = [(0.0, 0.0)]

        # GA-weighted sums
        sum_w_slice: Dict[Tuple[Any, ...], float] = defaultdict(float)
        sum_w_r_slice: Dict[Tuple[Any, ...], float] = defaultdict(float)

        gb_keys: List[Tuple[Any, ...]] = []
        g_keys: List[Tuple[Any, ...]] = []

        for i, it in enumerate(items):
            g = int(g_vec[i])
            wi = float(w[i])
            ri = float(r[i])

            # Ignore groups outside allowed list (if provided)
            if group_ids is not None and g not in uniq_groups:
                continue

            if cfg.include_group_bucket:
                # bucket within that group's bucket intervals
                b = int(bucket_index(bucket_edges[g], float(scores[i])))
                k_gb = ("gb", g, b)
                if k_gb not in sum_w_slice:
                    gb_keys.append(k_gb)
                sum_w_slice[k_gb] += wi
                sum_w_r_slice[k_gb] += wi * ri

                if cfg.include_struct and struct_edges:
                    x = it.get("x", {}) or {}
                    for feat, edges in struct_edges.items():
                        xv = float(x.get(feat, 0.0))
                        sb = int(bucket_index(edges, xv))
                        k = ("gbS", g, b, feat, sb)
                        sum_w_slice[k] += wi
                        sum_w_r_slice[k] += wi * ri

            if cfg.include_group_only:
                k_g = ("g", g)
                if k_g not in sum_w_slice:
                    g_keys.append(k_g)
                sum_w_slice[k_g] += wi
                sum_w_r_slice[k_g] += wi * ri

        keys_all = list(sum_w_slice.keys())

        # -----------------------------------------------------------------
        # Slice masses under GA weights
        # -----------------------------------------------------------------
        def _mass(k: Tuple[Any, ...]) -> float:
            sw = float(sum_w_slice.get(k, 0.0))
            if sw <= 0.0 or sum_w <= 0.0:
                return 0.0
            return sw / float(sum_w)

        # Helper: whether a slice should participate in eps/pmin (min-mass filter)
        min_mass = float(getattr(cfg, "min_mass", 0.0) or 0.0)

        def _keep(k: Tuple[Any, ...]) -> bool:
            if min_mass <= 0.0:
                return True
            return _mass(k) >= min_mass

        # -----------------------------------------------------------------
        # ε values
        # -----------------------------------------------------------------
        # Unconditional: max_k |E_w[ 1{k} r ]|.
        # IMPORTANT: apply the same min-mass filter to avoid "pmin" being
        # dominated by vanishing-mass slices that we otherwise ignore.
        eps_uncond = 0.0
        for k in keys_all:
            if not _keep(k):
                continue
            mu_uncond = sum_w_r_slice[k] / sum_w
            eps_uncond = max(eps_uncond, abs(mu_uncond))

        eps_cond_gb = 0.0
        for k in gb_keys:
            if not _keep(k):
                continue
            denom = sum_w_slice[k]
            if denom > 0:
                eps_cond_gb = max(eps_cond_gb, abs(sum_w_r_slice[k] / denom))

        eps_uncond_g = 0.0
        eps_cond_g = 0.0
        for k in g_keys:
            if not _keep(k):
                continue
            mu_uncond = sum_w_r_slice[k] / sum_w
            eps_uncond_g = max(eps_uncond_g, abs(mu_uncond))
            denom = sum_w_slice[k]
            if denom > 0:
                eps_cond_g = max(eps_cond_g, abs(sum_w_r_slice[k] / denom))

        # -----------------------------------------------------------------
        # p_min values
        # -----------------------------------------------------------------
        # p_min must be consistent with the min-mass filter used for ε.
        # Otherwise bound = (ε+β)/p_min can explode even when the tiny-mass
        # slices are excluded from ε (this was the root cause of the boundCal
        # blow-ups you observed).
        pmin_gb = 0.0
        if gb_keys:
            masses = [_mass(k) for k in gb_keys if _keep(k) and sum_w_slice[k] > 0.0]
            pmin_gb = float(min(masses)) if masses else 0.0

        pmin_g = 0.0
        if g_keys:
            masses = [_mass(k) for k in g_keys if _keep(k) and sum_w_slice[k] > 0.0]
            pmin_g = float(min(masses)) if masses else 0.0

        # NOTE: eps_uncond already applies min-mass filter above.

        H = max(1, len(keys_all))
        beta = math.sqrt(max(0.0, math.log((H + 1.0) / max(1e-12, cfg.delta))) / max(1.0, n_eff))

        H_g = max(1, n_groups)
        beta_g = math.sqrt(max(0.0, math.log((H_g + 1.0) / max(1e-12, cfg.delta))) / max(1.0, n_eff))

        # Top violations for debugging
        viols = []
        for k in keys_all:
            if not _keep(k):
                continue
            mass = _mass(k)
            mu_uncond = sum_w_r_slice[k] / sum_w
            denom = sum_w_slice[k]
            mu_cond = (sum_w_r_slice[k] / denom) if denom > 0 else 0.0
            viols.append(
                {
                    "slice": k,
                    "mass": float(mass),
                    "mu_uncond": float(mu_uncond),
                    "mu_cond": float(mu_cond),
                }
            )
        viols.sort(key=lambda d: abs(d.get("mu_uncond", 0.0)), reverse=True)
        topB = max(1, int(cfg.budget_B))
        viols = viols[:topB]

        return {
            "eps_uncond": float(eps_uncond),
            "eps_cond_group_bucket": float(eps_cond_gb),
            "pmin_group_bucket": float(pmin_gb),
            "eps_uncond_group": float(eps_uncond_g),
            "eps_cond_group": float(eps_cond_g),
            "pmin_group": float(pmin_g),
            "beta": float(beta),
            "beta_group": float(beta_g),
            "top_violations": viols,
            "n_eff": float(n_eff),
            "sum_w": float(sum_w),
            "total_samples": int(n),
        }

    # -----------------------
    # Any-kernel (RFF) auditor
    # -----------------------
    def _rff_init(self, in_dim: int) -> None:
        if self._rff_W is not None and self._rff_b is not None and self._rff_in_dim == int(in_dim):
            return
        self._rff_in_dim = int(in_dim)

        D = int(self.cfg.rff_dim)
        gamma = float(self.cfg.rff_gamma)
        gamma = max(1e-12, gamma)

        # For RBF kernel: k(x,y)=exp(-gamma ||x-y||^2),
        # sample w ~ N(0, 2*gamma I)
        self._rff_W = self.rng.normal(loc=0.0, scale=math.sqrt(2.0 * gamma), size=(in_dim, D))
        self._rff_b = self.rng.uniform(low=0.0, high=2.0 * math.pi, size=(D,))

    def _rff_features(self, items: Sequence[Dict[str, Any]]) -> np.ndarray:
        # Keep features compact & stable
        p_hat = np.array([float(it.get("p_hat", 0.5)) for it in items], dtype=float)
        tau_x = np.array([float(it.get("tau_x", 1.0)) for it in items], dtype=float)

        feats = [p_hat[:, None], tau_x[:, None]]

        if self.cfg.include_struct:
            struct_feats = ["degree_u", "degree_v", "time_since_last"]
            for feat in struct_feats:
                col = []
                for it in items:
                    x = it.get("x", {}) or {}
                    col.append(float(x.get(feat, 0.0)))
                feats.append(np.array(col, dtype=float)[:, None])

        X = np.concatenate(feats, axis=1)
        X = np.where(np.isfinite(X), X, 0.0)
        return X

    def _audit_rkhs(self, X: np.ndarray, r: np.ndarray, *, w: np.ndarray, sum_w: float, n_eff: float) -> Dict[str, Any]:
        if X.size == 0 or len(r) == 0:
            return {"eps": 0.0, "beta": 0.0, "n_eff": float(n_eff)}

        self._rff_init(in_dim=int(X.shape[1]))
        assert self._rff_W is not None and self._rff_b is not None

        Phi = math.sqrt(2.0 / float(self.cfg.rff_dim)) * np.cos(X @ self._rff_W + self._rff_b[None, :])
        wr = (w * r).astype(float)
        m = (wr[:, None] * Phi).sum(axis=0) / float(sum_w)

        eps = float(np.linalg.norm(m, ord=2))

        beta = math.sqrt(float(self.cfg.rff_dim) / max(1.0, n_eff)) * math.sqrt(
            max(0.0, math.log(2.0 / max(1e-12, self.cfg.delta)))
        )
        return {"eps": eps, "beta": float(beta), "n_eff": float(n_eff)}

    # -----
    # Public
    # -----
    def audit(self, dr_buffer: Sequence[Dict[str, Any]], *, groups: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        full_len = len(dr_buffer)

        # Apply window
        if self.cfg.window and self.cfg.window > 0:
            items = list(dr_buffer[-int(self.cfg.window) :])
        else:
            items = list(dr_buffer)

        g_vec, group_ids = self._parse_groups_arg(items, groups, full_len=full_len)

        w, sum_w, n_eff = self._ga_weights(items)

        disc_r0 = self._audit_discrete(
            items,
            g_vec=g_vec,
            group_ids=group_ids,
            residual_key="r0",
            w=w,
            sum_w=sum_w,
            n_eff=n_eff,
        )
        disc_r_delta = self._audit_discrete(
            items,
            g_vec=g_vec,
            group_ids=group_ids,
            residual_key="r_delta",
            w=w,
            sum_w=sum_w,
            n_eff=n_eff,
        )

        # Bounds (match scripts/run_copf.py expectations)
        # If p_min is effectively 0 after filtering, the certificate is undefined.
        pmin_gb_raw = float(disc_r0.get("pmin_group_bucket", 0.0))
        pmin_g_raw = float(disc_r_delta.get("pmin_group", 0.0))

        if not np.isfinite(pmin_gb_raw) or pmin_gb_raw <= 0.0:
            bound_gCal_max = float("nan")
        else:
            bound_gCal_max = (
                float(disc_r0.get("eps_uncond", 0.0)) + float(disc_r0.get("beta", 0.0))
            ) / max(1e-12, pmin_gb_raw)

        if not np.isfinite(pmin_g_raw) or pmin_g_raw <= 0.0:
            bound_gTE_gap = float("nan")
        else:
            bound_gTE_gap = 2.0 * (
                float(disc_r_delta.get("eps_uncond_group", 0.0)) + float(disc_r_delta.get("beta_group", 0.0))
            ) / max(1e-12, pmin_g_raw)

        out: Dict[str, Any] = {
            "disc_r0": disc_r0,
            "disc_r_delta": disc_r_delta,
            "bound_gCal_max": float(bound_gCal_max),
            "bound_gTE_gap": float(bound_gTE_gap),
        }

        # Optional RKHS auditing
        if self.cfg.rff_dim and self.cfg.rff_dim > 0:
            X = self._rff_features(items)
            r0 = np.array([float(it.get("r0", 0.0)) for it in items], dtype=float)
            rd = np.array([float(it.get("r_delta", 0.0)) for it in items], dtype=float)
            out["kernel_r0"] = self._audit_rkhs(X, r0, w=w, sum_w=sum_w, n_eff=n_eff)
            out["kernel_r_delta"] = self._audit_rkhs(X, rd, w=w, sum_w=sum_w, n_eff=n_eff)

        return out


# Backward-compatible alias (some earlier patches used this name)
OIAnyKernelAuditor = ResidualOIAuditor
