"""copf/oi_audit.py

Residual-OI auditing utilities + simple noisy-transfer certificates.

This module is *evaluation/auditing* only: it does not change model scores.

It aligns with:
  - Definition 2 (Residual OI on auditor class H)
  - Theorem 1 / Corollary 1 (noisy transfer to counterfactual gaps)
in the draft PDF.

We implement two practical auditor families:
  (1) Discrete slice auditors: group-only and group×score-bucket (optionally + structural bins)
  (2) Any-kernel auditors: RBF RKHS approximated with Random Fourier Features (RFF)

The returned epsilons are computed on DR plug-in residuals (r0 / r_delta) stored in GraphAwareDR.buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fl_utils.buckets import equal_mass_buckets, bucket_index


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _log1p_nonneg(x: float) -> float:
    return float(np.log1p(max(0.0, float(x))))


@dataclass
class OIAuditConfig:
    # Discrete auditors
    buckets_per_group: int = 10
    min_mass: float = 0.02
    include_group_only: bool = True
    include_group_bucket: bool = True

    # Optional structural auditors (still discrete): group×bucket×(feature-bin)
    struct_features: Tuple[str, ...] = ("degree_u", "degree_v", "time_since_last")
    struct_bins: int = 5
    include_struct: bool = False

    # Any-kernel (RKHS) auditors via RFF
    rff_dim: int = 0
    rff_gamma: float = 1.0
    seed: int = 0

    # How many recent DR items to use for auditing (0 => use full buffer)
    window: int = 50000

    # Just for reporting top violated discrete auditors
    budget_B: int = 50


class ResidualOIAuditor:
    """Compute residual-OI violations on DR plug-in residuals.

    Expected DR buffer item schema (GraphAwareDR.buffer entries):
      - a: group id (protected attribute)
      - p_hat: score/probability in [0,1]
      - r0: DR plug-in residual for arm 0 (gamma_0 - p_hat)
      - r_delta: DR plug-in residual for treatment effect ((gamma_1-gamma_0) - tau_x)
      - x: (optional) dict of local structural features
    """

    def __init__(self, cfg: OIAuditConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.seed))

        # RFF parameters are lazily initialized because we need feature dimension.
        self._rff_W: Optional[np.ndarray] = None
        self._rff_b: Optional[np.ndarray] = None
        self._rff_dim_in: Optional[int] = None

    # ------------------------------
    # Public API
    # ------------------------------
    def audit(self, dr_buffer: Sequence[Dict[str, Any]], groups: Sequence[Any]) -> Dict[str, Any]:
        """Audit both residuals on the current DR buffer."""
        items = self._select_window(dr_buffer)

        out: Dict[str, Any] = {
            "n": int(len(items)),
        }
        if not items:
            return out

        # Discrete auditors
        disc0 = self._audit_discrete(items, residual_key="r0", groups=groups)
        discd = self._audit_discrete(items, residual_key="r_delta", groups=groups)

        out.update({
            "disc_r0": disc0,
            "disc_r_delta": discd,
        })

        # Kernel auditors (optional)
        if int(self.cfg.rff_dim) > 0:
            out["rff_r0"] = self._audit_rff(items, residual_key="r0")
            out["rff_r_delta"] = self._audit_rff(items, residual_key="r_delta")

        # Transfer-style certificates (Corollary 1 form)
        #   gCal <= (eps0 + beta0)/pmin_slice
        #   gTE  <= 2*(epsD + betaD)/pmin_group
        # Here we compute eps from discrete auditors.
        eps0 = float(disc0.get("eps_uncond", 0.0))
        beta0 = float(disc0.get("beta", 0.0))
        pmin_slice = float(disc0.get("pmin_group_bucket", 0.0))
        out["bound_gCal_max"] = self._bound_one_over_pmin(eps0 + beta0, pmin_slice, factor=1.0)

        epsD = float(discd.get("eps_uncond_group", 0.0))
        betaD = float(discd.get("beta_group", 0.0))
        pmin_g = float(discd.get("pmin_group", 0.0))
        out["bound_gTE_gap"] = self._bound_one_over_pmin(epsD + betaD, pmin_g, factor=2.0)

        return out

    # ------------------------------
    # Discrete auditors
    # ------------------------------
    def _audit_discrete(self, items: Sequence[Dict[str, Any]], residual_key: str, groups: Sequence[Any]) -> Dict[str, Any]:
        """Compute discrete residual-OI violations.

        Returns both:
          - eps_uncond: max_h |E[ h * r ]| over chosen auditors (unconditional)
          - eps_cond:   max_h |E[ r | slice ]| over group×bucket slices (conditional)
        and pmins needed for transfer bounds.
        """

        N = int(len(items))
        if N <= 0:
            return {}

        # Build score buckets per group using p_hat distribution in the audited window.
        edges_by_group: Dict[Any, List[Tuple[float, float]]] = {}
        for g in groups:
            g_scores = [
                _safe_float(it.get("p_hat", 0.5), 0.5)
                for it in items
                if it.get("a", None) == g
            ]
            if len(g_scores) < 2:
                continue
            edges_by_group[g] = equal_mass_buckets(
                np.asarray(g_scores, float),
                int(self.cfg.buckets_per_group),
                min_mass=float(self.cfg.min_mass),
            )

        # Optional structural bins (global, not per-group)
        struct_edges: Dict[str, List[Tuple[float, float]]] = {}
        if bool(self.cfg.include_struct) and int(self.cfg.struct_bins) > 1:
            for feat in self.cfg.struct_features:
                vals = []
                for it in items:
                    x = it.get("x", {}) or {}
                    if feat not in x:
                        continue
                    v = _safe_float(x.get(feat, 0.0), 0.0)
                    # log1p on heavy-tailed features
                    if "degree" in feat or "cnt" in feat or "num" in feat:
                        v = _log1p_nonneg(v)
                    if "time" in feat:
                        v = _log1p_nonneg(v)
                    vals.append(v)
                if len(vals) < 2:
                    continue
                struct_edges[feat] = equal_mass_buckets(
                    np.asarray(vals, float),
                    int(self.cfg.struct_bins),
                    min_mass=float(self.cfg.min_mass),
                )

        # Accumulators for auditors: key -> (sum_r, count)
        sum_r: Dict[Tuple[Any, ...], float] = {}
        cnt: Dict[Tuple[Any, ...], int] = {}

        # Track pmins
        cnt_group: Dict[Any, int] = {}
        cnt_gb: Dict[Tuple[Any, int], int] = {}

        for it in items:
            g = it.get("a", None)
            if g is None:
                continue

            p = float(np.clip(_safe_float(it.get("p_hat", 0.5), 0.5), 1e-6, 1.0 - 1e-6))
            r = _safe_float(it.get(residual_key, 0.0), 0.0)

            # group count
            cnt_group[g] = cnt_group.get(g, 0) + 1

            # Group-only auditors: h = 1{A=g}
            if bool(self.cfg.include_group_only):
                k = ("g", g)
                sum_r[k] = sum_r.get(k, 0.0) + r
                cnt[k] = cnt.get(k, 0) + 1

            # Group×bucket auditors: h = 1{A=g, p_hat in I_b}
            if bool(self.cfg.include_group_bucket):
                edges = edges_by_group.get(g)
                if edges is None:
                    continue
                b = int(bucket_index(edges, p))
                k = ("gb", g, b)
                sum_r[k] = sum_r.get(k, 0.0) + r
                cnt[k] = cnt.get(k, 0) + 1
                cnt_gb[(g, b)] = cnt_gb.get((g, b), 0) + 1

                # Optional structural auditors: h = 1{A=g, bucket=b, feat in bin}
                if struct_edges:
                    x = it.get("x", {}) or {}
                    for feat, edges_f in struct_edges.items():
                        v = _safe_float(x.get(feat, 0.0), 0.0)
                        if "degree" in feat or "cnt" in feat or "num" in feat:
                            v = _log1p_nonneg(v)
                        if "time" in feat:
                            v = _log1p_nonneg(v)
                        sb = int(bucket_index(edges_f, float(v)))
                        kf = ("gbf", g, b, feat, sb)
                        sum_r[kf] = sum_r.get(kf, 0.0) + r
                        cnt[kf] = cnt.get(kf, 0) + 1

        # Compute epsilons
        # Unconditional means: E[h*r] ~ sum_r / N
        eps_uncond = 0.0
        top = []
        for k, sr in sum_r.items():
            mu_uncond = float(sr) / float(N)
            abs_mu = abs(mu_uncond)
            if abs_mu > eps_uncond:
                eps_uncond = abs_mu
            top.append((abs_mu, k, mu_uncond, cnt.get(k, 0)))

        top.sort(key=lambda x: x[0], reverse=True)
        top = top[: int(max(0, self.cfg.budget_B))]

        # Conditional means on group×bucket slices: E[r | slice]
        eps_cond_gb = 0.0
        for (g, b), c in cnt_gb.items():
            if c <= 0:
                continue
            sr = sum_r.get(("gb", g, b), 0.0)
            mu_cond = float(sr) / float(c)
            eps_cond_gb = max(eps_cond_gb, abs(mu_cond))

        # pmins
        pmin_g = 0.0
        if cnt_group:
            pmin_g = min(cnt_group.values()) / float(N)

        pmin_gb = 0.0
        if cnt_gb:
            pmin_gb = min(cnt_gb.values()) / float(N)

        # Confidence radius proxy beta ~ sqrt(log|H|/N)
        H_eff = max(2, len(sum_r))
        beta = float(np.sqrt(np.log(float(H_eff)) / float(max(1, N))))

        # Also compute group-only epsilon (needed for gTE bound)
        eps_uncond_group = 0.0
        if bool(self.cfg.include_group_only):
            for g in cnt_group.keys():
                sr = sum_r.get(("g", g), 0.0)
                eps_uncond_group = max(eps_uncond_group, abs(float(sr) / float(N)))

        beta_group = float(np.sqrt(np.log(float(max(2, len(cnt_group)))) / float(max(1, N))))

        return {
            "residual": residual_key,
            "N": N,
            "H_eff": int(H_eff),
            "beta": beta,
            "beta_group": beta_group,

            "eps_uncond": float(eps_uncond),
            "eps_uncond_group": float(eps_uncond_group),
            "eps_cond_group_bucket": float(eps_cond_gb),

            "pmin_group": float(pmin_g),
            "pmin_group_bucket": float(pmin_gb),

            "top_violations": [
                {
                    "abs_Ehr": float(a),
                    "key": tuple(k),
                    "Ehr": float(mu),
                    "count": int(c),
                }
                for (a, k, mu, c) in top
            ],
        }

    # ------------------------------
    # Any-kernel auditors via RFF
    # ------------------------------
    def _audit_rff(self, items: Sequence[Dict[str, Any]], residual_key: str) -> Dict[str, Any]:
        """Approximate RKHS residual-OI via Random Fourier Features.

        We compute the RKHS norm of the residual-weighted mean embedding:
            eps ≈ || (1/N) Σ r_i φ(x_i) ||_2
        where φ are RFF features for an RBF kernel.
        """
        N = int(len(items))
        if N <= 0:
            return {}

        X = self._rff_design_matrix(items)  # [N,d]
        r = np.asarray([_safe_float(it.get(residual_key, 0.0), 0.0) for it in items], float)
        Phi = self._rff_features(X)  # [N,M]

        m = (r.reshape(-1, 1) * Phi).mean(axis=0)
        eps = float(np.linalg.norm(m, ord=2))

        # A crude confidence proxy. (Optional; mainly for logging.)
        beta = float(np.sqrt(float(Phi.shape[1]) / float(max(1, N))))

        return {
            "residual": residual_key,
            "N": N,
            "M": int(Phi.shape[1]),
            "eps_rkhs": eps,
            "beta": beta,
        }

    def _rff_design_matrix(self, items: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Build a continuous feature vector per item for RFF.

        We intentionally keep it simple and robust:
          x = [group_id, p_hat, log1p(degree_u), log1p(degree_v), log1p(time_since_last)]
        Missing structural features are treated as 0.
        """
        rows = []
        for it in items:
            g = _safe_float(it.get("a", 0.0), 0.0)
            p = float(np.clip(_safe_float(it.get("p_hat", 0.5), 0.5), 1e-6, 1.0 - 1e-6))
            x = it.get("x", {}) or {}
            du = _log1p_nonneg(_safe_float(x.get("degree_u", 0.0), 0.0))
            dv = _log1p_nonneg(_safe_float(x.get("degree_v", 0.0), 0.0))
            tsl = _log1p_nonneg(_safe_float(x.get("time_since_last", 0.0), 0.0))
            rows.append([g, p, du, dv, tsl])
        X = np.asarray(rows, float)

        # Standardize to mean 0, std 1 for stable gamma
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd <= 1e-8, 1.0, sd)
        X = (X - mu) / sd
        return X

    def _init_rff(self, d: int) -> None:
        M = int(self.cfg.rff_dim)
        gamma = float(max(1e-12, self.cfg.rff_gamma))

        # For RBF kernel k(x,y) = exp(-gamma ||x-y||^2)
        # RFF uses w ~ N(0, 2*gamma I)
        self._rff_W = self.rng.normal(loc=0.0, scale=np.sqrt(2.0 * gamma), size=(M, d))
        self._rff_b = self.rng.uniform(low=0.0, high=2.0 * np.pi, size=(M,))
        self._rff_dim_in = int(d)

    def _rff_features(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 2
        N, d = X.shape
        M = int(self.cfg.rff_dim)
        if self._rff_W is None or self._rff_b is None or self._rff_dim_in != int(d):
            self._init_rff(d)

        W = self._rff_W  # [M,d]
        b = self._rff_b  # [M]
        Z = X @ W.T + b.reshape(1, -1)
        Phi = np.sqrt(2.0 / float(M)) * np.cos(Z)
        return Phi

    # ------------------------------
    # Helpers
    # ------------------------------
    def _select_window(self, dr_buffer: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not dr_buffer:
            return []
        w = int(self.cfg.window)
        if w <= 0 or w >= len(dr_buffer):
            return list(dr_buffer)
        return list(dr_buffer[-w:])

    @staticmethod
    def _bound_one_over_pmin(x: float, pmin: float, factor: float = 1.0) -> float:
        pmin = float(max(1e-9, pmin))
        return float(factor * float(x) / pmin)
