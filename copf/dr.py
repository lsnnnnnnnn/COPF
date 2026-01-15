"""fairlink/copf/dr.py

Graph-aware doubly robust (GA-DR) estimation utilities for COPF.

This module maintains an online buffer of candidate dyads and provides
self-normalized GA-weighted DR plug-in estimates needed for:
  - within-group counterfactual calibration (gCal)
  - treatment-effect parity (gTE)
  - residual-OI auditing certificates (bounds)

Paper-alignment notes
---------------------
The paper defines counterfactual residuals (Section 6):
  r(0) = Y(0) - p̂
  r(Δ) = (Y(1) - Y(0)) - τ(x)

In our implementation:
  - We estimate Y(a) with the DR score Γ̃(a) (Definition 1 in the paper).
  - Therefore we use r0_hat := Γ̃(0) - p̂ and r_delta_hat := (Γ̃(1)-Γ̃(0)) - τ(x).

IMPORTANT (TE-parity certificate fix; "method 1")
-------------------------------------------------
To make the residual-OI certificate for TE-parity match Lemma 2,
τ(x) must *not* vary across groups in a way that would survive the
(s1,s2) difference.

This file supports two modes for τ(x):
  - tau_mode='plugin': τ(x) = μ̂1(x) - μ̂0(x) (x-dependent).
  - tau_mode='global': τ(x) = τ_global (a global constant / EMA over time).

For TE-parity certificates, prefer tau_mode='global'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from collections import deque
import numpy as np


def winsorize01(p: float, eps: float) -> float:
    """Clip probability to [eps, 1-eps]."""
    p = float(p)
    if not np.isfinite(p):
        return 0.5
    eps = float(max(0.0, min(0.49, eps)))
    return float(min(1.0 - eps, max(eps, p)))


def ratio_stabilize(p: float, clip: float) -> float:
    """A tiny stabilizer for propensities to avoid 1/p blowing up."""
    p = winsorize01(p, clip)
    # shrink towards 0.5 slightly
    return float(0.98 * p + 0.01)


@dataclass
class DRConfig:
    clip: float = 0.05
    self_normalized: bool = True
    decay_gamma: float = 0.1
    max_buffer: int = 200000
    ratio_stab: bool = True
    winsor_eps: float = 0.02
    min_eff_samples: int = 100

    # --- TE target τ(x) options ---
    tau_mode: str = "global"  # ['global', 'plugin']
    tau_ema_alpha: float = 0.02
    tau_init: float = 0.0


class GraphAwareDR:
    """Graph-aware DR plug-in estimator with an online buffer."""

    def __init__(self, config: DRConfig, cross_fitter: Optional[Any] = None):
        self.cfg = config
        self.cross_fitter = cross_fitter

        self.buffer = deque(maxlen=int(config.max_buffer))

        # global τ target (used when tau_mode='global')
        self.tau_global: float = float(config.tau_init)
        self._tau_valid: bool = True  # even tau_init=0.0 is a valid constant
        self.last_tau_batch: float = float("nan")

    # -----------------
    # GA-weight helpers
    # -----------------
    def _weights(self, cond_mask: np.ndarray) -> Tuple[np.ndarray, float, float]:
        items = list(self.buffer)
        if len(items) == 0:
            return np.zeros(0, dtype=float), 0.0, 0.0

        w_local = np.array([float(it.get("w_local", 1.0)) for it in items], dtype=float)
        w_local = np.where(np.isfinite(w_local), w_local, 0.0)
        w_local = np.maximum(w_local, 0.0)

        gamma = float(getattr(self.cfg, "decay_gamma", 0.0) or 0.0)
        if gamma > 0.0:
            t = np.array([float(it.get("t_round", it.get("t", 0.0))) for it in items], dtype=float)
            t = np.where(np.isfinite(t), t, 0.0)
            t_max = float(np.max(t)) if t.size > 0 else 0.0
            dt = np.maximum(0.0, t_max - t)
            w_time = np.exp(-gamma * dt)
        else:
            w_time = np.ones(len(items), dtype=float)

        w = w_local * w_time
        w = np.where(np.isfinite(w), w, 0.0)
        w = np.maximum(w, 0.0)

        if cond_mask is not None:
            w = w * cond_mask.astype(float)

        sum_w = float(w.sum())
        if sum_w <= 0.0:
            return w, 0.0, 0.0

        # Kish effective sample size
        s2 = float(np.sum(w * w))
        n_eff = (sum_w * sum_w) / s2 if s2 > 0.0 else float(len(items))
        return w, sum_w, float(n_eff)

    # -----------------
    # Nuisance update
    # -----------------
    def update_nuisance(self, c: Dict[str, Any]) -> Tuple[float, float]:
        """Return (mu0, mu1) for candidate dict c."""
        if self.cross_fitter is None:
            # default fallback (do NOT use in real experiments)
            p = winsorize01(float(c.get("p_hat", 0.5)), self.cfg.winsor_eps)
            return float(p), float(p)

        # ------------------------------
        # Cross-fitter API compatibility
        # ------------------------------
        # Different versions of the repo expose nuisance predictions under
        # different method names. Prefer predict_mu() if available; otherwise
        # fall back to predict_outcome()/get_predictions_batch().
        t = c.get("t_round", c.get("t", None))
        try:
            t_int = int(t) if t is not None else None
        except Exception:
            t_int = None

        if hasattr(self.cross_fitter, "predict_mu"):
            mu0, mu1 = self.cross_fitter.predict_mu(c, t=t_int)
            return float(mu0), float(mu1)

        if hasattr(self.cross_fitter, "predict_outcome"):
            mu0 = self.cross_fitter.predict_outcome(c, arm=0, t=t_int)
            mu1 = self.cross_fitter.predict_outcome(c, arm=1, t=t_int)
            return float(mu0), float(mu1)

        if hasattr(self.cross_fitter, "get_predictions_batch"):
            preds = self.cross_fitter.get_predictions_batch([c], t=t_int)
            mu0 = float(np.asarray(preds.get("mu_0"))[0])
            mu1 = float(np.asarray(preds.get("mu_1"))[0])
            return float(mu0), float(mu1)

        # Last-resort fallback
        p = winsorize01(float(c.get("p_hat", 0.5)), self.cfg.winsor_eps)
        return float(p), float(p)

    # -----------------
    # Online ingest
    # -----------------
    def ingest(self, batch: Sequence[Dict[str, Any]]) -> None:
        """Ingest a single round's candidate set into the DR buffer."""
        if not batch:
            return

        tau_mode = str(getattr(self.cfg, "tau_mode", "global") or "global").strip().lower()
        if tau_mode not in {"global", "plugin"}:
            tau_mode = "global"

        # For tau_mode='global', τ(x) is a constant at this round.
        tau_for_resid = float(self.tau_global) if self._tau_valid else float(self.cfg.tau_init)

        # Track a batch-level TE estimate to update τ_global (EMA) if enabled
        batch_w_sum = 0.0
        batch_te_num = 0.0

        for c in batch:
            # Required fields
            a = int(c.get("a", 0))
            p_hat = winsorize01(float(c.get("p_hat", 0.5)), self.cfg.winsor_eps)

            d = int(c.get("d", 0))
            e_hat = float(c.get("e_hat", 0.5))
            y = float(c.get("y", 0.0))

            w_local = float(c.get("w_local", 1.0))
            if not np.isfinite(w_local) or w_local < 0.0:
                w_local = 0.0

            t_round = c.get("t_round", c.get("t", 0))
            try:
                t_round = int(t_round)
            except Exception:
                t_round = 0

            # Nuisance predictions
            mu0, mu1 = self.update_nuisance(c)
            mu0 = float(mu0); mu1 = float(mu1)
            if not np.isfinite(mu0):
                mu0 = p_hat
            if not np.isfinite(mu1):
                mu1 = p_hat

            # Propensity clipping
            clip = float(self.cfg.clip)
            e1 = winsorize01(e_hat, clip)
            e0 = winsorize01(1.0 - e_hat, clip)
            if bool(self.cfg.ratio_stab):
                e1 = ratio_stabilize(e1, clip)
                e0 = ratio_stabilize(e0, clip)

            # DR scores
            ind1 = 1.0 if d == 1 else 0.0
            ind0 = 1.0 if d == 0 else 0.0
            gamma1 = mu1 - (ind1 / e1) * (mu1 - y)
            gamma0 = mu0 - (ind0 / e0) * (mu0 - y)

            # τ(x) target
            if tau_mode == "global":
                tau_x = tau_for_resid
            else:
                # x-dependent plug-in target
                tau_x = float(mu1 - mu0)

            # counterfactual residuals (DR plug-ins)
            r0 = float(gamma0 - p_hat)
            r_delta = float((gamma1 - gamma0) - tau_x)

            item = {
                "a": a,
                "p_hat": p_hat,
                "d": d,
                "e_hat": float(e_hat),
                "y": y,
                "mu0": mu0,
                "mu1": mu1,
                "gamma0": float(gamma0),
                "gamma1": float(gamma1),
                "r0": r0,
                "r_delta": r_delta,
                "tau_x": float(tau_x),
                "w_local": w_local,
                # Provide GA time-decay gamma explicitly so that external
                # auditors (ResidualOIAuditor) can use the *same* GA weights
                # as GraphAwareDR when computing certificates.
                "ga_gamma": float(getattr(self.cfg, "decay_gamma", 0.0) or 0.0),
                "decay_gamma": float(getattr(self.cfg, "decay_gamma", 0.0) or 0.0),
                "t_round": t_round,
                "t": float(c.get("t", t_round)),
                "x": c.get("x", {}),
            }
            self.buffer.append(item)

            # batch TE estimate for τ_global update
            batch_w_sum += w_local
            batch_te_num += w_local * float(gamma1 - gamma0)

        # Update τ_global (EMA over rounds) if tau_mode='global'
        if tau_mode == "global" and batch_w_sum > 0.0 and np.isfinite(batch_te_num):
            tau_batch = float(batch_te_num / batch_w_sum)
            self.last_tau_batch = tau_batch

            alpha = float(getattr(self.cfg, "tau_ema_alpha", 0.0) or 0.0)
            alpha = float(max(0.0, min(1.0, alpha)))
            if alpha <= 0.0:
                # constant τ_global
                pass
            else:
                # EMA update
                self.tau_global = float((1.0 - alpha) * float(self.tau_global) + alpha * tau_batch)
                self._tau_valid = True

    # -----------------
    # Plug-in estimates
    # -----------------
    def estimate_EY(self, arm: int, cond: Optional[Tuple[str, int]] = None) -> Tuple[float, float]:
        """Estimate E[Y(arm) | cond] with GA weights.

        Returns (estimate, n_eff).

        cond options:
          - None
          - ("a", group_id) condition on protected group
        """
        items = list(self.buffer)
        if len(items) == 0:
            return 0.0, 0.0

        if cond is None:
            cond_mask = np.ones(len(items), dtype=float)
        else:
            key, val = cond
            if key == "a":
                cond_mask = np.array([1.0 if int(it.get("a", 0)) == int(val) else 0.0 for it in items], dtype=float)
            else:
                cond_mask = np.ones(len(items), dtype=float)

        w, sum_w, n_eff = self._weights(cond_mask)
        if sum_w <= 0.0 or n_eff < float(self.cfg.min_eff_samples):
            return 0.0, float(n_eff)

        if int(arm) == 1:
            g = np.array([float(it.get("gamma1", 0.0)) for it in items], dtype=float)
        else:
            g = np.array([float(it.get("gamma0", 0.0)) for it in items], dtype=float)

        g = np.where(np.isfinite(g), g, 0.0)
        est = float(np.sum(w * g) / sum_w) if bool(self.cfg.self_normalized) else float(np.mean(g))
        return est, float(n_eff)

    def estimate_TE_by_group(self, group_key: str = "a") -> Dict[int, Tuple[float, float]]:
        """Return {group: (TE_hat, n_eff)} using GA-DR plug-ins."""
        items = list(self.buffer)
        if not items:
            return {}

        groups = sorted({int(it.get(group_key, 0)) for it in items})
        out: Dict[int, Tuple[float, float]] = {}
        for g in groups:
            y1, n1 = self.estimate_EY(arm=1, cond=(group_key, g))
            y0, n0 = self.estimate_EY(arm=0, cond=(group_key, g))
            te = float(y1 - y0)
            n_eff = float(min(n1, n0))
            out[int(g)] = (te, n_eff)
        return out

    def diagnostics(self) -> Dict[str, float]:
        """Lightweight diagnostics for logging."""
        tau_mode = str(getattr(self.cfg, "tau_mode", "global") or "global").strip().lower()
        return {
            "tau_mode_global": 1.0 if tau_mode == "global" else 0.0,
            "tau_global": float(self.tau_global),
            "tau_last_batch": float(self.last_tau_batch) if np.isfinite(self.last_tau_batch) else float("nan"),
            "dr_buffer_len": float(len(self.buffer)),
        }











