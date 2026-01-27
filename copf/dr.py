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

   
    tau_mode: str = "global"  # ['global', 'plugin']
    tau_ema_alpha: float = 0.02
    tau_init: float = 0.0
    estimator: str = "dr"


class GraphAwareDR:
    """Graph-aware DR plug-in estimator with an online buffer."""

    def __init__(self, config: DRConfig, cross_fitter: Optional[Any] = None):
        self.cfg = config
        self.cross_fitter = cross_fitter

        self.buffer = deque(maxlen=int(config.max_buffer))

        
        self.tau_global: float = float(config.tau_init)
        self._tau_valid: bool = True  
        self.last_tau_batch: float = float("nan")

    
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

    
    def update_nuisance(self, c: Dict[str, Any]) -> Tuple[float, float]:
        """Return (mu0, mu1) for candidate dict c."""
        if self.cross_fitter is None:
            # default fallback (do NOT use in real experiments)
            p = winsorize01(float(c.get("p_hat", 0.5)), self.cfg.winsor_eps)
            return float(p), float(p)

        
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

        
        p = winsorize01(float(c.get("p_hat", 0.5)), self.cfg.winsor_eps)
        return float(p), float(p)

    
    def ingest(self, batch: Sequence[Dict[str, Any]]) -> None:
        if not batch:
            return

        estimator = str(getattr(self.cfg, "estimator", "dr") or "dr").strip().lower()
        # allow a couple common aliases
        if estimator in {"doubly_robust", "doubly-robust"}:
            estimator = "dr"
        if estimator in {"ipw", "ips", "iw"}:
            estimator = "ips"
        if estimator in {"observed", "observed_only", "observed-only", "naive"}:
            estimator = "obs"
        if estimator not in {"dr", "ips", "obs"}:
            estimator = "dr"

        tau_mode = str(getattr(self.cfg, "tau_mode", "global") or "global").strip().lower()
        if tau_mode not in {"global", "plugin"}:
            tau_mode = "global"

        
        tau_for_resid = float(self.tau_global) if self._tau_valid else float(self.cfg.tau_init)

        batch_w_sum = 0.0
        batch_te_num = 0.0
        batch_w1 = 0.0
        batch_y1 = 0.0
        batch_w0 = 0.0
        batch_y0 = 0.0

        for c in batch:
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

            
            mu0, mu1 = self.update_nuisance(c)
            mu0 = float(mu0); mu1 = float(mu1)
            if not np.isfinite(mu0):
                mu0 = p_hat
            if not np.isfinite(mu1):
                mu1 = p_hat

            
            clip = float(self.cfg.clip)
            e1_raw = float(e_hat)
            e0_raw = float(1.0 - e_hat)
            e1 = winsorize01(e1_raw, clip)
            e0 = winsorize01(e0_raw, clip)
            if bool(self.cfg.ratio_stab):
                e1 = ratio_stabilize(e1, clip)
                e0 = ratio_stabilize(e0, clip)

            ind1 = 1.0 if d == 1 else 0.0
            ind0 = 1.0 if d == 0 else 0.0
            if estimator == "dr":
                gamma1 = mu1 - (ind1 / e1) * (mu1 - y)
                gamma0 = mu0 - (ind0 / e0) * (mu0 - y)
            elif estimator == "ips":
                gamma1 = (ind1 / e1) * y
                gamma0 = (ind0 / e0) * y
            else:
                gamma1 = y
                gamma0 = y

            if tau_mode == "global":
                tau_x = tau_for_resid
            else:
                tau_x = float(mu1 - mu0)

            r0 = float(gamma0 - p_hat)
            if estimator == "obs":
                r_delta = 0.0
            else:
                r_delta = float((gamma1 - gamma0) - tau_x)

            item = {
                "a": a,
                "p_hat": p_hat,
                "d": d,
                "e_hat": float(e_hat),
                "e1": float(e1),
                "e0": float(e0),
                "e1_raw": float(e1_raw),
                "e0_raw": float(e0_raw),
                "y": y,
                "mu0": mu0,
                "mu1": mu1,
                "gamma0": float(gamma0),
                "gamma1": float(gamma1),
                "r0": r0,
                "r_delta": r_delta,
                "tau_x": float(tau_x),
                "w_local": w_local,
                "ga_gamma": float(getattr(self.cfg, "decay_gamma", 0.0) or 0.0),
                "decay_gamma": float(getattr(self.cfg, "decay_gamma", 0.0) or 0.0),
                "t_round": t_round,
                "t": float(c.get("t", t_round)),
                "x": c.get("x", {}),
            }
            self.buffer.append(item)

            batch_w_sum += w_local
            if estimator == "obs":
                batch_w1 += w_local * ind1
                batch_y1 += w_local * ind1 * float(y)
                batch_w0 += w_local * ind0
                batch_y0 += w_local * ind0 * float(y)
            else:
                batch_te_num += w_local * float(gamma1 - gamma0)

        if tau_mode == "global" and batch_w_sum > 0.0:
            if estimator == "obs":
                if batch_w1 > 0.0 and batch_w0 > 0.0:
                    tau_batch = float(batch_y1 / batch_w1) - float(batch_y0 / batch_w0)
                else:
                    tau_batch = 0.0
            else:
                if not np.isfinite(batch_te_num):
                    return
                tau_batch = float(batch_te_num / batch_w_sum)
            self.last_tau_batch = tau_batch

            alpha = float(getattr(self.cfg, "tau_ema_alpha", 0.0) or 0.0)
            alpha = float(max(0.0, min(1.0, alpha)))
            if alpha <= 0.0:
                pass
            else:
                self.tau_global = float((1.0 - alpha) * float(self.tau_global) + alpha * tau_batch)
                self._tau_valid = True

    
    def estimate_EY(self, arm: int, cond: Optional[Tuple[str, int]] = None) -> Tuple[float, float]:
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

        
        estimator = str(getattr(self.cfg, "estimator", "dr") or "dr").strip().lower()
        if estimator in {"observed", "observed_only", "observed-only", "naive"}:
            estimator = "obs"
        if estimator == "obs":
            a_arm = 1 if int(arm) == 1 else 0
            d_mask = np.array([1.0 if int(it.get("d", 0)) == a_arm else 0.0 for it in items], dtype=float)
            cond_mask = cond_mask * d_mask

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
        tau_mode = str(getattr(self.cfg, "tau_mode", "global") or "global").strip().lower()
        return {
            "tau_mode_global": 1.0 if tau_mode == "global" else 0.0,
            "tau_global": float(self.tau_global),
            "tau_last_batch": float(self.last_tau_batch) if np.isfinite(self.last_tau_batch) else float("nan"),
            "dr_buffer_len": float(len(self.buffer)),
        }











