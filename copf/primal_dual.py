"""fairlink/copf/primal_dual.py

Online primal-dual coordination utilities (COPF).

This repository's main runner (`scripts/run_copf.py`) is intentionally
non-differentiable end-to-end (it uses a slate decision rule + propensity logging
+ DR plug-ins). Therefore, the "primal" side we can reliably and safely control
without touching the base model is the *decision policy*.

This file implements the *dual* updates described in:
  - Algorithm 1 (Online Primal–Dual COPF)
  - Section 15.3 (Robust λ tuning with a PI controller)
  - Section 15.5 (Hierarchical priorities: TE-parity first, then gCal)

We provide a small controller that:
  - maintains λ_TE and λ_Cal (optionally λ_Min)
  - updates λ with a PI rule against a ramping target ρ_t
  - converts λ_TE into a per-group *logit offset* that biases the exposure policy
    toward under-benefited groups (low τ_s).

Important: This is a practical instantiation. It does not claim to implement the
exact gradient step in Algorithm 1 line 7 (which would require differentiable
surrogates for the slate policy and the underlying model). It *does* implement
Algorithm 1's dual dynamics and the paper's stability heuristics, and it makes
λ directly influence the deployed policy as requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def _sigmoid(z: float) -> float:
    # stable sigmoid
    z = float(z)
    if z >= 0:
        ez = np.exp(-z)
        return float(1.0 / (1.0 + ez))
    ez = np.exp(z)
    return float(ez / (1.0 + ez))


def _logit(p: float, eps: float = 1e-6) -> float:
    p = float(p)
    p = float(min(1.0 - eps, max(eps, p)))
    return float(np.log(p / (1.0 - p)))


@dataclass
class PIDualConfig:
    # enable/disable controller
    enabled: bool = False

    # PI gains (Section 15.3)
    gamma_p: float = 0.2
    gamma_i: float = 0.02

    # ramp schedule ρ_t,k (Section 15.3)
    ramp_start: int = 0
    ramp_end: int = 0

    # final targets
    te_target: float = 0.05
    cal_target: float = 0.05

    # optional start targets (if None, use final)
    te_target_start: Optional[float] = None
    cal_target_start: Optional[float] = None

    # confidence-radius softening g̃ = ĝ/(1+β) (Section 15.5)
    soften_by_beta: bool = True

    # hierarchical priorities (Section 15.5)
    hierarchical: bool = True
    te_margin: float = 0.0  # allow small slack when gating cal updates

    # clamp λ
    lambda_max: float = 50.0

    # --- policy effect (how λ influences decision scores) ---
    # Per-group logit shift is:
    #   shift_g = offset_scale * λ_TE * (τ_target - τ_g)
    offset_scale: float = 1.0
    logit_clip: float = 2.0


class PIDualController:
    """Dual controller producing a policy bias for TE-parity.

    State:
      - lambda_te, lambda_cal
      - integral errors for PI updates

    API:
      - update(...) -> diagnostics dict
      - group_logit_offsets(tau_by_group) -> {g: logit_shift}
      - apply_te_bias_to_candidates(cands, offsets) -> modifies c['p_hat']
    """

    def __init__(self, cfg: PIDualConfig, groups: Sequence[int]):
        self.cfg = cfg
        self.groups: List[int] = sorted({int(g) for g in groups}) if groups else [0]

        self.lambda_te: float = 0.0
        self.lambda_cal: float = 0.0

        self._int_te: float = 0.0
        self._int_cal: float = 0.0

        # last targets for logging
        self._rho_te: float = float(cfg.te_target)
        self._rho_cal: float = float(cfg.cal_target)

        # gating status
        self._cal_gate_open: bool = True

    # ---------------------
    # Ramp schedule helpers
    # ---------------------
    def _rho(self, t: int, total_T: int, start: Optional[float], final: float) -> float:
        """Linear ramp from start to final between [ramp_start, ramp_end]."""
        final = float(final)
        if start is None:
            start = final
        start = float(start)

        rs = int(self.cfg.ramp_start)
        re = int(self.cfg.ramp_end)
        if rs <= 0 or re <= 0 or re <= rs:
            return final

        # Clamp t to [0, total_T] for safety
        t = int(max(0, min(int(t), int(total_T))))

        if t <= rs:
            return start
        if t >= re:
            return final
        frac = float(t - rs) / float(re - rs)
        return float((1.0 - frac) * start + frac * final)

    # ---------------------
    # Public: dual update
    # ---------------------
    def update(
        self,
        *,
        step: int,
        total_T: int,
        gte_gap: float,
        gcal_max: float,
        beta_te: float = 0.0,
        beta_cal: float = 0.0,
    ) -> Dict[str, float]:
        """PI update for λ_TE and λ_Cal.

        Args:
          step: current round index (1..T)
          total_T: total rounds
          gte_gap: estimated gTE gap (max_{s1,s2} |τ_s1-τ_s2|)
          gcal_max: estimated max gCal over kept slices
          beta_te: confidence radius for TE (used to soften)
          beta_cal: confidence radius for gCal (used to soften)

        Returns:
          diagnostics dict with lambdas, targets, violations, gate.
        """
        if not bool(self.cfg.enabled):
            return {
                "pd_enabled": 0.0,
                "lambda_te": float(self.lambda_te),
                "lambda_cal": float(self.lambda_cal),
                "rho_te": float(self._rho_te),
                "rho_cal": float(self._rho_cal),
                "v_te": 0.0,
                "v_cal": 0.0,
                "cal_gate_open": 1.0 if self._cal_gate_open else 0.0,
            }

        # ramp targets
        self._rho_te = self._rho(step, total_T, self.cfg.te_target_start, self.cfg.te_target)
        self._rho_cal = self._rho(step, total_T, self.cfg.cal_target_start, self.cfg.cal_target)

        # confidence-radius softening (Section 15.5)
        if bool(self.cfg.soften_by_beta):
            gte_soft = float(gte_gap) / float(1.0 + max(0.0, float(beta_te)))
            gcal_soft = float(gcal_max) / float(1.0 + max(0.0, float(beta_cal)))
        else:
            gte_soft = float(gte_gap)
            gcal_soft = float(gcal_max)

        # PI errors v_t,k = F̃_cf - ρ_t,k (Section 15.3)
        v_te = float(gte_soft - self._rho_te)
        v_cal = float(gcal_soft - self._rho_cal)

        # Update λ_TE always
        self._int_te += v_te
        self.lambda_te = float(
            max(
                0.0,
                self.lambda_te + float(self.cfg.gamma_p) * v_te + float(self.cfg.gamma_i) * self._int_te,
            )
        )
        self.lambda_te = _clamp(self.lambda_te, 0.0, float(self.cfg.lambda_max))

        # Hierarchical gating: only tighten calibration after TE is within target
        if bool(self.cfg.hierarchical):
            self._cal_gate_open = bool(gte_soft <= (self._rho_te + float(self.cfg.te_margin)))
        else:
            self._cal_gate_open = True

        if self._cal_gate_open:
            self._int_cal += v_cal
            self.lambda_cal = float(
                max(
                    0.0,
                    self.lambda_cal + float(self.cfg.gamma_p) * v_cal + float(self.cfg.gamma_i) * self._int_cal,
                )
            )
            self.lambda_cal = _clamp(self.lambda_cal, 0.0, float(self.cfg.lambda_max))
        else:
            # Optional: do not integrate when gate is closed to avoid windup
            # (keeps behaviour stable in practice)
            pass

        return {
            "pd_enabled": 1.0,
            "lambda_te": float(self.lambda_te),
            "lambda_cal": float(self.lambda_cal),
            "rho_te": float(self._rho_te),
            "rho_cal": float(self._rho_cal),
            "gte_soft": float(gte_soft),
            "gcal_soft": float(gcal_soft),
            "v_te": float(v_te),
            "v_cal": float(v_cal),
            "cal_gate_open": 1.0 if self._cal_gate_open else 0.0,
        }

    # ---------------------
    # Public: policy effect
    # ---------------------
    def group_logit_offsets(self, tau_by_group: Dict[int, Tuple[float, float]]) -> Dict[int, float]:
        """Convert λ_TE and τ_s estimates into per-group logit shifts.

        The intuition is to boost exposure probability for groups whose estimated
        τ_s (benefit) is below a global target τ̄.

        Args:
          tau_by_group: {group: (tau_hat, n_eff)}

        Returns:
          offsets: {group: logit_shift}
        """
        if not bool(self.cfg.enabled) or self.lambda_te <= 0.0:
            return {int(g): 0.0 for g in self.groups}

        # compute τ_target as the mean over available groups with finite τ
        taus = []
        for g in self.groups:
            te = float(tau_by_group.get(int(g), (0.0, 0.0))[0])
            if np.isfinite(te):
                taus.append(te)
        tau_target = float(np.mean(taus)) if taus else 0.0

        offsets: Dict[int, float] = {}
        for g in self.groups:
            te_g = float(tau_by_group.get(int(g), (0.0, 0.0))[0])
            if not np.isfinite(te_g):
                te_g = tau_target
            shift = float(self.cfg.offset_scale) * float(self.lambda_te) * float(tau_target - te_g)
            shift = _clamp(shift, -float(self.cfg.logit_clip), float(self.cfg.logit_clip))
            offsets[int(g)] = float(shift)
        return offsets

    def apply_te_bias_to_candidates(
        self,
        cands: List[Dict[str, float]],
        offsets: Dict[int, float],
        *,
        store_debug: bool = True,
    ) -> Dict[str, float]:
        """Apply per-group logit offsets to each candidate's p_hat in-place."""
        if not cands:
            return {
                "pd_shift_abs_mean": 0.0,
                "pd_shift_abs_max": 0.0,
                "pd_shift_nonzero": 0.0,
            }

        shifts = []
        nonzero = 0
        for c in cands:
            g = int(c.get("a", 0))
            shift = float(offsets.get(g, 0.0))
            shifts.append(abs(shift))
            if abs(shift) > 1e-12:
                nonzero += 1

            p = float(c.get("p_hat", 0.5))
            if store_debug:
                c.setdefault("p_hat_base", p)
                c["pd_logit_shift"] = shift

            z = _logit(p)
            p_new = _sigmoid(z + shift)
            c["p_hat"] = float(p_new)

        return {
            "pd_shift_abs_mean": float(np.mean(shifts)) if shifts else 0.0,
            "pd_shift_abs_max": float(np.max(shifts)) if shifts else 0.0,
            "pd_shift_nonzero": float(nonzero) / float(len(cands)) if cands else 0.0,
        }
