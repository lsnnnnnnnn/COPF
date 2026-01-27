from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid."""
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
    enabled: bool = False
    gamma_p: float = 0.2
    gamma_i: float = 0.02
    ramp_start: int = 0
    ramp_end: int = 0
    te_target: float = 0.05
    cal_target: float = 0.05
    # gMin is a hinge >=0, so the natural target is 0.
    min_target: float = 0.0
    te_target_start: Optional[float] = None
    cal_target_start: Optional[float] = None
    min_target_start: Optional[float] = None
    soften_by_beta: bool = True
    hierarchical: bool = True
    te_margin: float = 0.0
    min_margin: float = 0.0
    lambda_max: float = 50.0
    offset_scale: float = 1.0
    min_offset_scale: float = 1.0
    logit_clip: float = 2.0


class PIDualController:
    def __init__(self, cfg: PIDualConfig, groups: Sequence[int]):
        self.cfg = cfg
        self.groups: List[int] = sorted({int(g) for g in groups}) if groups else [0]

        self.lambda_te: float = 0.0
        self.lambda_cal: float = 0.0
        self.lambda_min: float = 0.0

        self._int_te: float = 0.0
        self._int_cal: float = 0.0
        self._int_min: float = 0.0

        # last targets for logging
        self._rho_te: float = float(cfg.te_target)
        self._rho_cal: float = float(cfg.cal_target)
        self._rho_min: float = float(cfg.min_target)

        # gating status for calibration
        self._cal_gate_open: bool = True

    
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

        t = int(max(0, min(int(t), int(total_T))))

        if t <= rs:
            return start
        if t >= re:
            return final
        frac = float(t - rs) / float(re - rs)
        return float((1.0 - frac) * start + frac * final)

    
    def update(
        self,
        *,
        step: int,
        total_T: int,
        gte_gap: float,
        gcal_max: float,
        gmin: float = 0.0,
        beta_te: float = 0.0,
        beta_cal: float = 0.0,
        beta_min: float = 0.0,
    ) -> Dict[str, float]:
        
        if not bool(self.cfg.enabled):
            return {
                "pd_enabled": 0.0,
                "lambda_te": float(self.lambda_te),
                "lambda_cal": float(self.lambda_cal),
                "lambda_min": float(self.lambda_min),
                "rho_te": float(self._rho_te),
                "rho_cal": float(self._rho_cal),
                "rho_min": float(self._rho_min),
                "v_te": 0.0,
                "v_cal": 0.0,
                "v_min": 0.0,
                "cal_gate_open": 1.0 if self._cal_gate_open else 0.0,
            }

        # ramp targets
        self._rho_te = self._rho(step, total_T, self.cfg.te_target_start, self.cfg.te_target)
        self._rho_cal = self._rho(step, total_T, self.cfg.cal_target_start, self.cfg.cal_target)
        self._rho_min = self._rho(step, total_T, self.cfg.min_target_start, self.cfg.min_target)

        # confidence-radius softening (Section 15.5)
        if bool(self.cfg.soften_by_beta):
            gte_soft = float(gte_gap) / float(1.0 + max(0.0, float(beta_te)))
            gcal_soft = float(gcal_max) / float(1.0 + max(0.0, float(beta_cal)))
            gmin_soft = float(gmin) / float(1.0 + max(0.0, float(beta_min)))
        else:
            gte_soft = float(gte_gap)
            gcal_soft = float(gcal_max)
            gmin_soft = float(gmin)

        # PI errors v_t,k = F̃_cf - ρ_t,k (Section 15.3)
        v_te = float(gte_soft - self._rho_te)
        v_cal = float(gcal_soft - self._rho_cal)
        v_min = float(gmin_soft - self._rho_min)

        # Update λ_TE always
        self._int_te += v_te
        self.lambda_te = float(
            max(
                0.0,
                self.lambda_te + float(self.cfg.gamma_p) * v_te + float(self.cfg.gamma_i) * self._int_te,
            )
        )
        self.lambda_te = _clamp(self.lambda_te, 0.0, float(self.cfg.lambda_max))

        # Update λ_Min always
        self._int_min += v_min
        self.lambda_min = float(
            max(
                0.0,
                self.lambda_min + float(self.cfg.gamma_p) * v_min + float(self.cfg.gamma_i) * self._int_min,
            )
        )
        self.lambda_min = _clamp(self.lambda_min, 0.0, float(self.cfg.lambda_max))

        # Hierarchical gating: only tighten calibration when TE and Min are satisfied
        if bool(self.cfg.hierarchical):
            ok_te = bool(gte_soft <= (self._rho_te + float(self.cfg.te_margin)))
            ok_min = bool(gmin_soft <= (self._rho_min + float(self.cfg.min_margin)))
            self._cal_gate_open = bool(ok_te and ok_min)
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

        return {
            "pd_enabled": 1.0,
            "lambda_te": float(self.lambda_te),
            "lambda_cal": float(self.lambda_cal),
            "lambda_min": float(self.lambda_min),
            "rho_te": float(self._rho_te),
            "rho_cal": float(self._rho_cal),
            "rho_min": float(self._rho_min),
            "gte_soft": float(gte_soft),
            "gcal_soft": float(gcal_soft),
            "gmin_soft": float(gmin_soft),
            "v_te": float(v_te),
            "v_cal": float(v_cal),
            "v_min": float(v_min),
            "cal_gate_open": 1.0 if self._cal_gate_open else 0.0,
        }

    
    def group_logit_offsets(
        self,
        tau_by_group: Dict[int, Tuple[float, float]],
        *,
        tau_min: float = 0.0,
    ) -> Dict[int, float]:
        """Convert (λ_TE, λ_Min) + τ_s estimates into per-group logit shifts.

        Args:
          tau_by_group: {group: (tau_hat, n_eff)}
          tau_min: minimum-effect threshold (τ_min). If <=0, min-effect offsets are disabled.

        Returns:
          offsets: {group: logit_shift}
        """
        # Fast path: controller disabled
        if not bool(self.cfg.enabled):
            return {int(g): 0.0 for g in self.groups}

        # Compute τ̄ over configured groups (ignore missing/NaN)
        taus = []
        for g in self.groups:
            te_g = float(tau_by_group.get(int(g), (0.0, 0.0))[0])
            if np.isfinite(te_g):
                taus.append(te_g)
        tau_target = float(np.mean(taus)) if taus else 0.0

        tau_min = float(tau_min)
        use_min = (tau_min > 0.0) and (self.lambda_min > 0.0)
        use_te = (self.lambda_te > 0.0)

        offsets: Dict[int, float] = {}
        for g in self.groups:
            te_g = float(tau_by_group.get(int(g), (tau_target, 0.0))[0])
            if not np.isfinite(te_g):
                te_g = tau_target

            shift_te = 0.0
            if use_te:
                shift_te = float(self.cfg.offset_scale) * float(self.lambda_te) * float(tau_target - te_g)

            shift_min = 0.0
            if use_min:
                deficit = max(0.0, tau_min - te_g)
                shift_min = float(self.cfg.min_offset_scale) * float(self.lambda_min) * float(deficit)

            shift = float(shift_te + shift_min)
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
