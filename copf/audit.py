"""
Multicalibration with Active Auditor Selection
Implements Section 9 and Algorithm 2 Line 9
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
from fl_utils.buckets import equal_mass_buckets, bucket_index


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, float(p)))
    return float(np.log(p / (1.0 - p)))


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(z))))


@dataclass
class AuditorConfig:
    buckets_per_group: int = 10
    isotonic: bool = True  # 先保留开关；目前 offsets 更新用的是 logit-shift
    budget_B: int = 64
    step: float = 0.25
    min_mass: float = 0.02

    # Safety: cap logit-offset magnitude to avoid numerical blow-ups when
    # using aggressive steps (e.g., λCal-scaled updates).
    offset_clip: float = 5.0

    # Learning-rate normalization.
    #
    # We want the calibrator to keep adapting in nonstationary online settings
    # (temporal graphs drift). A purely cumulative 1/sqrt(count) schedule tends
    # to freeze the calibrator after a few thousand samples, even if fairness
    # violations re-emerge later.
    #
    # We therefore normalize by the *current batch size* only, with a reference
    # count `lr_ref_n`: for buckets with >= lr_ref_n residual samples in the
    # current update, we downscale the learning rate as 1/sqrt(n/lr_ref_n);
    # otherwise we keep lr at the base step. This keeps lr <= base_step and
    # prevents both explosion (small n) and freezing (large cumulative n).
    lr_ref_n: int = 200


class ActiveAuditorSet:
    """Maintains active set H^act of most violated auditors"""

    def __init__(self, budget: int = 64):
        self.budget = int(budget)
        self.active_auditors: Set[Tuple[Any, int]] = set()
        self.violation_scores: Dict[Tuple[Any, int], float] = {}

    def update(self, violations: Dict[Tuple[Any, int], float]) -> None:
        """Update active set with top-B violated auditors"""
        for key, viol in violations.items():
            self.violation_scores[key] = abs(float(viol))

        sorted_auditors = sorted(
            self.violation_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        self.active_auditors = {auditor for auditor, _ in sorted_auditors[: self.budget]}

    def is_active(self, key: Tuple[Any, int]) -> bool:
        return key in self.active_auditors


class MultiCalibrator:
    """
    Multi-calibration with counterfactual residuals
    Uses residual OI: r^(0) = Y^(0) - p_hat
    """

    def __init__(self, cfg: AuditorConfig, active_set: Optional[ActiveAuditorSet] = None):
        self.cfg = cfg
        self.active_set = active_set or ActiveAuditorSet(budget=cfg.budget_B)

        # Calibration offsets for each (group, bucket) in logit space
        self.offsets: Dict[Tuple[Any, int], float] = {}

        # Bucket edges for each group: g -> [(lo, hi), ...]
        self.edges_by_group: Dict[Any, List[Tuple[float, float]]] = {}

        # Sample counts for adaptive step size
        self.counts: Dict[Tuple[Any, int], int] = {}

        self.total_samples = 0

    def conf_radius(self) -> float:
        """β_t ~ sqrt(log |H| / t)"""
        if self.total_samples <= 0:
            return 1.0
        H_size = max(2, len(self.edges_by_group) * max(1, self.cfg.buckets_per_group))
        return float(np.sqrt(np.log(H_size) / float(self.total_samples)))

    def apply(self, cands: List[Dict[str, Any]]) -> None:
        """Apply calibration offsets to candidates (before decision)."""
        for c in cands:
            p = float(c.get("p_hat", 0.5))
            g = c.get("a", None)

            edges = self.edges_by_group.get(g)
            if edges is None:
                continue

            bidx = bucket_index(edges, p)
            key = (g, bidx)

            # Only apply if active
            if not self.active_set.is_active(key):
                continue

            offset = float(self.offsets.get(key, 0.0))
            if abs(offset) <= 1e-12:
                continue

            z = _logit(p) + offset
            p_cal = _sigmoid(z)

            c["p_hat_orig"] = p
            c["p_hat"] = float(np.clip(p_cal, 0.001, 0.999))

    def calibrate(
        self,
        cands: List[Dict[str, Any]],
        dr,
        prefer_arm: int = 0,
        dr_clip: float = 0.02,
        step_scale: float = 1.0,
    ) -> None:
        """
        Update calibration using counterfactual residuals.
        Default uses arm=0 residuals: r0 = Y^(0) - p_hat
        """
        if not cands or dr is None or not getattr(dr, "buffer", None):
            return

        self.total_samples += len(cands)

        # Group candidates by (group, bucket) based on current p_hat distribution
        buckets_data = self._group_by_buckets(cands)

        lookup = self._dr_lookup(dr)  # dict: (u,v,t)->buffer_item

        violations: Dict[Tuple[Any, int], float] = {}

        for (g, bidx), bucket_cands in buckets_data.items():
            if len(bucket_cands) < 5:
                continue

            residuals: List[float] = []
            for c in bucket_cands:
                key3 = (c.get("u"), c.get("v"), c.get("t"))
                it = lookup.get(key3)
                if it is None:
                    continue

                p = float(c.get("p_hat", 0.5))

                # prefer_arm 保留接口：现在默认用 gamma_0 / r0
                if prefer_arm == 1:
                    # treated residual: Y^(1) - p_hat (如果你 dr.buffer 里有 r1/gamma_1)
                    y1_est = float(it.get("gamma_1", it.get("mu1_hat", 0.5)))
                    r = y1_est - p
                else:
                    # control residual
                    # 优先用 r0（更直接），否则 gamma_0 - p
                    r = float(it.get("r0", float(it.get("gamma_0", it.get("mu0_hat", 0.5))) - p))

                if dr_clip is not None:
                    r = float(np.clip(r, -float(dr_clip), float(dr_clip)))

                residuals.append(r)

            if not residuals:
                continue

            avg_r = float(np.mean(residuals))
            violations[(g, bidx)] = avg_r

            key = (g, bidx)
            # Track cumulative samples for diagnostics only.
            self.counts[key] = int(self.counts.get(key, 0) + len(residuals))

            # Learning rate: batch-size normalized (see AuditorConfig.lr_ref_n).
            base_step = float(self.cfg.step) * float(max(0.0, step_scale))
            ref_n = float(max(1, int(getattr(self.cfg, "lr_ref_n", 200))))
            # lr <= base_step, and decays as 1/sqrt(n/ref_n) for large buckets.
            lr = float(base_step / np.sqrt(max(1.0, float(len(residuals)) / ref_n)))
            new_off = float(self.offsets.get(key, 0.0) + lr * avg_r)
            if float(self.cfg.offset_clip) > 0:
                new_off = float(np.clip(new_off, -float(self.cfg.offset_clip), float(self.cfg.offset_clip)))
            self.offsets[key] = float(new_off)

        if violations:
            self.active_set.update(violations)

    def _group_by_buckets(self, cands: List[Dict[str, Any]]) -> Dict[Tuple[Any, int], List[Dict[str, Any]]]:
        """Group candidates by (group, score_bucket) using equal-mass buckets per group."""
        result: Dict[Tuple[Any, int], List[Dict[str, Any]]] = {}

        groups = list({c.get("a") for c in cands})
        for g in groups:
            g_cands = [c for c in cands if c.get("a") == g]
            if len(g_cands) < 2:
                continue

            scores = np.array([float(c.get("p_hat", 0.5)) for c in g_cands], dtype=float)
            edges = equal_mass_buckets(scores, self.cfg.buckets_per_group, min_mass=self.cfg.min_mass)
            self.edges_by_group[g] = edges

            for c in g_cands:
                p = float(c.get("p_hat", 0.5))
                bidx = bucket_index(edges, p)
                key = (g, bidx)
                result.setdefault(key, []).append(c)

        return result

    def compute_violations(self, cands: List[Dict[str, Any]], dr, prefer_arm: int = 0, dr_clip: float = 0.02) -> Dict[Tuple[Any, int], float]:
        """Compute residual violations for all auditors (no parameter update)."""
        if not cands or dr is None or not getattr(dr, "buffer", None):
            return {}

        buckets_data = self._group_by_buckets(cands)
        lookup = self._dr_lookup(dr)

        violations: Dict[Tuple[Any, int], float] = {}
        for (g, bidx), bucket_cands in buckets_data.items():
            if len(bucket_cands) < 5:
                continue

            residuals: List[float] = []
            for c in bucket_cands:
                it = lookup.get((c.get("u"), c.get("v"), c.get("t")))
                if it is None:
                    continue

                p = float(c.get("p_hat", 0.5))
                if prefer_arm == 1:
                    y1_est = float(it.get("gamma_1", it.get("mu1_hat", 0.5)))
                    r = y1_est - p
                else:
                    r = float(it.get("r0", float(it.get("gamma_0", it.get("mu0_hat", 0.5))) - p))

                if dr_clip is not None:
                    r = float(np.clip(r, -float(dr_clip), float(dr_clip)))
                residuals.append(r)

            if residuals:
                violations[(g, bidx)] = float(np.mean(residuals))

        return violations

    def _dr_lookup(self, dr) -> Dict[Tuple[Any, Any, Any], Dict[str, Any]]:
        """Build a lookup map from DR buffer items: (u,v,t)->item."""
        m: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
        buf = getattr(dr, "buffer", None)
        if not buf:
            return m
        for it in buf:
            key = (it.get("u"), it.get("v"), it.get("t"))
            m[key] = it
        return m

    def update_from_dr(
        self,
        dr,
        max_items: int = 5000,
        dr_clip: float = 0.02,
        step_scale: float = 1.0,
    ) -> None:
        """
        Alternative update route: update buckets + offsets directly from DR buffer residuals.
        Useful when you want calibrator driven purely by DR history rather than per-step cands.
        """
        buf = getattr(dr, "buffer", None)
        if not buf:
            return

        # NOTE: dr.buffer is a collections.deque in our GA-DR implementation.
        # deque supports integer indexing but NOT slicing. Convert to list first.
        seq = list(buf)
        items = seq[-max_items:] if max_items is not None else seq
        if not items:
            return

        self.total_samples += len(items)

        # refresh edges per group using p_hat distribution in buffer
        groups = list({it.get("a") for it in items})
        for g in groups:
            g_items = [it for it in items if it.get("a") == g]
            if len(g_items) < 2:
                continue
            scores = np.array([float(it.get("p_hat", 0.5)) for it in g_items], dtype=float)
            edges = equal_mass_buckets(scores, self.cfg.buckets_per_group, min_mass=self.cfg.min_mass)
            self.edges_by_group[g] = edges

        residuals_by_key: Dict[Tuple[Any, int], List[float]] = {}
        for it in items:
            g = it.get("a")
            edges = self.edges_by_group.get(g)
            if edges is None:
                continue

            p = float(it.get("p_hat", 0.5))
            bidx = bucket_index(edges, p)
            key = (g, bidx)

            r0 = float(it.get("r0", float(it.get("gamma_0", it.get("mu0_hat", 0.5))) - p))
            if dr_clip is not None:
                r0 = float(np.clip(r0, -float(dr_clip), float(dr_clip)))

            residuals_by_key.setdefault(key, []).append(r0)

        violations: Dict[Tuple[Any, int], float] = {}
        for key, rs in residuals_by_key.items():
            if len(rs) < 5:
                continue
            avg_r = float(np.mean(rs))
            violations[key] = avg_r

            # Track cumulative samples for diagnostics only.
            self.counts[key] = int(self.counts.get(key, 0) + len(rs))

            # Learning rate: batch-size normalized (see AuditorConfig.lr_ref_n).
            base_step = float(self.cfg.step) * float(max(0.0, step_scale))
            ref_n = float(max(1, int(getattr(self.cfg, "lr_ref_n", 200))))
            lr = float(base_step / np.sqrt(max(1.0, float(len(rs)) / ref_n)))
            new_off = float(self.offsets.get(key, 0.0) + lr * avg_r)
            if float(self.cfg.offset_clip) > 0:
                new_off = float(np.clip(new_off, -float(self.cfg.offset_clip), float(self.cfg.offset_clip)))
            self.offsets[key] = float(new_off)

        if violations:
            self.active_set.update(violations)

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "n_active": len(self.active_set.active_auditors),
            "n_groups": len(self.edges_by_group),
            "beta_t": self.conf_radius(),
            "total_samples": self.total_samples,
            "active_auditors": list(self.active_set.active_auditors),
        }