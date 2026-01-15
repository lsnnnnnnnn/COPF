"""copf/coverage.py

Coverage-driven exploration utilities.

This implements a lightweight variant of the paper's "coverage-driven exploration"
idea: if some *slices* (group × score-bucket) receive too little exposure over a
recent horizon, we temporarily (i) increase the global exploration rate epsilon
and (ii) restrict the exploration pool to candidates that lie in those
under-covered slices.

Why this design?
----------------
The codebase uses doubly-robust (DR) plug-ins that rely on *logged propensities*.
To keep DR unbiased, the logged propensities must match the *actual* sampling
policy. The decision module (copf/decision.py) supports a single exploration
probability at the slate level (epsilon) and an exploration pool (explore_mask).

We therefore implement a policy that remains exactly of that form:

  - With prob 1-epsilon_t: exploit (TopK / PL-TopK)
  - With prob epsilon_t: explore uniformly within explore_pool_t

Where explore_pool_t contains candidates whose slice is under-covered.

The "coverage" we track here is the *share of exposures* that each slice has
received (not the candidate mass of the slice). This is the quantity that the
learner can directly influence via the exposure policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from collections import defaultdict, deque
import numpy as np

from fl_utils.buckets import equal_mass_buckets, bucket_index


SliceKey = Tuple[Any, int]  # (group, bucket_id)


@dataclass
class CoverageExplorerConfig:
    enabled: bool = False

    # Slice definition: group × score bucket.
    buckets_per_group: int = 10

    # How many recent scores per group to use when recomputing equal-mass buckets.
    score_window: int = 40000
    update_buckets_every: int = 200
    bucket_min_mass: float = 0.02

    # Coverage tracking (EMA over rounds).
    ema_window_rounds: int = 200
    warmup_rounds: int = 50

    # Target minimum *exposure share* per slice and epsilon schedule.
    ptar: float = 0.02
    c: float = 1.0
    eps_min: float = 0.0
    eps_max: float = 0.5

    # If True: only boost epsilon when an under-covered slice appears in the current
    # candidate list. If False: compute deficit over all tracked slices.
    only_consider_present_slices: bool = True


class CoverageDrivenExplorer:
    """Stateful coverage tracker + exploration planner."""

    def __init__(self, cfg: CoverageExplorerConfig, groups: List[Any]):
        self.cfg = cfg
        self.groups = list(groups)

        # Per-group score history for bucket recomputation.
        self._score_hist: Dict[Any, deque] = {
            g: deque(maxlen=int(max(1, cfg.score_window))) for g in self.groups
        }

        # Per-group bucket intervals.
        self._buckets: Dict[Any, List[Tuple[float, float]]] = {g: [(0.0, 1.0)] for g in self.groups}

        # EMA exposure counters.
        self._expo_ema: Dict[SliceKey, float] = defaultdict(float)
        self._expo_total_ema: float = 0.0

        # Derived
        self._alpha = 1.0
        if int(cfg.ema_window_rounds) > 1:
            # standard EMA mapping
            self._alpha = 2.0 / (float(cfg.ema_window_rounds) + 1.0)

        self.round: int = 0

    # ----------------
    # Slice assignment
    # ----------------
    def _ensure_group(self, g: Any) -> None:
        if g in self._score_hist:
            return
        self._score_hist[g] = deque(maxlen=int(max(1, self.cfg.score_window)))
        self._buckets[g] = [(0.0, 1.0)]
        self.groups.append(g)

    def slice_key(self, g: Any, score: float) -> SliceKey:
        self._ensure_group(g)
        b = int(bucket_index(self._buckets.get(g, [(0.0, 1.0)]), float(score)))
        return (g, b)

    # -----------------
    # Bucket maintenance
    # -----------------
    def _recompute_buckets(self) -> None:
        B = int(max(1, self.cfg.buckets_per_group))
        min_mass = float(max(0.0, self.cfg.bucket_min_mass))
        for g in list(self._score_hist.keys()):
            scores = np.asarray(list(self._score_hist[g]), dtype=float)
            if scores.size < 50:
                # not enough signal; keep default bucket
                continue
            self._buckets[g] = equal_mass_buckets(scores, B, min_mass=min_mass)

    # -----------------
    # Coverage statistics
    # -----------------
    def exposure_share(self, key: SliceKey) -> float:
        tot = float(self._expo_total_ema)
        if tot <= 1e-12:
            return 0.0
        return float(self._expo_ema.get(key, 0.0) / tot)

    def _update_exposure_ema(self, expo_this: Dict[SliceKey, float], total_this: float) -> None:
        a = float(self._alpha)
        self._expo_total_ema = (1.0 - a) * float(self._expo_total_ema) + a * float(total_this)
        # decay all known slices a bit; then add current counts.
        for k in list(self._expo_ema.keys()):
            self._expo_ema[k] = (1.0 - a) * float(self._expo_ema[k])
        for k, v in expo_this.items():
            self._expo_ema[k] = float(self._expo_ema.get(k, 0.0) + a * float(v))

    # -----------------
    # Public API
    # -----------------
    def plan(
        self,
        cands: List[Dict[str, Any]],
        *,
        base_epsilon: float,
        topk: int,
    ) -> Tuple[float, Optional[np.ndarray], Dict[str, float]]:
        """Return (epsilon_eff, explore_mask, diagnostics).

        explore_mask marks candidates that belong to under-covered slices.
        """
        diag: Dict[str, float] = {
            "cov_enabled": 1.0 if bool(self.cfg.enabled) else 0.0,
            "cov_eps": float(base_epsilon),
            "cov_pool_frac": 0.0,
            "cov_min_share_present": 0.0,
            "cov_max_deficit_present": 0.0,
        }

        if not bool(self.cfg.enabled) or not cands:
            return float(base_epsilon), None, diag

        n = len(cands)
        scores = np.array([float(c.get("p_hat", 0.5)) for c in cands], dtype=float)
        groups = np.array([c.get("a", 0) for c in cands], dtype=object)

        # Assign slice keys
        keys: List[SliceKey] = []
        for g, sc in zip(groups.tolist(), scores.tolist()):
            keys.append(self.slice_key(g, sc))

        # Determine under-covered slices *present in this candidate list*
        present = sorted(set(keys))
        shares = [self.exposure_share(k) for k in present]
        min_share = float(min(shares)) if shares else 0.0

        ptar = float(max(0.0, self.cfg.ptar))
        deficits = [max(0.0, ptar - s) for s in shares]
        max_def = float(max(deficits)) if deficits else 0.0

        diag["cov_min_share_present"] = float(min_share)
        diag["cov_max_deficit_present"] = float(max_def)

        # If we're still warming up, do not bias exploration.
        if int(self.round) < int(self.cfg.warmup_rounds):
            return float(base_epsilon), None, diag

        # Build exploration pool mask: candidates whose slice share < ptar.
        under_present = {k for k, s in zip(present, shares) if s < ptar}
        if not under_present:
            return float(base_epsilon), None, diag

        explore_mask = np.array([k in under_present for k in keys], dtype=bool)
        pool_frac = float(np.mean(explore_mask.astype(float)))
        diag["cov_pool_frac"] = float(pool_frac)

        # Epsilon schedule (paper-inspired): eps_t = max(eps_min, c*(ptar - p_hat(s))+).
        # We implement the *worst-case* deficit among the present slices.
        eps_min = float(max(0.0, self.cfg.eps_min))
        eps_max = float(max(0.0, self.cfg.eps_max))
        c = float(max(0.0, self.cfg.c))
        eps_boost = float(c * max_def)
        eps_eff = float(max(float(base_epsilon), eps_min, eps_boost))
        eps_eff = float(min(max(0.0, eps_eff), min(1.0, eps_max)))

        diag["cov_eps"] = float(eps_eff)
        return eps_eff, explore_mask, diag

    def update(self, cands: List[Dict[str, Any]], d_list: List[int]) -> None:
        """Update the tracker with realized exposures for this round."""
        if not bool(self.cfg.enabled) or not cands:
            self.round += 1
            return

        # Update score history
        for c in cands:
            g = c.get("a", 0)
            sc = float(c.get("p_hat", 0.5))
            self._ensure_group(g)
            self._score_hist[g].append(sc)

        # Update bucket edges periodically
        if int(self.round) % int(max(1, self.cfg.update_buckets_every)) == 0:
            self._recompute_buckets()

        # Exposure counts for this round
        expo_this: Dict[SliceKey, float] = defaultdict(float)
        total_this = 0.0
        for c, d in zip(cands, d_list):
            if int(d) != 1:
                continue
            g = c.get("a", 0)
            sc = float(c.get("p_hat", 0.5))
            k = self.slice_key(g, sc)
            expo_this[k] += 1.0
            total_this += 1.0

        self._update_exposure_ema(expo_this, total_this)

        self.round += 1
