# fairness.py

"""Counterfactual fairness metrics for COPF.

This module implements the paper's fairness family G' (Section 7) in the
practical "Scheme A" form used by this repo:

  - Within-group counterfactual calibration (gCal)
  - Treatment-effect parity (gTE)
  - Minimum-effect guard (gMin)
  - Baseline risk reporting (gRisk)

Important implementation note (alignment with Residual-OI certificates)
----------------------------------------------------------------------
The residual-OI certificate in `copf/oi_audit.py` is derived for residuals
  r(0) = Y(0) - p̂
  r(Δ) = (Y(1)-Y(0)) - τ(x)

Therefore, for calibration we use the *residual form*:
  gCal_{s,I} = | E[Y(0) | A=s, p̂∈I] - E[p̂ | A=s, p̂∈I] |
             = | E[r(0) | A=s, p̂∈I] |

This makes `bound_gCal_max` a direct certificate for `gCal_max` (when both
are computed on the same slice family and with the same GA weights).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .dr import GraphAwareDR
from fl_utils.buckets import equal_mass_buckets


_SCORE_KEYS = ("p_hat", "p", "score", "prob", "s")


def _get_score(c: Dict[str, Any], default: float = 0.5) -> float:
    for k in _SCORE_KEYS:
        if k in c and c[k] is not None:
            try:
                return float(c[k])
            except Exception:
                continue
    return float(default)


def _pav_isotonic(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators algorithm (simple isotonic regression)."""
    y = np.asarray(y, float).copy()
    w = np.asarray(w, float).copy()
    if y.size == 0:
        return y

    blocks = [(float(y[i]), float(w[i]), 1) for i in range(y.size)]  # (avg, wsum, len)
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            y_sum = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            w_sum = blocks[i][1] + blocks[i + 1][1]
            L = blocks[i][2] + blocks[i + 1][2]
            avg = y_sum / (w_sum if w_sum > 0 else max(1.0, float(L)))
            blocks[i] = (float(avg), float(w_sum), int(L))
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1

    out: List[float] = []
    for avg, _, L in blocks:
        out.extend([float(avg)] * int(L))
    return np.asarray(out, float)


def _ga_weights_for_items(items: List[Dict[str, Any]], fallback_gamma: float = 0.0) -> np.ndarray:
    """Return GA weights for a list of DR items.

    This matches `ResidualOIAuditor._ga_weights` semantics:
      w = w_local * exp(-gamma * (t_max - t_i))

    where gamma is read from the items if available (ga_gamma / decay_gamma),
    otherwise from `fallback_gamma`.
    """
    n = len(items)
    if n == 0:
        return np.zeros(0, dtype=float)

    w_local = np.array([float(it.get("w_local", 1.0)) for it in items], dtype=float)
    w_local = np.where(np.isfinite(w_local), w_local, 0.0)
    w_local = np.maximum(w_local, 0.0)

    # Prefer per-item gamma (set by GraphAwareDR.ingest), fall back to provided.
    g = items[0].get("ga_gamma", items[0].get("decay_gamma", fallback_gamma))
    try:
        gamma = float(g)
    except Exception:
        gamma = float(fallback_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        gamma = 0.0

    if gamma > 0.0:
        t_key = "t_round" if "t_round" in items[0] else "t"
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

    # If degenerate, fall back to uniform.
    if float(np.sum(w)) <= 0.0:
        w = np.ones(n, dtype=float)
    return w


def gCal(
    batch: List[Dict[str, Any]],
    dr: GraphAwareDR,
    groups: List[Any],
    buckets_per_group: int = 10,
    isotonic: bool = False,
    min_mass: float = 0.02,
    min_per_bucket: int = 0,
    arm_for_cal: int = 0,
    ga_mass_filter: bool = True,
) -> Dict[Tuple[Any, int], float]:
    """Within-group counterfactual calibration gaps.

    Scheme A (certificate-aligned):
        gCal_{s,I} = |E[Y^(a) | A=s, p_hat in I] - E[p_hat | A=s, p_hat in I]|

    Args:
      batch: list of DR buffer items (as produced by GraphAwareDR.ingest).
      dr: GraphAwareDR instance (used for config gates like min_eff_samples).
      groups: list of group IDs.
      buckets_per_group: number of equal-mass buckets per group.
      isotonic: if True, apply isotonic smoothing across buckets.
      min_mass: minimum GA mass for a slice to be considered (certificate alignment).
      min_per_bucket: minimum raw count for a bucket.
      arm_for_cal: 0 or 1 (default 0; calibration is defined on Y(0) in the paper).
      ga_mass_filter: if True, drop slices whose GA mass < min_mass.

    Returns:
      dict mapping (group, bucket_index) -> gap value.
    """
    out: Dict[Tuple[Any, int], float] = {}
    if not batch:
        return out

    use_arm = int(arm_for_cal) if int(arm_for_cal) in (0, 1) else 0

    # scores and group labels
    arr_scores = np.asarray([_get_score(b, 0.5) for b in batch], dtype=float)
    arr_groups = np.asarray([b.get("a", None) for b in batch], dtype=object)

    # GA weights over the full audit window
    fallback_gamma = float(getattr(getattr(dr, "cfg", None), "decay_gamma", 0.0) or 0.0)
    w_all = _ga_weights_for_items(list(batch), fallback_gamma=fallback_gamma)
    w_all = np.where(np.isfinite(w_all), w_all, 0.0)
    w_all = np.maximum(w_all, 0.0)
    sum_w_total = float(np.sum(w_all))
    if not np.isfinite(sum_w_total) or sum_w_total <= 0.0:
        w_all = np.ones(len(batch), dtype=float)
        sum_w_total = float(np.sum(w_all))

    # DR pseudo-outcomes
    if use_arm == 0:
        arr_gamma = np.asarray(
            [
                float(
                    b.get(
                        "gamma0",
                        b.get("gamma_0", b.get("mu0", b.get("mu0_hat", 0.5))),
                    )
                )
                for b in batch
            ],
            dtype=float,
        )
    else:
        arr_gamma = np.asarray(
            [
                float(
                    b.get(
                        "gamma1",
                        b.get("gamma_1", b.get("mu1", b.get("mu1_hat", 0.5))),
                    )
                )
                for b in batch
            ],
            dtype=float,
        )
    arr_gamma = np.where(np.isfinite(arr_gamma), arr_gamma, 0.0)

    # mirror GraphAwareDR's minimum-effective-samples gate
    min_eff = int(getattr(getattr(dr, "cfg", None), "min_eff_samples", 0) or 0)
    min_eff = max(0, min_eff)
    min_per_bucket = int(max(0, min_per_bucket))

    for s in groups:
        mask_s = (arr_groups == s)
        n_s = int(np.sum(mask_s))
        if n_s <= 0:
            continue

        # buckets are defined on (unweighted) equal mass within group
        min_mass_local = float(max(float(min_mass), float(min_per_bucket) / float(max(1, n_s))))
        buckets = equal_mass_buckets(arr_scores[mask_s], int(buckets_per_group), min_mass=min_mass_local)

        Ey = np.zeros(len(buckets), dtype=float)
        Ep = np.zeros(len(buckets), dtype=float)
        W = np.zeros(len(buckets), dtype=float)
        valid = np.zeros(len(buckets), dtype=bool)

        for bi, (lo, hi) in enumerate(buckets):
            last = (bi == len(buckets) - 1)
            if last:
                in_bin = (arr_scores >= float(lo)) & (arr_scores <= float(hi) + 1e-12)
            else:
                in_bin = (arr_scores >= float(lo)) & (arr_scores < float(hi))
            m = mask_s & in_bin
            n = int(np.sum(m))
            if n < max(min_eff, min_per_bucket):
                continue

            w = w_all[m]
            wsum = float(np.sum(w))
            if not np.isfinite(wsum) or wsum <= 0.0:
                continue

            # Certificate-aligned slice family: keep only slices with enough GA mass.
            if bool(ga_mass_filter):
                mass = float(wsum / max(1e-12, sum_w_total))
                if mass < float(min_mass):
                    continue

            y = arr_gamma[m]
            p = arr_scores[m]
            Ey[bi] = float(np.sum(w * y) / wsum)
            Ep[bi] = float(np.sum(w * p) / wsum)
            W[bi] = wsum
            valid[bi] = True

        # optional isotonic smoothing on Ey across buckets
        if bool(isotonic) and int(np.sum(valid)) >= 2:
            idx = np.where(valid)[0]
            Ey[idx] = _pav_isotonic(Ey[idx], np.maximum(W[idx], 0.0))

        for bi in range(len(buckets)):
            if valid[bi]:
                out[(s, bi)] = float(abs(Ey[bi] - Ep[bi]))

    return out


def gTE(dr: GraphAwareDR, groups: List[Any]) -> float:
    """Treatment-effect parity gap: max_s τ_s - min_s τ_s."""
    te = dr.estimate_TE_by_group(group_key="a")
    vals = [float(v) for (v, n) in te.values() if float(n) > 0]
    if not vals:
        return 0.0
    arr = np.asarray(vals, dtype=float)
    return float(np.max(arr) - np.min(arr))


def gMin(dr: GraphAwareDR, tau_min: float = 0.0) -> float:
    """Minimum effect guard: max_s [tau_min - τ_s]_+."""
    te = dr.estimate_TE_by_group(group_key="a")
    worst = 0.0
    for _, (v, n) in te.items():
        if float(n) > 0:
            worst = max(worst, max(0.0, float(tau_min) - float(v)))
    return float(worst)


def gRisk(dr: GraphAwareDR, groups: List[Any]) -> float:
    """Baseline risk reporting: max_s E[Y(0)|s] - min_s E[Y(0)|s]."""
    risks: List[float] = []
    for g in groups:
        # GraphAwareDR.estimate_EY signature: (arm, cond)
        Ey0, n_eff = dr.estimate_EY(arm=0, cond=("a", int(g)))
        if float(n_eff) > 0.0:
            risks.append(float(Ey0))

    if len(risks) < 2:
        return 0.0
    arr = np.asarray(risks, dtype=float)
    return float(np.max(arr) - np.min(arr))























