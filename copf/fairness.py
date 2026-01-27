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
    n = len(items)
    if n == 0:
        return np.zeros(0, dtype=float)

    w_local = np.array([float(it.get("w_local", 1.0)) for it in items], dtype=float)
    w_local = np.where(np.isfinite(w_local), w_local, 0.0)
    w_local = np.maximum(w_local, 0.0)
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
    
    out: Dict[Tuple[Any, int], float] = {}
    if not batch:
        return out

    use_arm = int(arm_for_cal) if int(arm_for_cal) in (0, 1) else 0

    
    arr_scores = np.asarray([_get_score(b, 0.5) for b in batch], dtype=float)
    arr_groups = np.asarray([b.get("a", None) for b in batch], dtype=object)
    estimator = str(getattr(getattr(dr, "cfg", None), "estimator", "dr") or "dr").strip().lower()
    if estimator in {"observed", "observed_only", "observed-only", "naive"}:
        estimator = "obs"
    if estimator == "obs":
        arr_d = np.asarray([int(b.get("d", -1)) for b in batch], dtype=int)
    else:
        arr_d = None

    
    fallback_gamma = float(getattr(getattr(dr, "cfg", None), "decay_gamma", 0.0) or 0.0)
    w_all = _ga_weights_for_items(list(batch), fallback_gamma=fallback_gamma)
    w_all = np.where(np.isfinite(w_all), w_all, 0.0)
    w_all = np.maximum(w_all, 0.0)
    sum_w_total = float(np.sum(w_all))
    if not np.isfinite(sum_w_total) or sum_w_total <= 0.0:
        w_all = np.ones(len(batch), dtype=float)
        sum_w_total = float(np.sum(w_all))

    
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

    
    min_eff = int(getattr(getattr(dr, "cfg", None), "min_eff_samples", 0) or 0)
    min_eff = max(0, min_eff)
    min_per_bucket = int(max(0, min_per_bucket))

    for s in groups:
        mask_s = (arr_groups == s)
        n_s = int(np.sum(mask_s))
        if n_s <= 0:
            continue

        
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
            if estimator == "obs" and arr_d is not None:
                m = m & (arr_d == int(use_arm))
            n = int(np.sum(m))
            if n < max(min_eff, min_per_bucket):
                continue

            w = w_all[m]
            wsum = float(np.sum(w))
            if not np.isfinite(wsum) or wsum <= 0.0:
                continue

            
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

        
        if bool(isotonic) and int(np.sum(valid)) >= 2:
            idx = np.where(valid)[0]
            Ey[idx] = _pav_isotonic(Ey[idx], np.maximum(W[idx], 0.0))

        for bi in range(len(buckets)):
            if valid[bi]:
                out[(s, bi)] = float(abs(Ey[bi] - Ep[bi]))

    return out


def gTE(dr: GraphAwareDR, groups: List[Any]) -> float:
    te = dr.estimate_TE_by_group(group_key="a")
    vals: List[float] = []
    for g in groups:
        if int(g) in te:
            v, n = te[int(g)]
            if float(n) > 0.0 and np.isfinite(float(v)):
                vals.append(float(v))
    if len(vals) < 2:
        return 0.0
    arr = np.asarray(vals, dtype=float)
    return float(np.max(arr) - np.min(arr))


def gMin(dr: GraphAwareDR, tau_min: float = 0.0, groups: List[Any] | None = None) -> float:
    te = dr.estimate_TE_by_group(group_key="a")
    if groups is None:
        groups_iter = list(te.keys())
    else:
        groups_iter = [int(g) for g in groups]

    worst = 0.0
    for g in groups_iter:
        if int(g) not in te:
            continue
        v, n = te[int(g)]
        if float(n) > 0.0 and np.isfinite(float(v)):
            worst = max(worst, max(0.0, float(tau_min) - float(v)))
    return float(worst)


def gRisk(dr: GraphAwareDR, groups: List[Any]) -> float:
    risks: List[float] = []
    for g in groups:
        Ey0, n_eff = dr.estimate_EY(arm=0, cond=("a", int(g)))
        if float(n_eff) > 0.0:
            risks.append(float(Ey0))

    if len(risks) < 2:
        return 0.0
    arr = np.asarray(risks, dtype=float)
    return float(np.max(arr) - np.min(arr))































































































