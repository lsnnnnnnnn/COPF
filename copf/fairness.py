#fairness.py
from typing import List, Dict, Any, Tuple
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
    """Pool-adjacent-violators for isotonic regression"""
    y = np.asarray(y, float).copy()
    w = np.asarray(w, float).copy()
    if y.size == 0:
        return y
        
    blocks = [(y[i], w[i], 1) for i in range(y.size)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            y_sum = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            w_sum = blocks[i][1] + blocks[i + 1][1]
            L = blocks[i][2] + blocks[i + 1][2]
            avg = y_sum / (w_sum if w_sum > 0 else max(1.0, L))
            blocks[i] = (avg, w_sum, L)
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
            
    out = []
    for avg, _, L in blocks:
        out.extend([avg] * L)
    return np.asarray(out, float)

def gCal(
    batch: List[Dict[str, Any]],
    dr: GraphAwareDR,
    groups: List[Any],
    buckets_per_group: int = 10,
    isotonic: bool = False,
    min_mass: float = 0.02,
    min_per_bucket: int = 0,
    arm_for_cal: int = 0,
) -> Dict[Tuple[Any, int], float]:
    """
    Within-group counterfactual calibration: |E[Y^(a)|bucket, group] - mid(bucket)|
    """
    out: Dict[Tuple[Any, int], float] = {}
    if not batch:
        return out
        
    use_arm = arm_for_cal if arm_for_cal in (0, 1) else 0
    arr_scores = np.array([_get_score(b, 0.5) for b in batch], float)
    arr_groups = np.array([b.get("a", None) for b in batch], object)
    
    for s in groups:
        mask_s = (arr_groups == s)
        n_s = int(mask_s.sum())
        if n_s == 0:
            continue
            
        min_mass_local = float(max(min_mass, float(min_per_bucket) / float(max(1, n_s))))
        buckets = equal_mass_buckets(arr_scores[mask_s], int(buckets_per_group), 
                                    min_mass=min_mass_local)
        
        vals, ns, mids = [], [], []
        for i, (lo, hi) in enumerate(buckets):
            last = (i == len(buckets) - 1)
            
            def _cond(c, s=s, lo=lo, hi=hi, last=last):
                sc = _get_score(c, 0.5)
                if last:
                    in_bin = (sc >= lo) and (sc <= hi + 1e-12)
                else:
                    in_bin = (sc >= lo) and (sc < hi)
                return (c.get("a", None) == s) and in_bin
                
            Ey, n = dr.estimate_EY(_cond, arm=use_arm)
            Ey = float(max(1e-3, min(1.0 - 1e-3, Ey)))
            vals.append(Ey)
            ns.append(int(n))
            mids.append(0.5 * (float(lo) + float(hi)))
            
        vals = np.asarray(vals, float)
        ns = np.asarray(ns, int)
        mids = np.asarray(mids, float)
        
        if isotonic and (ns > 0).sum() >= 2:
            vals = _pav_isotonic(vals, ns.clip(min=0))
            
        for i in range(len(mids)):
            if ns[i] > 0:
                out[(s, i)] = float(abs(vals[i] - mids[i]))
                
    return out

def gTE(dr: GraphAwareDR, groups: List[Any]) -> float:
    """Treatment effect parity gap"""
    te = dr.estimate_TE_by_group(group_key="a")
    vals = [v for (v, n) in te.values() if n > 0]
    if not vals:
        return 0.0
    arr = np.array(vals, float)
    return float(np.max(arr) - np.min(arr))

def gMin(dr: GraphAwareDR, tau_min: float = 0.0) -> float:
    """Minimum effect guard"""
    te = dr.estimate_TE_by_group(group_key="a")
    worst = 0.0
    for s, (v, n) in te.items():
        if n > 0:
            worst = max(worst, max(0.0, float(tau_min) - v))
    return float(worst)

def gRisk(dr: GraphAwareDR, groups: List[Any]) -> float:
    """Baseline risk reporting (optional metric)"""
    risks = []
    for g in groups:
        def cond(c, g=g):
            return c.get("a", None) == g
        Ey0, n = dr.estimate_EY(cond, arm=0)
        if n > 0:
            risks.append(Ey0)
            
    if len(risks) < 2:
        return 0.0
        
    arr = np.array(risks, float)
    return float(np.max(arr) - np.min(arr))











