from __future__ import annotations
from typing import List, Tuple
import numpy as np

Bucket = Tuple[float, float]

def _finite_unit(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=float).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return s
    return np.clip(s, 0.0, 1.0)

def equal_mass_buckets(scores: np.ndarray, n_buckets: int, min_mass: float = 0.02) -> List[Bucket]:
    """
    Equal-mass buckets on [0,1]. Robust to NaN/Inf and degenerate distributions.
    Will merge buckets whose empirical mass < min_mass.
    """
    s = _finite_unit(scores)
    if s.size == 0:
        return [(0.0, 1.0)]

    B = int(max(1, n_buckets))
    qs = np.linspace(0.0, 1.0, B + 1)
    edges = np.quantile(s, qs)

    # Force coverage and monotonicity
    edges[0] = 0.0
    edges[-1] = 1.0
    edges = np.maximum.accumulate(edges)

    # Build initial buckets (keep B buckets even if edges repeat; we'll merge by mass)
    buckets: List[Bucket] = [(float(edges[i]), float(edges[i + 1])) for i in range(B)]

    def mass(i: int) -> float:
        lo, hi = buckets[i]
        if i == len(buckets) - 1:
            return float(((s >= lo) & (s <= hi)).mean())
        return float(((s >= lo) & (s < hi)).mean())

    masses = [mass(i) for i in range(len(buckets))]

    # Merge tiny-mass buckets with a neighbor
    i = 0
    while i < len(buckets) and len(buckets) > 1:
        if masses[i] >= min_mass:
            i += 1
            continue

        left = i - 1
        right = i + 1

        # choose merge direction: prefer neighbor with larger mass if exists
        if left < 0:
            j = right
        elif right >= len(buckets):
            j = left
        else:
            j = left if masses[left] >= masses[right] else right

        lo = min(buckets[i][0], buckets[j][0])
        hi = max(buckets[i][1], buckets[j][1])
        ii, jj = (i, j) if i < j else (j, i)

        # delete higher index first
        del buckets[jj]; del masses[jj]
        del buckets[ii]; del masses[ii]

        buckets.insert(ii, (lo, hi))
        masses.insert(ii, mass(ii))

        i = max(0, ii - 1)

    # Ensure exact [0,1]
    if buckets[0][0] != 0.0:
        buckets[0] = (0.0, buckets[0][1])
    if buckets[-1][1] != 1.0:
        buckets[-1] = (buckets[-1][0], 1.0)

    return buckets

def bucket_index(buckets: List[Bucket], s: float) -> int:
    if not buckets:
        return 0
    x = float(s)
    if not np.isfinite(x):
        x = 0.0
    x = max(0.0, min(1.0, x))
    for i, (lo, hi) in enumerate(buckets):
        if i == len(buckets) - 1:
            if lo <= x <= hi + 1e-12:
                return i
        else:
            if lo <= x < hi:
                return i
    return len(buckets) - 1

def bucket_mid(b: Bucket) -> float:
    return 0.5 * (float(b[0]) + float(b[1]))