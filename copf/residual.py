# copf/residual.py
from __future__ import annotations
import numpy as np

def dr_estimate(Y: float, D: int, e: float, mu: float, a: int, eps: float = 1e-6) -> float:
    """
    Minimal self-normalized DR score for arm a ∈ {0,1}
    """
    e = float(np.clip(e, eps, 1.0 - eps))
    if D == a:
        return mu - (mu - Y) / e
    else:
        return mu


def compute_residuals(
    score: float,
    Y: float,
    D: int,
    e: float,
    mu0: float,
    mu1: float,
) -> tuple[float, float]:
    """
    Returns:
      r0      = Y(0) - score
      rDelta  = (Y(1)-Y(0)) - (mu1-mu0)
    """
    y0_hat = dr_estimate(Y, D, e, mu0, a=0)
    y1_hat = dr_estimate(Y, D, e, mu1, a=1)

    r0 = y0_hat - score
    rDelta = (y1_hat - y0_hat) - (mu1 - mu0)
    return float(r0), float(rDelta)