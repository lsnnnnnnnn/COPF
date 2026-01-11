#/copf/opp/propensity.py
from typing import List, Dict, Any

def log_propensities(cands: List[Dict[str, Any]], d: List[int], e: List[float]):
    assert len(cands) == len(d) == len(e)
    for c, di, ei in zip(cands, d, e):
        c["d"] = int(di)
        c["e_hat"] = float(max(1e-6, min(1.0 - 1e-6, ei)))

def attach_scores(cands: List[Dict[str, Any]], scores):
    for c, s in zip(cands, scores):
        c["p_hat"] = float(s)
