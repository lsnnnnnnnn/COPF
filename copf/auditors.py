from __future__ import annotations
import numpy as np
from typing import Callable, List, Dict


def build_group_auditors(group_map: Dict[int, int], groups: List[int]) -> List[Callable]:
    auditors = []
    for g in groups:
        def h(x, g=g):
            return 1.0 if x["group"] == g else 0.0
        auditors.append(h)
    return auditors


def build_score_bucket_auditors(buckets: List[tuple[float, float]]) -> List[Callable]:
    auditors = []
    for lo, hi in buckets:
        def h(x, lo=lo, hi=hi):
            return 1.0 if (lo <= x["score"] < hi) else 0.0
        auditors.append(h)
    return auditors


def combine_auditors(A: List[Callable], B: List[Callable]) -> List[Callable]:
    auditors = []
    for ha in A:
        for hb in B:
            def h(x, ha=ha, hb=hb):
                return ha(x) * hb(x)
            auditors.append(h)
    return auditors