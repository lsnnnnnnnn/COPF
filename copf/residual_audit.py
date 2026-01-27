from __future__ import annotations
from collections import deque
import numpy as np
from typing import Callable, List


class ResidualOIAudit:
    def __init__(self, auditors: List[Callable], window: int = 1000):
        self.auditors = auditors
        self.buffer = deque(maxlen=window)

    def log(self, x: dict, r: float) -> None:
        self.buffer.append((x, r))

    def sup_violation(self) -> float:
        if not self.buffer:
            return 0.0

        vals = []
        for h in self.auditors:
            s = 0.0
            for x, r in self.buffer:
                s += h(x) * r
            vals.append(abs(s / len(self.buffer)))
        return float(max(vals))