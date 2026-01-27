from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .dr import winsorize01, ratio_stabilize


@dataclass
class OverlapDiagConfig:
    prop_bins: int = 20
    clip: float = 0.05
    ratio_stab: bool = True


class OverlapDiagnostics:
    """Online aggregator for propensity / overlap diagnostics."""

    def __init__(self, cfg: OverlapDiagConfig):
        self.cfg = cfg
        self._edges = np.linspace(0.0, 1.0, int(max(2, cfg.prop_bins)) + 1, dtype=float)

        # stats[(phase, group)] -> dict accumulators
        self.stats: Dict[Tuple[str, int], Dict[str, float]] = {}

        # histograms
        self.hist_total: Dict[Tuple[str, int], np.ndarray] = {}
        self.hist_d1: Dict[Tuple[str, int], np.ndarray] = {}
        self.hist_d0: Dict[Tuple[str, int], np.ndarray] = {}

    def _get_key(self, phase: str, a: Any) -> Tuple[str, int]:
        return (str(phase), int(a) if a is not None else 0)

    def _ensure(self, key: Tuple[str, int]) -> None:
        if key not in self.stats:
            self.stats[key] = {
                "n": 0.0,
                "n_d1": 0.0,
                "n_d0": 0.0,
                "sum_e": 0.0,
                "sum_e_clipped": 0.0,
                "min_e": 1.0,
                "max_e": 0.0,
                "clip_any": 0.0,
                "sum_w1": 0.0,
                "sum_w1_sq": 0.0,
                "sum_w0": 0.0,
                "sum_w0_sq": 0.0,
            }
            B = len(self._edges) - 1
            self.hist_total[key] = np.zeros(B, dtype=float)
            self.hist_d1[key] = np.zeros(B, dtype=float)
            self.hist_d0[key] = np.zeros(B, dtype=float)

    def _clip(self, p: float) -> float:
        p = float(p)
        p2 = winsorize01(p, float(self.cfg.clip))
        if bool(self.cfg.ratio_stab):
            p2 = ratio_stabilize(p2, float(self.cfg.clip))
        return float(p2)

    def update(self, cands: Iterable[Dict[str, Any]], phase: str) -> None:
        """Update diagnostics from one round of candidates."""
        clip = float(self.cfg.clip)
        for c in cands:
            try:
                a = int(c.get("a", 0))
            except Exception:
                a = 0
            key = self._get_key(phase, a)
            self._ensure(key)

            try:
                e = float(c.get("e_hat", np.nan))
            except Exception:
                e = float("nan")
            if not np.isfinite(e):
                continue
            e = float(np.clip(e, 0.0, 1.0))
            d = int(c.get("d", 0))

            st = self.stats[key]
            st["n"] += 1.0
            st["n_d1"] += 1.0 if d == 1 else 0.0
            st["n_d0"] += 1.0 if d == 0 else 0.0
            st["sum_e"] += e
            st["min_e"] = float(min(float(st["min_e"]), e))
            st["max_e"] = float(max(float(st["max_e"]), e))

            clipped_flag = (e <= clip) or (e >= (1.0 - clip))
            st["clip_any"] += 1.0 if clipped_flag else 0.0

            e1 = self._clip(e)
            st["sum_e_clipped"] += e1

            # ESS for IPS weights: treated uses 1/e, control uses 1/(1-e).
            if d == 1:
                w = 1.0 / max(1e-12, float(e1))
                st["sum_w1"] += w
                st["sum_w1_sq"] += w * w
            else:
                e0_raw = 1.0 - e
                e0 = self._clip(e0_raw)
                w = 1.0 / max(1e-12, float(e0))
                st["sum_w0"] += w
                st["sum_w0_sq"] += w * w

            # Histogram bins
            bi = int(np.searchsorted(self._edges, e, side="right") - 1)
            bi = int(np.clip(bi, 0, len(self._edges) - 2))
            self.hist_total[key][bi] += 1.0
            if d == 1:
                self.hist_d1[key][bi] += 1.0
            else:
                self.hist_d0[key][bi] += 1.0

    def to_frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (summary_df, hist_df)."""
        rows: List[Dict[str, Any]] = []
        for (phase, g), st in sorted(self.stats.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            n = float(st.get("n", 0.0))
            n_d1 = float(st.get("n_d1", 0.0))
            n_d0 = float(st.get("n_d0", 0.0))

            mean_e = float(st.get("sum_e", 0.0) / max(1.0, n))
            mean_e_clip = float(st.get("sum_e_clipped", 0.0) / max(1.0, n))
            clip_rate = float(st.get("clip_any", 0.0) / max(1.0, n))

            sw1 = float(st.get("sum_w1", 0.0))
            sw1sq = float(st.get("sum_w1_sq", 0.0))
            ess1 = float((sw1 * sw1) / max(1e-12, sw1sq)) if sw1sq > 0 else 0.0

            sw0 = float(st.get("sum_w0", 0.0))
            sw0sq = float(st.get("sum_w0_sq", 0.0))
            ess0 = float((sw0 * sw0) / max(1e-12, sw0sq)) if sw0sq > 0 else 0.0

            rows.append(
                {
                    "phase": phase,
                    "group": int(g),
                    "n_candidates": int(n),
                    "n_d1": int(n_d1),
                    "n_d0": int(n_d0),
                    "mean_e": mean_e,
                    "mean_e_clipped": mean_e_clip,
                    "min_e": float(st.get("min_e", 0.0)),
                    "max_e": float(st.get("max_e", 0.0)),
                    "clip_rate_any": clip_rate,
                    "ess_d1": ess1,
                    "ess_d0": ess0,
                }
            )

        summary = pd.DataFrame(rows)

        hrows: List[Dict[str, Any]] = []
        for (phase, g), h in sorted(self.hist_total.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            h1 = self.hist_d1[(phase, g)]
            h0 = self.hist_d0[(phase, g)]
            for bi in range(len(self._edges) - 1):
                hrows.append(
                    {
                        "phase": phase,
                        "group": int(g),
                        "bin": int(bi),
                        "bin_left": float(self._edges[bi]),
                        "bin_right": float(self._edges[bi + 1]),
                        "count_total": float(h[bi]),
                        "count_d1": float(h1[bi]),
                        "count_d0": float(h0[bi]),
                    }
                )
        hist = pd.DataFrame(hrows)
        return summary, hist

    def write(self, out_dir: str) -> Tuple[str, str]:
        """Write CSVs and return their paths."""
        summary, hist = self.to_frames()
        path_summary = str(out_dir).rstrip("/") + "/overlap_diag_summary.csv"
        path_hist = str(out_dir).rstrip("/") + "/overlap_diag_propensity_hist.csv"
        summary.to_csv(path_summary, index=False)
        hist.to_csv(path_hist, index=False)
        return path_summary, path_hist