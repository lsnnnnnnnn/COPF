from typing import List, Dict, Any, Tuple, Optional
import math
import json
import numpy as np
import pandas as pd
import os


def _stable_hash(u: int, v: int, t: int) -> int:
    return ((u * 1000003) ^ (v * 0x9E3779B1) ^ (t * 0x85EBCA6B)) & 0xFFFFFFFF

def _sorted_by_score(cands: List[Dict[str, Any]]):
    return sorted(
        cands,
        key=lambda c: (
            -float(c.get("p_hat", 0.0)),  # Negative for descending sort
            _stable_hash(int(c.get("u", 0)), int(c.get("v", 0)), int(c.get("t", 0)))
        )
    )

def _labels(cands: List[Dict[str, Any]]):
    return [int(c.get("y_eval", c.get("y_true", 0))) for c in cands]


def average_precision(cands: List[Dict[str, Any]]) -> float:
    if not cands:
        return 0.0
    ranked = _sorted_by_score(cands)
    ys = [int(c.get("y_eval", c.get("y_true", 0))) for c in ranked]
    npos = sum(ys)
    if npos == 0:
        return 0.0
    num, hit = 0.0, 0
    for i, y in enumerate(ys, start=1):
        if y > 0:
            hit += 1
            num += hit / float(i)
    return float(num / npos)

def mrr(cands: List[Dict[str, Any]]) -> float:
    if not cands:
        return 0.0
    ranked = _sorted_by_score(cands)
    for i, c in enumerate(ranked, start=1):
        if int(c.get("y_eval", c.get("y_true", 0))) > 0:
            return 1.0 / float(i)
    return 0.0

def hits_at_k(cands: List[Dict[str, Any]], k: int) -> float:
    if not cands or k <= 0:
        return 0.0
    ranked = _sorted_by_score(cands)[:k]
    return 1.0 if any(int(c.get("y_eval", c.get("y_true", 0))) > 0 for c in ranked) else 0.0




def recall_at_k(cands: List[Dict[str, Any]], k: int) -> float:
    if not cands or k <= 0:
        return 0.0
    
    npos = sum(1 for c in cands if c.get("y_eval", c.get("y_true", 0)) > 0)
    if npos == 0:
        return 0.0
    
    ranked = _sorted_by_score(cands)[:k]
    hit = sum(1 for c in ranked if c.get("y_eval", c.get("y_true", 0)) > 0)
    recall = float(hit) / float(npos)
    
    return recall





def ndcg_at_k(cands: List[Dict[str, Any]], k: int) -> float:
    if not cands or k <= 0:
        return 0.0
    
    all_labels = _labels(cands)
    n_pos = sum(all_labels)
    
    if n_pos == 0:
        return 0.0
    
    ranked = _sorted_by_score(cands)[:k]
    
    dcg = 0.0
    for i, c in enumerate(ranked):
        rel = float(c.get("y_eval", c.get("y_true", 0)))
        if rel > 0:
            dcg += 1.0 / math.log2(i + 2.0)
    
    idcg = 0.0
    for i in range(min(k, n_pos)):
        idcg += 1.0 / math.log2(i + 2.0)
    
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_score_distribution(cands: List[Dict[str, Any]]) -> Dict[str, float]:
    if not cands:
        return {}
    
    pos_scores = [c.get("p_hat", 0.0) for c in cands if c.get("y_eval", c.get("y_true", 0)) > 0]
    neg_scores = [c.get("p_hat", 0.0) for c in cands if c.get("y_eval", c.get("y_true", 0)) == 0]
    
    stats = {}
    if pos_scores:
        stats["pos_mean"] = float(np.mean(pos_scores))
        stats["pos_std"] = float(np.std(pos_scores))
        stats["pos_min"] = float(np.min(pos_scores))
        stats["pos_max"] = float(np.max(pos_scores))
        
    if neg_scores:
        stats["neg_mean"] = float(np.mean(neg_scores))
        stats["neg_std"] = float(np.std(neg_scores))
        stats["neg_min"] = float(np.min(neg_scores))
        stats["neg_max"] = float(np.max(neg_scores))
        
    if pos_scores and neg_scores:
        stats["score_separation"] = stats["pos_mean"] - stats["neg_mean"]
        
    return stats

def compute_deltas(pre_fair: Dict[str, Any], post_fair: Dict[str, Any]) -> Tuple[float, float]:
    keys = ["gCal_max", "gTE_gap", "gMin"]
    drift = max(abs(float(post_fair.get(k, 0.0)) - float(pre_fair.get(k, 0.0))) for k in keys)
    
    def _avg_tau(f):
        tbg = f.get("tau_by_group", {})
        if not tbg:
            return 0.0
        vals = [float(v.get("tau", 0.0)) for v in tbg.values() if isinstance(v, dict)]
        return float(sum(vals) / max(1, len(vals))) if vals else 0.0
        
    delta_eff = abs(_avg_tau(post_fair) - _avg_tau(pre_fair))
    return float(drift), float(delta_eff)

def tau_bands_json(tau_by_group: Dict[Any, Dict[str, float]], z: float = 1.96) -> str:
    out = {}
    for g, d in tau_by_group.items():
        if isinstance(d, dict):
            tau = float(d.get("tau", 0.0))
            n = max(1, int(d.get("n_eff", 0)))
            rad = float(z * (0.5 / float(n)) ** 0.5)
            out[str(g)] = {
                "tau": tau, 
                "n_eff": n, 
                "ci_low": tau - rad, 
                "ci_high": tau + rad
            }
    return json.dumps(out, ensure_ascii=False, sort_keys=True)

def export_phase_results(phase_results: Dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    
    for phase in ["pre", "deploy", "post"]:
        if phase not in phase_results:
            continue
            
        res = phase_results[phase]
        util = res.get("utility", {})
        fair = res.get("fairness", {})
        
        row = {
            "phase": phase,
            "mrr": util.get("mrr", 0.0),
            "ap": util.get("ap", 0.0),
            "hits@k": util.get("hits@10", util.get("hits10", 0.0)),
            "recall@k": util.get("recall", 0.0),
            "ndcg@k": util.get("ndcg", util.get("ndcg@10", 0.0)),
            "gCal_max": fair.get("gCal_max", 0.0),
            "gTE_gap": fair.get("gTE_gap", 0.0),
            "gMin": fair.get("gMin", 0.0),
            "gRisk": fair.get("gRisk", 0.0),
            "tau_avg": fair.get("tau_avg", 0.0),
            "tau_min": fair.get("tau_min", 0.0),
            "tau_bands": tau_bands_json(fair.get("tau_by_group", {})),
        }
        rows.append(row)
        
    if "pre" in phase_results and "post" in phase_results:
        drift, eff = compute_deltas(
            phase_results["pre"]["fairness"],
            phase_results["post"]["fairness"]
        )
        summary_row = {
            "phase": "summary",
            "delta_drift": drift,
            "delta_eff": eff,
            "utility_preservation": (
                phase_results["post"]["utility"].get("mrr", 0.0) / 
                max(1e-6, phase_results["pre"]["utility"].get("mrr", 0.0))
            ),
        }
        rows.append(summary_row)
        
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, "summary.csv")
        df.to_csv(csv_path, index=False)
        return df
    return None






