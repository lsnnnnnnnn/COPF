"""
Decision policies with exploration (OPP-2)
Implements coverage-driven exploration from Section 14.4
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

def softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Compute softmax with temperature"""
    x = np.asarray(x, dtype=float)
    tau = max(1e-8, float(tau))
    x = x / tau
    x = x - np.max(x)  # Numerical stability
    ex = np.exp(x)
    s = ex.sum()
    if s <= 0:
        return np.ones_like(ex) / float(len(ex))
    return ex / s

def _plackett_luce_sample_once(scores: np.ndarray, k: int, temperature: float,
                               rng: np.random.Generator) -> List[int]:
    """Sample k items using Plackett-Luce model"""
    n = int(scores.shape[0])
    remaining = list(range(n))
    s = np.array(scores, dtype=float)
    chosen: List[int] = []
    k = int(max(0, min(k, n)))
    
    for _ in range(k):
        if not remaining:
            break
        p = softmax(s[remaining], tau=temperature)
        idx = int(rng.choice(len(remaining), p=p))
        item = remaining.pop(idx)
        chosen.append(item)
    return chosen

def _entry_probs_pl_mc(scores: np.ndarray, k: int, temperature: float, mc: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Monte-Carlo estimation of entry probabilities under PL-TopK"""
    n = len(scores)
    k = int(max(0, min(k, n)))
    if k == 0 or n == 0:
        return np.zeros(n, dtype=float)
    
    counts = np.zeros(n, dtype=float)
    mc = int(max(32, mc))
    
    for _ in range(mc):
        slate = _plackett_luce_sample_once(scores, k, temperature, rng)
        counts[slate] += 1.0
    
    return counts / float(mc)

def decide_with_exploration(
    cands: List[Dict[str, Any]],
    policy: str,
    topk: int,
    epsilon: float,
    temperature: float,
    explore_mask: Optional[np.ndarray] = None,
    epsilon_vec: Optional[np.ndarray] = None,
    mc_samples: int = 128,
    rng: Optional[np.random.Generator] = None
) -> Tuple[List[int], List[float]]:
    """
    Make exposure decisions with exploration and log propensities (OPP-2)
    
    Args:
        cands: List of candidates with scores
        policy: "topk_stochastic" or "epsilon_greedy"
        topk: Number of items to select
        epsilon: Base exploration rate
        temperature: Temperature for stochastic selection
        explore_mask: Boolean mask for forced exploration
        epsilon_vec: Per-candidate exploration rates (overrides epsilon)
        mc_samples: Number of Monte Carlo samples for propensity estimation
        rng: Random number generator
    
    Returns:
        decisions: List[int] - Binary exposure decisions
        propensities: List[float] - P(D=1|policy) for each candidate
    """
    assert policy in ("topk_stochastic", "epsilon_greedy"), \
        "policy must be one of {'topk_stochastic','epsilon_greedy'}"
    
    n = len(cands)
    if n == 0:
        return [], []
    
    if rng is None:
        rng = np.random.default_rng()
    
    scores = np.array([float(c.get("p_hat", 0.5)) for c in cands], dtype=float)
    topk = int(max(0, min(topk, n)))
    
    # IMPORTANT (DR correctness): the logged propensities must match the
    # actual sampling policy. We therefore implement a *single* exploration
    # rate at the slate level (epsilon_avg). If a caller provides epsilon_vec,
    # we reduce it to its mean. Per-candidate epsilon requires a different
    # policy definition and entry-probability computation.
    if epsilon_vec is not None:
        epsilon_vec = np.asarray(epsilon_vec, dtype=float)
        assert epsilon_vec.shape[0] == n, f"epsilon_vec size {epsilon_vec.shape[0]} != n {n}"
        epsilon_avg = float(np.mean(epsilon_vec))
    else:
        epsilon_avg = float(epsilon)

    epsilon_avg = float(np.clip(epsilon_avg, 0.0, 1.0))
    temperature = float(max(1e-6, temperature))
    
    # Exploration pool (candidates marked for exploration)
    idx_all = np.arange(n)
    if explore_mask is not None and explore_mask.dtype == bool and explore_mask.shape[0] == n:
        pool = idx_all[explore_mask]
    else:
        pool = idx_all  # Fallback: all candidates
    
    mc = int(max(32, mc_samples))
    entry = np.zeros(n, dtype=float)
    
    if policy == "topk_stochastic":
        # Plackett-Luce Top-K with exploration
        
        # Entry probabilities under PL
        entry_pl = _entry_probs_pl_mc(scores, topk, temperature, mc, rng) if topk > 0 else np.zeros(n, float)
        
        # Entry probabilities under uniform exploration (only in pool)
        if len(pool) == 0 or topk == 0:
            entry_uni = np.zeros(n, dtype=float)
        else:
            if len(pool) <= topk:
                # All pool items selected
                entry_uni = np.zeros(n, dtype=float)
                entry_uni[pool] = 1.0
            else:
                # Uniform sampling from pool
                p_in = float(topk) / float(len(pool))
                entry_uni = np.zeros(n, dtype=float)
                entry_uni[pool] = p_in
        
        # Mix exploitation and exploration
        entry = (1.0 - epsilon_avg) * entry_pl + epsilon_avg * entry_uni
        
        # Sample actual slate
        if rng.random() < (1.0 - epsilon_avg):
            # Exploit: PL sampling
            slate = _plackett_luce_sample_once(scores, topk, temperature, rng)
        else:
            # Explore: uniform from pool
            if len(pool) <= topk:
                slate = pool.tolist()
            else:
                slate = rng.choice(pool, size=topk, replace=False).tolist()
    
    else:  # "epsilon_greedy"
        # Deterministic Top-K with epsilon-greedy exploration
        
        # Deterministic top-K
        if topk > 0:
            topk_idx = np.argsort(-scores)[:topk]
        else:
            topk_idx = np.array([], dtype=int)
        
        entry_top = np.zeros(n, dtype=float)
        entry_top[topk_idx] = 1.0  # Deterministic selection
        
        # Uniform exploration
        if len(pool) == 0 or topk == 0:
            entry_uni = np.zeros(n, dtype=float)
        else:
            if len(pool) <= topk:
                entry_uni = np.zeros(n, dtype=float)
                entry_uni[pool] = 1.0
            else:
                p_in = float(topk) / float(len(pool))
                entry_uni = np.zeros(n, dtype=float)
                entry_uni[pool] = p_in
        
        # Mix exploitation and exploration
        entry = (1.0 - epsilon_avg) * entry_top + epsilon_avg * entry_uni
        
        # Sample actual slate
        if rng.random() < (1.0 - epsilon_avg):
            # Exploit: deterministic top-K
            slate = topk_idx.tolist()
        else:
            # Explore: uniform from pool
            if len(pool) <= topk:
                slate = pool.tolist()
            else:
                slate = rng.choice(pool, size=topk, replace=False).tolist()
    
    # Ensure propensities are valid probabilities
    entry = np.clip(entry, 1e-6, 1.0 - 1e-6)
    
    # Create decision vector
    d = np.zeros(n, dtype=int)
    d[slate] = 1
    
    return d.tolist(), entry.tolist()




