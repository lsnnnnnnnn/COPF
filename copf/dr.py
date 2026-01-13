"""
Graph-Aware Doubly Robust Estimation (GA-DR)
Implements Definition 1 and Section 5 from the paper
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Tuple, Optional
import numpy as np

def winsorize01(x: np.ndarray, eps: float = 0.02) -> np.ndarray:
    """Winsorize to [eps, 1-eps] for stability"""
    lo, hi = eps, 1.0 - eps
    return np.clip(x, lo, hi)

def ratio_stabilize(r: np.ndarray) -> np.ndarray:
    """Stabilize ratios to prevent explosion"""
    return r / (1.0 + np.abs(r))

@dataclass
class DRConfig:
    clip: float = 0.05  # emin for propensity clipping
    self_normalized: bool = True  # Self-normalized DR (Definition 1)
    decay_gamma: float = 0.1  # Locality weight decay γ
    max_buffer: int = 200000
    ratio_stab: bool = True
    winsor_eps: float = 0.02
    min_eff_samples: int = 100

def ga_dr_score_arm(y: np.ndarray, d: np.ndarray, e: np.ndarray, mu: np.ndarray,
                    clip: float = 0.05, do_ratio_stab: bool = True) -> np.ndarray:
    """Compute GA-DR score for one arm"""
    e = np.clip(e, clip, 1.0 - clip)
    res = y - mu
    if do_ratio_stab:
        res = ratio_stabilize(res)
    return mu + (d / e) * res

def ga_dr_aggregate(slice_mask: np.ndarray, gamma: np.ndarray, weight: np.ndarray = None,
                    winsor_eps: float = 0.02) -> float:
    """Aggregate GA-DR scores with weights"""
    m = slice_mask.astype(bool)
    if weight is None:
        weight = np.ones_like(gamma)
    num = np.sum(weight[m] * gamma[m])
    den = np.sum(weight[m]) + 1e-12
    est = num / den
    return float(winsorize01(np.array([est]), eps=winsor_eps)[0])

class GraphAwareDR:
    """
    Graph-Aware Doubly Robust estimator with locality weights
    Implements Definition 1 from the paper
    """
    
    def __init__(self, config: DRConfig, 
                 cross_fitter=None,
                 neighborhood_tracker=None):
        self.cfg = config
        self.buffer: List[Dict[str, Any]] = []
        self.cross_fitter = cross_fitter
        self.neighborhood_tracker = neighborhood_tracker
        self.t = 0
    
    def _clip_propensity(self, e: float) -> float:
        """Clip propensity to ensure overlap (Assumption 1)"""
        eps = float(self.cfg.clip)
        return float(max(eps, min(1.0 - eps, e)))
    
    def ingest(self, cands: List[Dict[str, Any]]):
        """
        Process candidates and compute DR components with cross-fitting
        Implements OPP-4 update step
        """
        for c in cands:
            # Extract base score
            p = float(c.get('p_hat', 0.5))
            
            # Get logged propensity
            e_logged = float(c.get('e_hat', 0.5))
            
            # Treatment and outcome
            d = int(c.get('d', 0))
            y_obs = float(c.get('y', 0))
            
            # Protected attribute and time
            a = c.get('a', None)
            t_step = c.get('t', self.t)
            
            # Get cross-fitted predictions if available
            if self.cross_fitter is not None:
                # Use cross-fitted estimates directly (no blending)
                mu0 = self.cross_fitter.predict_outcome(c, arm=0, t=t_step)
                mu1 = self.cross_fitter.predict_outcome(c, arm=1, t=t_step)
                e1 = self.cross_fitter.predict_propensity(c, t=t_step)
            else:
                # Fallback to simple estimates
                e1 = e_logged
                mu0 = float(c.get('mu0_hat', p))
                mu1 = float(c.get('mu1_hat', p * 1.2))
            
            # Clip propensity for stability
            e1 = self._clip_propensity(e1)
            e0 = 1.0 - e1
            
            # Compute locality weight w_t(i,j) = exp(-γ * NeighborhoodChangeRate)
            if self.neighborhood_tracker is not None and hasattr(c, '__getitem__'):
                if 'u' in c and 'v' in c:
                    w_local = self.neighborhood_tracker.compute_locality_weight(
                        int(c['u']), int(c['v']), int(t_step)
                    )
                else:
                    w_local = 1.0
            else:
                w_local = float(c.get('locality_weight', 1.0))
            
            # Compute GA-DR scores (Definition 1)
            # Γ^(a)_t(i,j) = μ̂_a + (1{D=a}/ê^(a)) * (Y - μ̂_a)
            
            # For Y^(0)
            if d == 0:
                # Observed control
                gamma_0 = mu0 + (1.0 / e0) * (y_obs - mu0)
                if self.cfg.ratio_stab:
                    gamma_0 = mu0 + ratio_stabilize((y_obs - mu0) / e0)
            else:
                # Unobserved control
                gamma_0 = mu0
            
            # For Y^(1)
            if d == 1:
                # Observed treatment
                gamma_1 = mu1 + (1.0 / e1) * (y_obs - mu1)
                if self.cfg.ratio_stab:
                    gamma_1 = mu1 + ratio_stabilize((y_obs - mu1) / e1)
            else:
                # Unobserved treatment
                gamma_1 = mu1
            
            # Compute counterfactual residuals for auditor (Section 6)
            # r^(0) = Y^(0) - p̂
            r0 = gamma_0 - p
            
            # r^(Δ) = (Y^(1) - Y^(0)) - τ(x)
            tau_x = mu1 - mu0  # Plug-in estimate of τ(x)
            r_delta = (gamma_1 - gamma_0) - tau_x
            
            # Store complete record
            item = {
                # Identifiers
                'u': c.get('u'), 'v': c.get('v'),
                't': t_step, 'a': a,
                
                # Scores and propensities
                'p_hat': p, 'e_hat': e_logged, 'e_cf': e1,
                
                # Treatment and outcome
                'd': d, 'y': y_obs, 'y_dr': y_obs,
                
                # Cross-fitted nuisance estimates
                'mu0_hat': mu0, 'mu1_hat': mu1,
                
                # GA-DR scores
                'gamma_0': gamma_0, 'gamma_1': gamma_1,
                
                # Locality weight (Definition 1)
                'w_local': w_local,
                
                # Counterfactual residuals (Section 6)
                'r0': r0, 'r_delta': r_delta, 'tau_x': tau_x, 'x': c.get('x', {}) 
            }
            
            self.buffer.append(item)
        
        # Maintain buffer size
        if len(self.buffer) > self.cfg.max_buffer:
            self.buffer = self.buffer[-self.cfg.max_buffer:]
        
        self.t += 1
    
    def _weights(self, items: List[Dict[str, Any]]) -> np.ndarray:
        """
        Compute combined weights: locality * time_decay
        Implements w_t from Definition 1
        """
        n = len(items)
        if n == 0:
            return np.zeros(0, dtype=float)
        
        # Locality weights from neighborhood change
        w_local = np.array([float(i.get('w_local', 1.0)) for i in items])
        
        # Time decay weights (optional)
        if self.cfg.decay_gamma > 0:
            try:
                t_max = max(int(i.get('t', 0)) for i in items)
            except (ValueError, TypeError):
                t_max = 0
            
            w_time = np.array([
                np.exp(-self.cfg.decay_gamma * max(0, t_max - int(i.get('t', 0))))
                for i in items
            ])
        else:
            w_time = np.ones(n)
        
        return w_local * w_time
    
    def estimate_EY(self, cond: Callable[[Dict[str, Any]], bool], 
                    arm: int) -> Tuple[float, int]:
        """
        Estimate E[Y^(arm) | condition] using GA-DR
        Implements self-normalized estimator from Definition 1
        """
        # Filter buffer by condition
        S = [c for c in self.buffer if cond(c)]
        n = len(S)
        
        if n < self.cfg.min_eff_samples:
            return 0.0, 0
        
        # Get weights
        w = self._weights(S)
        
        # Get GA-DR scores
        if arm == 0:
            gamma = np.array([c['gamma_0'] for c in S])
        else:
            gamma = np.array([c['gamma_1'] for c in S])
        
        # Self-normalized estimator (Definition 1)
        if self.cfg.self_normalized:
            numerator = np.sum(w * gamma)
            denominator = np.sum(w)
            if denominator > 0:
                est = numerator / denominator
            else:
                est = 0.0
        else:
            # Simple weighted average
            est = np.mean(gamma)
        
        # Winsorize for stability
        est = float(winsorize01(np.array([est]), eps=self.cfg.winsor_eps)[0])
        
        return est, n
    
    def estimate_Y0(self, c: Dict[str, Any]) -> float:
        """Estimate Y^(0) for a single candidate (used by auditor)"""
        return float(c.get('gamma_0', c.get('mu0_hat', 0.5)))
    
    def estimate_Y1(self, c: Dict[str, Any]) -> float:
        """Estimate Y^(1) for a single candidate"""
        return float(c.get('gamma_1', c.get('mu1_hat', 0.5)))
    
    def estimate_TE_by_group(self, group_key: str = "a") -> Dict[Any, Tuple[float, int]]:
        """
        Estimate treatment effects τ_s = E[Y^(1) - Y^(0) | A=s]
        Used for gTE and gMin computation
        """
        # Get unique groups
        groups = sorted(list({c.get(group_key) for c in self.buffer}))
        
        results = {}
        for s in groups:
            # Define condition for group s
            def cond_s(c, s=s):
                return c.get(group_key) == s
            
            # Estimate E[Y^(1) | A=s] and E[Y^(0) | A=s]
            ey1, n1 = self.estimate_EY(cond_s, arm=1)
            ey0, n0 = self.estimate_EY(cond_s, arm=0)
            
            # Treatment effect
            te = float(ey1 - ey0)
            
            # Clip to reasonable range
            te = float(np.clip(te, -0.5, 0.5))
            
            # Effective sample size
            n_eff = min(n1, n0)
            
            results[s] = (te, n_eff)
        
        return results
    
    def estimate_slice(self, slice_items: List[Dict[str, Any]], 
                      arm: int, use_stabilization: bool = True) -> Tuple[float, int]:
        """
        Estimate E[Y^(arm)] for a specific slice
        Used for gCal computation
        """
        n = len(slice_items)
        if n < 10:  # Need minimum samples
            return 0.0, 0
        
        # Get weights
        w = self._weights(slice_items)
        
        # Get GA-DR scores
        if arm == 0:
            gamma = np.array([c.get('gamma_0', c.get('mu0_hat', 0.5)) for c in slice_items])
        else:
            gamma = np.array([c.get('gamma_1', c.get('mu1_hat', 0.5)) for c in slice_items])
        
        # Self-normalized estimate
        if self.cfg.self_normalized and w.sum() > 0:
            est = np.sum(w * gamma) / np.sum(w)
        else:
            est = np.mean(gamma)
        
        # Winsorize
        est = float(winsorize01(np.array([est]), eps=self.cfg.winsor_eps)[0])
        
        return est, n
    
    def get_residuals(self, group: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get counterfactual residuals for auditor
        Returns r^(0) and r^(Δ) for residual OI
        """
        if group is not None:
            items = [c for c in self.buffer if c.get('a') == group]
        else:
            items = self.buffer
        
        if not items:
            return np.array([]), np.array([])
        
        r0 = np.array([c.get('r0', 0.0) for c in items])
        r_delta = np.array([c.get('r_delta', 0.0) for c in items])
        
        return r0, r_delta
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information"""
        if not self.buffer:
            return {'buffer_size': 0, 't': self.t}
        
        # Get recent buffer items
        recent = self.buffer[-100:] if len(self.buffer) > 100 else self.buffer
        
        # Compute summary statistics
        e_vals = [c.get('e_cf', c.get('e_hat', 0.5)) for c in recent]
        
        diag = {
            'buffer_size': len(self.buffer),
            't': self.t,
            'e_mean': float(np.mean(e_vals)),
            'e_min': float(np.min(e_vals)),
            'e_max': float(np.max(e_vals)),
            'clip_rate': float(np.mean([
                e <= self.cfg.clip or e >= (1 - self.cfg.clip) 
                for e in e_vals
            ]))
        }
        
        # Add cross-fitter diagnostics if available
        if self.cross_fitter is not None:
            cf_diag = self.cross_fitter.get_model_diagnostics()
            diag['cross_fitter'] = cf_diag
        
        return diag








