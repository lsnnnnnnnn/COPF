"""
Online Primal-Dual Optimizer (Algorithm 1)
Implements constrained optimization with PI controller for fairness
"""
from typing import Dict, Any, List, Optional
import numpy as np

class PrimalDualOptimizer:
    """
    Online primal-dual optimization for COPF (Algorithm 1)
    Uses PI control for dual updates (Section 14.3)
    """
    
    def __init__(self, gamma_p: float = 0.1, gamma_i: float = 0.01, lr: float = 1e-3):
        """
        Args:
            gamma_p: Proportional gain for PI controller
            gamma_i: Integral gain for PI controller  
            lr: Learning rate for primal updates (η in Algorithm 1)
        """
        self.gamma_p = gamma_p
        self.gamma_i = gamma_i
        self.lr = lr
        
        # Dual variables λ_k (Lagrange multipliers)
        self.lambdas = {
            "gCal": 0.0,
            "gTE": 0.0, 
            "gMin": 0.0,
            "gRisk": 0.0
        }
        
        # Integral error terms for PI control
        self.integral_errors = {
            "gCal": 0.0,
            "gTE": 0.0,
            "gMin": 0.0, 
            "gRisk": 0.0
        }
        
        # History for diagnostics
        self.violation_history = {k: [] for k in self.lambdas.keys()}
        self.dual_history = {k: [] for k in self.lambdas.keys()}
        
        self.t = 0
        
    def update_duals(self, violations: Dict[str, float], targets: Dict[str, float]):
        """
        Update dual variables via PI controller (Algorithm 2, Line 13)
        
        λ_{t+1,k} ← [λ_{t,k} + γ_p v_{t,k} + γ_i Σ_{s≤t} v_{s,k}]_+
        where v_{t,k} = F̃^{cf}_{t,k}(f_t) - ρ_{t,k}
        
        Args:
            violations: Current (softened) constraint violations
            targets: Ramp schedule targets ρ_{t,k}
        """
        for k in self.lambdas:
            if k not in violations:
                continue
                
            # Compute violation v_{t,k} = constraint - target
            v_t = violations[k] - targets.get(k, 0.0)
            
            # Update integral term
            self.integral_errors[k] += v_t
            
            # PI control update (Section 14.3)
            delta = self.gamma_p * v_t + self.gamma_i * self.integral_errors[k]
            
            # Update dual with projection to non-negative
            self.lambdas[k] = max(0.0, self.lambdas[k] + delta)
            
            # Store history
            self.violation_history[k].append(v_t)
            self.dual_history[k].append(self.lambdas[k])
            
        self.t += 1
        
    def get_loss_weights(self, cands: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """
        Compute per-candidate loss weights for gradient modification
        Implements ∇L_total = ∇ℓ_util + Σ_k λ_k ∇F^{cf}_k from Algorithm 1, Line 7
        
        Returns:
            Per-sample weights for weighted BCE loss
        """
        if not cands or all(v == 0.0 for v in self.lambdas.values()):
            return None
            
        n = len(cands)
        weights = np.ones(n, dtype=float)
        
        for i, c in enumerate(cands):
            # Extract features
            g = int(c.get("a", 0))  # Group membership
            p = float(c.get("p_hat", 0.5))  # Score
            
            # Base weight
            w = 1.0
            
            # gTE: Treatment effect parity
            # Gradient pushes for equal treatment effects across groups
            if self.lambdas["gTE"] > 0:
                # Upweight minority group to increase their treatment effect
                if g == 1:  # Assuming binary groups 0/1
                    w += self.lambdas["gTE"] * 0.2
                else:
                    w -= self.lambdas["gTE"] * 0.1
                    
            # gCal: Calibration
            # Gradient pushes for better calibration
            if self.lambdas["gCal"] > 0:
                # Get observed outcome (if exposed)
                if c.get("d", 0) == 1:
                    y = float(c.get("y", 0))
                    cal_error = abs(y - p)
                    # Upweight miscalibrated samples
                    w += self.lambdas["gCal"] * cal_error
                    
            # gMin: Minimum effect guard
            # Gradient pushes for higher overall treatment effects
            if self.lambdas["gMin"] > 0:
                # Prefer candidates with higher expected benefit
                expected_benefit = p * 0.2  # Simplified TE estimate
                w *= (1.0 + self.lambdas["gMin"] * expected_benefit)
                
            # gRisk: Baseline risk (optional)
            if self.lambdas["gRisk"] > 0:
                # Balance baseline risks across groups
                if g == 1:
                    w *= (1.0 - self.lambdas["gRisk"] * 0.05)
                    
            weights[i] = max(0.1, min(10.0, w))  # Clip for stability
            
        # Normalize weights to maintain average gradient magnitude
        weights = weights / weights.mean()
        
        return weights
        
    def compute_fairness_penalty(self, metrics: Dict[str, float]) -> float:
        """
        Compute fairness penalty for regularized objective (if using penalized mode)
        L_t(f) = ℓ_util(f) + Σ_k λ_k g_k(f)
        
        Args:
            metrics: Current fairness metric values (already softened)
            
        Returns:
            Total fairness penalty
        """
        penalty = 0.0
        
        for k, lam in self.lambdas.items():
            if k in metrics and lam > 0:
                penalty += lam * metrics[k]
                
        return float(penalty)
        
    def get_gradient_multipliers(self) -> Dict[str, float]:
        """
        Get gradient multipliers for each constraint
        Used for explicit gradient modification in model update
        """
        return self.lambdas.copy()
        
    def reset_integrals(self):
        """Reset integral terms (useful between phases)"""
        self.integral_errors = {k: 0.0 for k in self.lambdas.keys()}
        
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get optimizer diagnostics"""
        diag = {
            't': self.t,
            'duals': self.lambdas.copy(),
            'integral_errors': self.integral_errors.copy()
        }
        
        # Add recent violation statistics
        for k in self.lambdas:
            if self.violation_history[k]:
                recent = self.violation_history[k][-10:]
                diag[f'{k}_recent_mean'] = float(np.mean(recent))
                diag[f'{k}_recent_std'] = float(np.std(recent))
                diag[f'{k}_converged'] = bool(
                    len(recent) >= 10 and np.std(recent) < 0.01
                )
                
        return diag
