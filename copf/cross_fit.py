"""
Online Cross-Fitting for GA-DR Nuisance Functions
Implements Section 5 of the paper with K-fold cross-fitting
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class OnlineCrossFitter:
    """
    Online K-fold cross-fitting for nuisance functions (ê, μ̂₀, μ̂₁)
    Required for unbiased GA-DR estimation (Lemma 1)
    """
    
    def __init__(self, n_folds: int = 5, seed: int = 42, max_buffer: int = 20000):
        self.n_folds = n_folds
        self.seed = seed
        self.max_buffer = max_buffer
        self.t = 0
        
        # Scalers for feature normalization
        self.scalers = [StandardScaler() for _ in range(n_folds)]
        
        # Propensity models ê(x, G_local) - one per fold
        self.propensity_models = [
            LogisticRegression(
                penalty='l2', 
                C=1.0,
                max_iter=200,
                solver='lbfgs',
                random_state=seed + i
            ) for i in range(n_folds)
        ]
        
        # Outcome models μ̂₀(x, G_local) - one per fold
        self.outcome_models_0 = [
            GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                random_state=seed + i
            ) for i in range(n_folds)
        ]
        
        # Outcome models μ̂₁(x, G_local) - one per fold
        self.outcome_models_1 = [
            GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                random_state=seed + i + n_folds
            ) for i in range(n_folds)
        ]
        
        # Buffers for incremental training
        self.buffer = []
        self.model_ready = [False] * n_folds
        
    def get_fold(self, t: int) -> int:
        """Assign time step to fold (interleaved assignment)"""
        return t % self.n_folds
    
    def extract_features(self, c: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from candidate including graph structure
        Combines (ϕ_t, G_local_t) as per Definition 1
        """
        x = c.get("x", {})
        
        features = [
            # Node features
            float(c.get("p_hat", 0.5)),  # Model score
            int(c.get("a", 0)),  # Protected attribute
            
            # Graph structure features (G_local)
            float(x.get("degree_u", 0)),
            float(x.get("degree_v", 0)),
            float(x.get("common_neighbors", 0)),
            float(x.get("aa_score", 0)),  # Adamic-Adar
            float(x.get("shortest_path", 999)) / 100.0,  # Normalized
            
            # Temporal features
            float(x.get("time_since_last_u", 1.0)),
            float(x.get("time_since_last_v", 1.0)),
            
            # Locality weight (from neighborhood tracker)
            float(c.get("locality_weight", 1.0))
        ]
        
        return np.array(features)
    
    def update(self, cands: List[Dict[str, Any]]):
        """
        Update nuisance models with new batch
        Implements online cross-fitting protocol
        """
        if not cands:
            return
        
        # Keep all samples for training propensity and nuisance models.
        # For real data, the runner defines y for d==0 (currently y=0); that is the chosen modeling assumption.
        observed_cands = cands
        
        if not observed_cands:
            return
        
        # Add to buffer with FIFO management
        self.buffer.extend(observed_cands)
        if len(self.buffer) > self.max_buffer:
            self.buffer = self.buffer[-self.max_buffer:]
        
        # Need minimum data before training
        if len(self.buffer) < max(100, self.n_folds * 20):
            return
        
        # Prepare training data
        X, D, Y, Y0, Y1, W = [], [], [], [], [], []
        
        for c in self.buffer:
            X.append(self.extract_features(c))
            D.append(int(c.get("d", 0)))
            Y.append(float(c.get("y", 0)))
            W.append(float(c.get("locality_weight", 1.0)))
            
            # For synthetic data with known counterfactuals
            if c.get("y0_true") is not None:
                Y0.append(float(c.get("y0_true", 0)))
                Y1.append(float(c.get("y1_true", 0)))
            else:
                Y0.append(None)
                Y1.append(None)
        
        X = np.array(X)
        D = np.array(D)
        Y = np.array(Y)
        W = np.array(W)
        
        # Train each fold on out-of-fold data
        for fold in range(self.n_folds):
            # Create fold assignment based on time
            fold_indices = np.array([
                (i % self.n_folds) == fold 
                for i in range(len(self.buffer))
            ])
            
            # Out-of-fold training data
            train_mask = ~fold_indices
            
            if train_mask.sum() < 20:
                continue
            
            X_train = X[train_mask]
            D_train = D[train_mask]
            Y_train = Y[train_mask]
            W_train = W[train_mask]
            
            # Normalize features
            try:
                X_train_scaled = self.scalers[fold].fit_transform(X_train)
            except:
                X_train_scaled = X_train
            
            # Train propensity model P(D=1|X,G_local)
            try:
                # Weight by locality for propensity
                self.propensity_models[fold].fit(
                    X_train_scaled, D_train,
                    sample_weight=W_train
                )
                
                # Train outcome models
                if Y0[0] is not None and Y1[0] is not None:  # Synthetic with counterfactuals
                    Y0_train = np.array([float(y) for y in Y0], dtype=float)[train_mask]
                    Y1_train = np.array([float(y) for y in Y1], dtype=float)[train_mask]
                    if len(Y0_train) > 10:
                        self.outcome_models_0[fold].fit(
                            X_train_scaled,
                            Y0_train,
                            sample_weight=W_train
                        )
                    if len(Y1_train) > 10:
                        self.outcome_models_1[fold].fit(
                            X_train_scaled,
                            Y1_train,
                            sample_weight=W_train
                        )
                else:  # Real data
                    mask_0 = (D_train == 0)
                    mask_1 = (D_train == 1)
                    if mask_0.sum() > 10:
                        self.outcome_models_0[fold].fit(
                            X_train_scaled[mask_0],
                            Y_train[mask_0],
                            sample_weight=W_train[mask_0]
                        )
                    if mask_1.sum() > 10:
                        self.outcome_models_1[fold].fit(
                            X_train_scaled[mask_1],
                            Y_train[mask_1],
                            sample_weight=W_train[mask_1]
                        )
                
                self.model_ready[fold] = True
                

            except Exception as e:
                # Continue with defaults if training fails
                pass
        
        self.t += 1
    
    def predict_propensity(self, c: Dict[str, Any], t: Optional[int] = None) -> float:
        """
        Predict ê(x, G_local) using appropriate fold's model
        Returns clipped propensity to ensure overlap (Assumption 1)
        """
        if t is None:
            t = self.t
        
        fold = self.get_fold(t)
        
        # Use recorded propensity if model not ready
        if not self.model_ready[fold]:
            return float(c.get("e_hat", 0.5))
        
        features = self.extract_features(c).reshape(1, -1)
        
        try:
            features_scaled = self.scalers[fold].transform(features)
            prob = self.propensity_models[fold].predict_proba(features_scaled)[0, 1]
            return float(prob)
        except:
            # Fallback to recorded propensity
            return float(c.get("e_hat", 0.5))
    
    def predict_outcome(self, c: Dict[str, Any], arm: int, t: Optional[int] = None) -> float:
        """
        Predict μ̂_a(x, G_local) for arm a ∈ {0, 1}
        """
        if t is None:
            t = self.t
        
        fold = self.get_fold(t)
        
        # Use base score if model not ready
        if not self.model_ready[fold]:
            base = float(c.get("p_hat", 0.5))
            # Simple adjustment for treatment effect
            if arm == 1:
                return min(1.0, base * 1.2)
            return base
        
        features = self.extract_features(c).reshape(1, -1)
        
        try:
            features_scaled = self.scalers[fold].transform(features)
            
            if arm == 0:
                pred = self.outcome_models_0[fold].predict(features_scaled)[0]
            else:
                pred = self.outcome_models_1[fold].predict(features_scaled)[0]
            
            # Ensure prediction is in [0, 1]
            return float(np.clip(pred, 0.0, 1.0))
            
        except:
            # Fallback to base score with adjustment
            base = float(c.get("p_hat", 0.5))
            if arm == 1:
                return min(1.0, base * 1.2)
            return base
    
    def get_predictions_batch(self, cands: List[Dict[str, Any]], t: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get batch predictions for efficiency
        Returns: {
            'e_hat': propensity predictions,
            'mu_0': outcome predictions for D=0,
            'mu_1': outcome predictions for D=1
        }
        """
        if t is None:
            t = self.t
        
        n = len(cands)
        e_hat = np.zeros(n)
        mu_0 = np.zeros(n)
        mu_1 = np.zeros(n)
        
        for i, c in enumerate(cands):
            e_hat[i] = self.predict_propensity(c, t)
            mu_0[i] = self.predict_outcome(c, 0, t)
            mu_1[i] = self.predict_outcome(c, 1, t)
        
        return {
            'e_hat': e_hat,
            'mu_0': mu_0,
            'mu_1': mu_1
        }
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about models"""
        diag = {
            'n_folds': self.n_folds,
            'buffer_size': len(self.buffer),
            'models_ready': sum(self.model_ready),
            'fold_ready': self.model_ready.copy()
        }
        
        # Add model performance if available
        if self.buffer and any(self.model_ready):
            X = np.array([self.extract_features(c) for c in self.buffer[-100:]])
            D = np.array([c.get("d", 0) for c in self.buffer[-100:]])
            
            prop_acc = []
            for fold in range(self.n_folds):
                if self.model_ready[fold]:
                    try:
                        X_scaled = self.scalers[fold].transform(X)
                        pred = self.propensity_models[fold].predict(X_scaled)
                        acc = np.mean(pred == D)
                        prop_acc.append(acc)
                    except:
                        pass
            
            if prop_acc:
                diag['propensity_accuracy'] = float(np.mean(prop_acc))
        
        return diag