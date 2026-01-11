"""
COPF Online Prequential Protocol Runner (Algorithm 2)
Complete implementation aligned with paper specifications
"""
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import torch
import os
import json
import math

from ..utils.logging import Logger, ensure_dir
from ..utils.seeds import set_seed
from ..utils.structures import LocalGraph
from ..data.tgb_stream import EventStream
from ..data.samplers import TemporalUniformSampler, KNNStructureSampler, RateMatchSampler
from ..models.base import BaseModel
from ..models.edgebank import EdgeBankOnline
from ..models.tgn import TGNMinimal
from ..models.tgn_fast import TGNFast
from .decision import decide_with_exploration
from .dr import GraphAwareDR, DRConfig
from .cross_fit import OnlineCrossFitter
from .audit import MultiCalibrator, AuditorConfig, ActiveAuditorSet
from .fairness import gCal, gTE, gMin, gRisk
from .eval import mrr, hits_at_k, average_precision, recall_at_k, ndcg_at_k
from .primal_dual import PrimalDualOptimizer

@dataclass
class RunnerArgs:
    """Arguments for COPF runner matching paper notation"""
    # Dataset
    dataset_name: str
    dataset_root: str
    
    # Sampling (OPP-1)
    sampler_name: str
    neg_factor: int
    neg_cap: Optional[int]
    knn_k: int
    
    # Decision (OPP-2)
    policy: str
    topk: int
    epsilon_min: float
    p_target: float
    temperature: float
    mc_samples: int
    
    # DR parameters (Section 5)
    dr_clip: float
    dr_self_norm: bool
    dr_decay_gamma: float
    dr_ratio_stab: bool
    dr_winsor_eps: float
    n_folds: int
    
    # Fairness (Definition 3)
    buckets_per_group: int
    auditor_budget_B: int
    isotonic_smoothing: bool
    tau_min_q: float
    fairness_mode: str
    
    # Primal-dual (Algorithm 1)
    pd_objective: str
    pd_gamma_p: float
    pd_gamma_i: float
    pd_ramp_start: float
    pd_ramp_end: float
    pd_update_every: int
    
    # Model
    model_name: str
    model_params: Dict[str, Any]
    
    # Deployment protocol (Definition 5)
    phase: str
    deploy_epsilon: float
    deploy_temperature: float
    
    # System
    seed: int
    device: str
    amp: bool
    max_time: int
    log_every: int
    out_dir: str

class CoverageTracker:
    """Track and enforce coverage for slices (Section 14.4)"""
    
    def __init__(self, buckets_per_group: int = 10):
        self.buckets = buckets_per_group
        self.exp_counts = defaultdict(int)
        self.seen_counts = defaultdict(int)
        self.total_steps = 0
    
    def _bucketize(self, p_vec: np.ndarray) -> np.ndarray:
        """Create equal-mass score buckets"""
        if len(p_vec) == 0:
            return np.array([], dtype=int)
        if p_vec.min() == p_vec.max():
            return np.zeros(len(p_vec), dtype=int)
        
        qs = np.linspace(0.0, 1.0, self.buckets + 1)
        edges = np.quantile(p_vec, qs)
        edges[0] = -1e9
        edges[-1] = 1e9
        
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9
        
        return np.digitize(p_vec, edges[1:-1], right=True)
    
    def compute_coverage_mask(self, cands: List[Dict], p_target: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute exploration mask for coverage-driven exploration (Section 14.4)
        ε_t(s) = max{ε_min, c·(p_tar - p̂_t(s))₊}
        """
        n = len(cands)
        if n == 0:
            return np.array([], dtype=bool), np.array([], dtype=int), np.array([], dtype=int)
        
        p_hat = np.array([float(c.get("p_hat", 0.5)) for c in cands])
        a = np.array([c.get("a", 0) for c in cands], dtype=int)
        pb = self._bucketize(p_hat)
        
        explore_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            slice_key = (int(a[i]), int(pb[i]))
            self.seen_counts[slice_key] += 1
            
            current_coverage = self.exp_counts[slice_key] / max(1, self.seen_counts[slice_key])
            
            if current_coverage < p_target:
                explore_mask[i] = True
        
        return explore_mask, pb, a
    
    def update(self, cands: List[Dict], decisions: List[int], pb: np.ndarray, a: np.ndarray):
        """Update coverage statistics after decisions"""
        self.total_steps += 1
        decisions = np.asarray(decisions, dtype=int)
        
        for i in range(len(cands)):
            slice_key = (int(a[i]), int(pb[i]))
            if decisions[i] == 1:
                self.exp_counts[slice_key] += 1

class NeighborhoodTracker:
    """Track neighborhood changes for locality weights (Definition 1)"""
    
    def __init__(self, decay_gamma: float = 0.1):
        self.decay_gamma = decay_gamma
        self.edge_times: Dict[Tuple[int, int], int] = {}
        self.node_last_change: Dict[int, int] = {}
        self.current_t = 0
    
    def add_edge(self, u: int, v: int, t: int):
        """Record edge addition time"""
        u, v, t = int(u), int(v), int(t)
        self.edge_times[(u, v)] = t
        self.edge_times[(v, u)] = t
        self.node_last_change[u] = t
        self.node_last_change[v] = t
        self.current_t = max(self.current_t, t)
    
    def compute_locality_weight(self, u: int, v: int, t: int) -> float:
        """
        Compute w_t(i,j) = exp(-γ * NeighborhoodChangeRate_t)
        """
        u, v, t = int(u), int(v), int(t)
        last_u = self.node_last_change.get(u, 0)
        last_v = self.node_last_change.get(v, 0)
        last_change = max(last_u, last_v)
        
        time_since = max(1, t - last_change)
        change_rate = 1.0 / time_since
        
        return float(np.exp(-self.decay_gamma * change_rate))

class OPPRunner:
    """
    Main COPF Runner implementing Algorithm 2 from the paper
    """
    
    def __init__(self, args: RunnerArgs):
        self.logger = Logger().get()
        self.args = args
        
        set_seed(args.seed)
        ensure_dir(args.out_dir)
        
        self.device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
        
        self.log_path = os.path.join(args.out_dir, "log.jsonl")
        self.log_f = open(self.log_path, "w")
        
        self.glocal = LocalGraph()
        self.neighborhood_tracker = NeighborhoodTracker(decay_gamma=args.dr_decay_gamma)
        
        self.sampler = self._make_sampler(args)
        self.model = self._make_model(args)
        
        self.cross_fitter = OnlineCrossFitter(
            n_folds=args.n_folds,
            seed=args.seed
        )
        
        self.dr = GraphAwareDR(
            config=DRConfig(
                clip=args.dr_clip,
                self_normalized=args.dr_self_norm,
                decay_gamma=args.dr_decay_gamma,
                ratio_stab=args.dr_ratio_stab,
                winsor_eps=args.dr_winsor_eps
            ),
            cross_fitter=self.cross_fitter,
            neighborhood_tracker=self.neighborhood_tracker
        )
        
        self.active_auditors = ActiveAuditorSet(budget=args.auditor_budget_B)
        
        self.auditor = MultiCalibrator(
            cfg=AuditorConfig(
                buckets_per_group=args.buckets_per_group,
                isotonic=args.isotonic_smoothing,
                budget_B=args.auditor_budget_B
            ),
            active_set=self.active_auditors
        )
        
        self.pd_optimizer = PrimalDualOptimizer(
            gamma_p=args.pd_gamma_p,
            gamma_i=args.pd_gamma_i,
            lr=args.pd_gamma_p / np.sqrt(args.max_time)
        )
        
        self.coverage_tracker = CoverageTracker(buckets_per_group=args.buckets_per_group)
        
        self._init_histories()
        
        self.current_phase = args.phase
        self.phase_results = {}
        self.total_steps = 0
        self.sat_steps = 0
    
    def _init_histories(self):
        """Initialize metric histories"""
        self.hist_t: List[int] = []
        self.hist_mrr: List[float] = []
        self.hist_ap: List[float] = []
        self.hist_hits: List[float] = []
        self.hist_recall: List[float] = []
        self.hist_ndcg: List[float] = []
        
        self.hist_t_fair: List[int] = []
        self.hist_gcal: List[float] = []
        self.hist_gte: List[float] = []
        self.hist_gmin: List[float] = []
        self.hist_grisk: List[float] = []
        
        self.hist_mean_e_nonsel: List[float] = []
        self.hist_clip_mass: List[float] = []
    
    def _make_sampler(self, args: RunnerArgs):
        """Create negative sampler (OPP-1)"""
        n = int(args.model_params.get("num_nodes", 10000))
        
        if args.sampler_name == "temporal_uniform":
            return TemporalUniformSampler(n, seed=args.seed)
        elif args.sampler_name == "knn_structure":
            return KNNStructureSampler(n, pool_limit=args.knn_k, seed=args.seed)
        elif args.sampler_name == "rate_match":
            return RateMatchSampler(n, seed=args.seed)
        else:
            raise ValueError(f"Unknown sampler: {args.sampler_name}")
    
    def _make_model(self, args: RunnerArgs) -> BaseModel:
        """Create base model"""
        if args.model_name == "edgebank":
            return EdgeBankOnline(self.device)
        elif args.model_name == "tgn":
            return TGNMinimal(
                self.device,
                num_nodes=int(args.model_params.get("num_nodes", 10000)),
                x_dim=int(args.model_params.get("x_dim", 8)),
                mem_dim=int(args.model_params.get("mem_dim", 128)),
                lr=float(args.model_params.get("lr", 1e-3)),
                amp=args.amp,
                n_folds=args.n_folds
            )
        elif args.model_name == "tgn_fast":
            return TGNFast(
                self.device,
                num_nodes=int(args.model_params.get("num_nodes", 10000)),
                x_dim=int(args.model_params.get("x_dim", 8)),
                mem_dim=int(args.model_params.get("mem_dim", 128)),
                lr=float(args.model_params.get("lr", 1e-3)),
                amp=args.amp,
                n_folds=args.n_folds
            )
        else:
            raise ValueError(f"Unknown model: {args.model_name}")
    
    def _compute_fairness_targets(self, t: int) -> Dict[str, float]:
        """Compute ramp schedule for fairness constraints (Section 14.3)"""
        T = max(1, self.args.max_time or t)
        frac = min(1.0, t / T)
        
        tol = self.args.pd_ramp_start + (self.args.pd_ramp_end - self.args.pd_ramp_start) * frac
        
        return {
            "gCal": tol,
            "gTE": tol,
            "gMin": tol,
            "gRisk": 0.1
        }
    
    def run_phase(self, stream: EventStream, phase: str, max_time: Optional[int] = None):
        """
        Run a single phase of the Pre→Deploy→Post protocol
        Implements Algorithm 2 from the paper
        """
        self.current_phase = phase
        self.logger.info(f"[PROTOCOL] Starting {phase.upper()} phase")
        
        if phase == "deploy":
            epsilon = self.args.deploy_epsilon
            temperature = self.args.deploy_temperature
        else:
            epsilon = self.args.epsilon_min
            temperature = self.args.temperature
        
        t = 0
        last_stream_t = None
        metrics_agg = defaultdict(list)
        
        for batch in stream:
            if last_stream_t is not None and int(batch.t) < int(last_stream_t):
                raise RuntimeError(f"[COPF] Non-monotone time: {batch.t} < {last_stream_t}")
            last_stream_t = batch.t
            
            t += 1
            self.total_steps += 1
            if max_time is not None and t > max_time:
                break
            
            # OPP-1: Online candidates
            P = batch.positives
            base_neg = self.args.neg_factor * max(1, len(P))
            
            SAFETY_MARGIN = 20
            need_neg = max(0, (self.args.topk + SAFETY_MARGIN) - len(P))
            k_req = max(base_neg, need_neg)
            
            if self.args.neg_cap is not None:
                k_req = min(k_req, int(self.args.neg_cap))
            
            N = self.sampler.sample(P, k=k_req)
            C = list(dict.fromkeys(P + N))
            
            if len(C) <= self.args.topk:
                self.sat_steps += 1
            
            # Build candidate dictionaries with features
            cand_dicts: List[Dict[str, Any]] = []
            for (u, v) in C:
                x = self.glocal.features(u, v)
                a = batch.meta.get("group", {}).get(u, 0)
                
                # Compute locality weight with int conversion
                w = self.neighborhood_tracker.compute_locality_weight(int(u), int(v), int(batch.t))
                
                cand_dicts.append({
                    "u": int(u), "v": int(v), "x": x, "a": a, 
                    "t": int(batch.t), "locality_weight": w
                })
            
            _rng = np.random.default_rng(self.args.seed + int(batch.t))
            _rng.shuffle(cand_dicts)
            
            max_idx = max(max(c["u"], c["v"]) for c in cand_dicts)
            if hasattr(self.model, "ensure_capacity"):
                self.model.ensure_capacity(max_idx + 1)
            if hasattr(self.sampler, "num_nodes") and self.sampler.num_nodes < (max_idx + 1):
                self.sampler.num_nodes = max_idx + 1
            
            # Predict scores
            scores = self.model.score(cand_dicts).detach().cpu().numpy()
            for c, s in zip(cand_dicts, scores):
                c["p_hat"] = float(np.clip(s, 0.0, 1.0))
            
            # Apply multicalibration if in COPF mode
            if self.args.fairness_mode == "copf":
                self.auditor.apply(cand_dicts)
            
            # Coverage-driven exploration
            explore_mask, pb, a_arr = self.coverage_tracker.compute_coverage_mask(
                cand_dicts, p_target=self.args.p_target
            )
            
            # Compute per-candidate exploration rates
            epsilon_vec = np.full(len(cand_dicts), epsilon)
            if phase == "pre":
                epsilon_vec = np.maximum(epsilon_vec, 0.05)
            
            for i, should_explore in enumerate(explore_mask):
                if should_explore:
                    epsilon_vec[i] = max(epsilon_vec[i], 0.2)
            
            # Make decisions with exploration
            d, e = decide_with_exploration(
                cand_dicts,
                policy=self.args.policy,
                topk=self.args.topk,
                epsilon=epsilon,
                epsilon_vec=epsilon_vec,  # Pass per-candidate rates
                temperature=temperature,
                explore_mask=explore_mask,
                mc_samples=int(self.args.mc_samples)
            )
            
            for i, (di, ei) in enumerate(zip(d, e)):
                cand_dicts[i]["d"] = int(di)
                cand_dicts[i]["e_hat"] = float(ei)
            
            self.coverage_tracker.update(cand_dicts, d, pb, a_arr)
            
            # Observe outcomes
            if self.args.dataset_name == "synthetic":
                for c in cand_dicts:
                    p0, p1 = self._synth_potential_outcomes(c, t=batch.t)
                    y0 = int(np.random.rand() < p0)
                    y1 = int(np.random.rand() < p1)
                    
                    c["y0_true"] = y0
                    c["y1_true"] = y1
                    
                    c["y"] = y1 if c["d"] == 1 else y0
                    c["y_true"] = int((c["u"], c["v"]) in set(P))
                    c["y_eval"] = c["y_true"]
            else:
                P_set = set(P)
                for c in cand_dicts:
                    is_tp = ((c["u"], c["v"]) in P_set)
                    c["y_true"] = 1 if is_tp else 0
                    c["y_eval"] = c["y_true"]
                    
                    if c["d"] == 1:
                        c["y"] = c["y_true"]
                    else:
                        c["y"] = 0
                    
                    c["y0_true"] = None
                    c["y1_true"] = None
            
            # Update graph structure
            for c in cand_dicts:
                if c["d"] == 1 and c["y"] == 1:
                    self.glocal.add_edge(c["u"], c["v"])
                    self.neighborhood_tracker.add_edge(c["u"], c["v"], int(batch.t))
                    if hasattr(self.sampler, "add_edge"):
                        self.sampler.add_edge(c["u"], c["v"])
            
            # Update DR with cross-fitting
            self.cross_fitter.update(cand_dicts)
            self.dr.ingest(cand_dicts)
            
            # Update multicalibration using DR residuals
            if self.args.fairness_mode == "copf":
                self.auditor.update_from_dr(self.dr)

            
            # Update active auditors
            if self.args.fairness_mode == "copf" and t % 100 == 0:
                violations = self.auditor.compute_violations(cand_dicts, self.dr)
                self.active_auditors.update(violations)
            
            # Model update
            train_batch = [c for c in cand_dicts if c["d"] == 1]
            
            if train_batch:
                if self.args.fairness_mode == "copf" and self.args.pd_objective == "constrained":
                    loss_weights = self.pd_optimizer.get_loss_weights(train_batch)
                else:
                    loss_weights = None
                
                if isinstance(self.model, EdgeBankOnline):
                    self.model.update(train_batch)
                else:
                    self.model.update(train_batch, loss_weights=loss_weights)
            
            # Evaluate fairness
            if (t % self.args.pd_update_every) == 0 and self.args.fairness_mode == "copf":
                bucket_batch = list(self.dr.buffer[-20000:]) if self.dr.buffer else []
                groups = sorted(list({c.get("a", 0) for c in self.dr.buffer})) if self.dr.buffer else [0]
                
                gcal_map = gCal(
                    bucket_batch, self.dr, groups=groups,
                    buckets_per_group=self.args.buckets_per_group,
                    isotonic=self.args.isotonic_smoothing,
                    arm_for_cal=0
                )
                gcal_max = float(max(gcal_map.values()) if gcal_map else 0.0)
                
                gte_gap = float(gTE(self.dr, groups=groups))
                
                te_by_group = self.dr.estimate_TE_by_group(group_key="a")
                te_vals = [v for (v, n) in te_by_group.values() if n > 0]
                tau_min = float(np.quantile(te_vals, self.args.tau_min_q) if te_vals else 0.0)
                tau_min = max(0.0, tau_min)
                gmin_val = float(gMin(self.dr, tau_min=tau_min))
                
                grisk_val = float(gRisk(self.dr, groups=groups))
                
                self.hist_t_fair.append(t)
                self.hist_gcal.append(gcal_max)
                self.hist_gte.append(gte_gap)
                self.hist_gmin.append(gmin_val)
                self.hist_grisk.append(grisk_val)
                
                if self.args.pd_objective == "constrained":
                    beta = float(self.auditor.conf_radius())
                    soft = lambda x: x / (1.0 + beta)
                    
                    violations = {
                        "gCal": soft(gcal_max),
                        "gTE": soft(gte_gap),
                        "gMin": soft(gmin_val),
                        "gRisk": soft(grisk_val)
                    }
                    
                    targets = self._compute_fairness_targets(t)
                    self.pd_optimizer.update_duals(violations, targets)
            
            # Compute utility metrics
            m = mrr(cand_dicts)
            a = average_precision(cand_dicts)
            h = hits_at_k(cand_dicts, k=self.args.topk)
            r = recall_at_k(cand_dicts, k=self.args.topk)
            n = ndcg_at_k(cand_dicts, k=self.args.topk)
            
            metrics_agg["mrr"].append(m)
            metrics_agg["ap"].append(a)
            metrics_agg[f"hits@{self.args.topk}"].append(h)
            metrics_agg["recall"].append(r)
            metrics_agg["ndcg"].append(n)
            
            self.hist_t.append(t)
            self.hist_mrr.append(m)
            self.hist_ap.append(a)
            self.hist_hits.append(h)
            self.hist_recall.append(r)
            self.hist_ndcg.append(n)
            
            if (t % self.args.log_every) == 0:
                self._log_step(t, phase, cand_dicts, metrics_agg)
        
        return self._phase_summary(phase, metrics_agg)
    
    def _synth_potential_outcomes(self, c: Dict[str, Any], t: int) -> Tuple[float, float]:
        """Generate synthetic potential outcomes for testing"""
        u, v = int(c["u"]), int(c["v"])
        a = int(c.get("a", 0))
        
        phi = np.array([
            (u % 97) / 97.0,
            (v % 89) / 89.0,
            math.sin(0.01 * t),
            math.cos(0.01 * t),
        ])
        
        w0 = np.array([0.8, -0.6, 0.5, -0.3])
        b0 = -1.2 + 0.15 * (a == 1)
        p0 = 1.0 / (1.0 + math.exp(-(w0 @ phi + b0)))
        
        w1 = np.array([0.4, 0.4, 0.2, 0.2])
        b1 = -0.2
        delta = 0.25 * (1.0 / (1.0 + math.exp(-(w1 @ phi + b1))))
        p1 = min(1.0, max(0.0, p0 + delta))
        
        return p0, p1
    
    def _log_step(self, t: int, phase: str, cands: List[Dict], metrics_agg: Dict):
        """Log step statistics"""
        p_arr = np.array([c.get("p_hat", 0.5) for c in cands])
        e_arr = np.array([c.get("e_hat", 0.5) for c in cands])
        
        rec = {
            "phase": phase,
            "t": t,
            "metrics": {k: float(np.mean(v)) for k, v in metrics_agg.items()},
            "diag": {
                "p_q10": float(np.quantile(p_arr, 0.10)),
                "p_q50": float(np.quantile(p_arr, 0.50)),
                "p_q90": float(np.quantile(p_arr, 0.90)),
                "e_q10": float(np.quantile(e_arr, 0.10)),
                "e_q90": float(np.quantile(e_arr, 0.90)),
                "cand_size": int(len(cands)),
            }
        }
        
        self.log_f.write(json.dumps(rec) + "\n")
        self.log_f.flush()
        
        self.logger.info(
            f"{phase} @t={t} "
            f"mrr={rec['metrics']['mrr']:.4f} "
            f"ap={rec['metrics']['ap']:.4f} "
            f"hits@{self.args.topk}={rec['metrics'].get(f'hits@{self.args.topk}', 0):.4f} "
            f"recall={rec['metrics']['recall']:.4f} "
            f"ndcg={rec['metrics']['ndcg']:.4f}"
        )
    
    def _phase_summary(self, phase: str, metrics_agg: Dict) -> Dict:
        """Compute phase summary statistics"""
        bucket_batch = list(self.dr.buffer[-20000:]) if self.dr.buffer else []
        groups = sorted(list({c.get("a", 0) for c in self.dr.buffer})) if self.dr.buffer else [0]
        
        gcal_map = gCal(
            bucket_batch, self.dr, groups=groups,
            buckets_per_group=self.args.buckets_per_group,
            isotonic=self.args.isotonic_smoothing,
            arm_for_cal=0
        )
        gcal_max = float(max(gcal_map.values()) if gcal_map else 0.0)
        gte_gap = float(gTE(self.dr, groups=groups))
        
        te_by_group = self.dr.estimate_TE_by_group(group_key="a")
        te_vals = [v for (v, n) in te_by_group.values() if n > 0]
        tau_min = float(np.quantile(te_vals, self.args.tau_min_q) if te_vals else 0.0)
        gmin_val = float(gMin(self.dr, tau_min=max(0.0, tau_min)))
        grisk_val = float(gRisk(self.dr, groups=groups))
        
        util = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics_agg.items()}
        
        tau_by_group = {
            int(s if s is not None else -1): {"tau": float(v), "n_eff": int(n)}
            for s, (v, n) in te_by_group.items()
        }
        
        fair = {
            "gCal_max": gcal_max,
            "gTE_gap": gte_gap,
            "gMin": gmin_val,
            "gRisk": grisk_val,
            "tau_min": tau_min,
            "groups": groups,
            "tau_by_group": tau_by_group,
            "tau_avg": float(np.mean([d["tau"] for d in tau_by_group.values()])) if tau_by_group else 0.0,
        }
        
        history = {
            "t": self.hist_t,
            "mrr": self.hist_mrr,
            "ap": self.hist_ap,
            f"hits@{self.args.topk}": self.hist_hits,
            f"hits{self.args.topk}": self.hist_hits,  # Both formats
            "recall": self.hist_recall,
            "ndcg": self.hist_ndcg,
            "tfair": self.hist_t_fair,
            "gCal_max": self.hist_gcal,
            "gTE_gap": self.hist_gte,
            "gMin": self.hist_gmin,
            "gRisk": self.hist_grisk,
        }
        
        self.phase_results[phase] = {"utility": util, "fairness": fair, "history": history}
        
        self.logger.info(f"[{phase.upper()} SUMMARY]")
        self.logger.info(f"  Utility: MRR={util['mrr']:.4f} AP={util['ap']:.4f}")
        self.logger.info(f"  Fairness: gCal={fair['gCal_max']:.4f} gTE={fair['gTE_gap']:.4f} gMin={fair['gMin']:.4f}")
        self.logger.info(f"  Saturation: {self.sat_steps}/{self.total_steps} = {self.sat_steps/max(1,self.total_steps):.1%}")
        
        return {"utility": util, "fairness": fair, "history": history}
    
    def run_pre_deploy_post(self, stream: EventStream) -> Tuple[Dict, Dict]:
        """Run complete Pre→Deploy→Post protocol (Definition 5)"""
        results = {}
        
        total_time = self.args.max_time
        phase_times = [
            int(total_time * 0.33),
            int(total_time * 0.33),
            int(total_time * 0.34),
        ]
        
        results["pre"] = self.run_phase(stream, "pre", max_time=phase_times[0])
        results["deploy"] = self.run_phase(stream, "deploy", max_time=phase_times[1])
        results["post"] = self.run_phase(stream, "post", max_time=phase_times[2])
        
        drift = self._compute_deployment_drift(results)
        
        return results, drift
    
    def _compute_deployment_drift(self, results: Dict) -> Dict:
        """Compute fairness drift and effect preservation"""
        pre = results["pre"]["fairness"]
        post = results["post"]["fairness"]
        
        drift = {
            "gCal_drift": abs(post["gCal_max"] - pre["gCal_max"]),
            "gTE_drift": abs(post["gTE_gap"] - pre["gTE_gap"]),
            "gMin_drift": abs(post["gMin"] - pre["gMin"]),
            "effect_preservation": abs(post["tau_avg"] - pre["tau_avg"]),
            "max_drift": max(
                abs(post["gCal_max"] - pre["gCal_max"]),
                abs(post["gTE_gap"] - pre["gTE_gap"]),
                abs(post["gMin"] - pre["gMin"])
            )
        }
        
        self.logger.info("[DEPLOYMENT STABILITY]")
        self.logger.info(f"  gCal drift: {drift['gCal_drift']:.4f}")
        self.logger.info(f"  gTE drift: {drift['gTE_drift']:.4f}")
        self.logger.info(f"  gMin drift: {drift['gMin_drift']:.4f}")
        self.logger.info(f"  Effect preservation: {drift['effect_preservation']:.4f}")
        self.logger.info(f"  Max drift: {drift['max_drift']:.4f}")
        
        return drift
    
    def close(self):
        """Clean up resources"""
        try:
            self.log_f.close()
        except Exception:
            pass















