from __future__ import annotations
import argparse
import os
import json
import numpy as np
import pandas as pd

from copf.decision import decide_with_exploration
from copf.dr import GraphAwareDR, DRConfig
from copf.audit import MultiCalibrator, AuditorConfig, ActiveAuditorSet
from copf.fairness import gCal, gTE, gMin, gRisk
from copf.eval import mrr, hits_at_k, average_precision

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)   # contains edges.csv, nodes.csv, meta.json
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--T", type=int, default=50000)          # number of events to run
    ap.add_argument("--neg", type=int, default=200)          # negatives per event
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--policy", type=str, default="topk_stochastic", choices=["topk_stochastic","epsilon_greedy"])
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    edges = pd.read_csv(os.path.join(args.data_dir, "edges.csv"))
    nodes = pd.read_csv(os.path.join(args.data_dir, "nodes.csv"))
    meta_path = os.path.join(args.data_dir, "meta.json")
    meta = json.load(open(meta_path, "r")) if os.path.exists(meta_path) else {}

    # infer bipartite split
    n_users = int(meta.get("n_users", int((nodes["group"] >= 0).sum())))
    num_nodes = int(meta.get("num_nodes", int(nodes["node"].max() + 1)))
    n_items = int(meta.get("n_items", num_nodes - n_users))

    user_group = nodes.set_index("node")["group"].to_dict()
    item_ids = np.arange(n_users, n_users + n_items, dtype=np.int64)

    # simple online “model”: item popularity + tiny noise
    item_deg = np.ones(n_items, dtype=np.float64)

    dr = GraphAwareDR(DRConfig(clip=0.05, self_normalized=True, decay_gamma=0.0, winsor_eps=0.02, min_eff_samples=200))

    auditor_cfg = AuditorConfig(buckets_per_group=10, isotonic=True, budget_B=64, step=0.25, min_mass=0.02)
    auditor = MultiCalibrator(cfg=auditor_cfg, active_set=ActiveAuditorSet(budget=auditor_cfg.budget_B))

    # init bucket edges for groups (needed for apply())
    groups = sorted(set(int(g) for g in nodes["group"].unique() if int(g) >= 0))
    # start with uniform buckets in [0,1]
    from fl_utils.buckets import equal_mass_buckets
    auditor.edges_by_group = {g: equal_mass_buckets(np.linspace(0,1,1001), auditor_cfg.buckets_per_group) for g in groups}
    # make all auditors active at start (otherwise apply() does nothing)
    auditor.active_set.active_auditors = set((g, b) for g in groups for b in range(auditor_cfg.buckets_per_group))

    rows = []
    T = min(int(args.T), len(edges))

    for idx in range(T):
        u = int(edges.iloc[idx]["src"])
        v_true = int(edges.iloc[idx]["dst"])
        t = int(edges.iloc[idx]["t"])
        a = int(user_group.get(u, 0))

        # sample negatives among items
        # (ensure we sample item-side nodes, not users)
        neg_vs = rng.choice(item_ids, size=min(args.neg, len(item_ids)), replace=False)
        # ensure true item included
        cand_vs = np.unique(np.concatenate([[v_true], neg_vs])).astype(np.int64)

        # build candidate dicts
        cands = []
        for v in cand_vs:
            is_pos = int(v == v_true)
            j = int(v - n_users)  # item index 0..n_items-1

            # simple score = sigmoid(log(deg))
            score = sigmoid(np.log(item_deg[j] + 1.0) + 0.05 * rng.normal())

            cands.append({
                "u": u, "v": int(v), "t": t,
                "a": a,
                "p_hat": float(np.clip(score, 1e-3, 1-1e-3)),
                "y_true": is_pos,
                # synthetic potential outcomes (simple): control=0, treated=y_true
                "y0_true": 0.0,
                "y1_true": float(is_pos),
                # optional plug-in fallback for DR
                "mu0_hat": float(np.clip(score, 1e-3, 1-1e-3)),
                "mu1_hat": float(np.clip(min(0.999, 1.2 * score), 1e-3, 1-1e-3)),
            })

        # multicalibration offset (pre-decision)
        auditor.apply(cands)

        # decide + log propensity
        decisions, prop = decide_with_exploration(
            cands=cands,
            policy=args.policy,
            topk=args.topk,
            epsilon=args.epsilon,
            temperature=args.temperature,
            mc_samples=128,
            rng=rng
        )
        for c, d, e in zip(cands, decisions, prop):
            c["d"] = int(d)
            c["e_hat"] = float(np.clip(e, 1e-6, 1.0))
            # observed outcome: only see label if exposed
            c["y"] = float(c["y_true"]) if int(d) == 1 else 0.0

        # update DR buffer
        dr.ingest(cands)

        # update calibrator using residuals (your audit.py 里自带逻辑)
        auditor.calibrate(cands, dr=dr, prefer_arm=0, dr_clip=0.02)

        # update popularity after “true edge” happens (stream dynamics)
        # (this is just to keep scores non-degenerate)
        item_deg[int(v_true - n_users)] += 1.0

        if (idx + 1) % args.log_every == 0:
            util = {
                "mrr": mrr(cands),
                "ap": average_precision(cands),
                "hits10": hits_at_k(cands, 10),
            }
            fair_cal = gCal(batch=cands, dr=dr, groups=groups, buckets_per_group=10, isotonic=False, arm_for_cal=0)
            fair = {
                "gCal_max": float(max(fair_cal.values()) if fair_cal else 0.0),
                "gTE_gap": gTE(dr, groups),
                "gMin": gMin(dr, tau_min=0.0),
                "gRisk": gRisk(dr, groups),
            }
            row = {"step": idx + 1, **util, **fair}
            rows.append(row)
            print(f"[{idx+1}/{T}] " + " ".join([f"{k}={row[k]:.4f}" for k in ["mrr","ap","hits10","gCal_max","gTE_gap","gMin"]]))

    out_csv = os.path.join(args.out_dir, "smoke_metrics.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

if __name__ == "__main__":
    main()