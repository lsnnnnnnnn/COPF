import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def stable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax that returns a valid probability vector."""
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=-1e30, posinf=1e30, neginf=-1e30)
    x = x - np.max(x)  # stability
    ex = np.exp(x)
    ex = np.nan_to_num(ex, nan=0.0, posinf=0.0, neginf=0.0)
    s = float(ex.sum())
    if (not np.isfinite(s)) or (s <= 0.0):
        p = np.full_like(ex, 1.0 / len(ex), dtype=np.float64)
    else:
        p = ex / s
        # Force exact sum=1 for numpy choice (avoid floating error)
        p[-1] = 1.0 - float(p[:-1].sum())
        if (not np.isfinite(p[-1])) or (p[-1] <= 0.0):
            p = np.full_like(ex, 1.0 / len(ex), dtype=np.float64)
    return p


def normalize_probs(p: np.ndarray) -> np.ndarray:
    """Safe normalize for any nonnegative vector."""
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if (not np.isfinite(s)) or (s <= 0.0):
        p = np.full_like(p, 1.0 / len(p), dtype=np.float64)
    else:
        p = p / s
        p[-1] = 1.0 - float(p[:-1].sum())
        if (not np.isfinite(p[-1])) or (p[-1] <= 0.0):
            p = np.full_like(p, 1.0 / len(p), dtype=np.float64)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_users", type=int, default=600)
    ap.add_argument("--n_items", type=int, default=4000)
    ap.add_argument("--n_events", type=int, default=200_000)
    ap.add_argument("--group_p", type=float, default=0.5, help="P(group=1) for users")
    ap.add_argument("--seed", type=int, default=42)

    # dynamics knobs
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--cand_pool", type=int, default=200, help="candidate items per event (for efficient sampling)")
    ap.add_argument("--alpha_pop", type=float, default=0.75, help="popularity exponent")
    ap.add_argument(
        "--group_bias",
        type=float,
        default=0.6,
        help="bias favoring group=1 users (positive -> higher interaction propensity)",
    )
    ap.add_argument("--drift_scale", type=float, default=0.002, help="time drift per event (small)")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_users = args.n_users
    n_items = args.n_items
    T = args.n_events

    # ---- user groups (sensitive attribute proxy) ----
    user_group = rng.binomial(1, args.group_p, size=n_users).astype(int)

    # ---- latent factors ----
    U = rng.normal(0, 1.0, size=(n_users, args.latent_dim)).astype(np.float32)
    V = rng.normal(0, 1.0, size=(n_items, args.latent_dim)).astype(np.float32)

    # ---- item popularity state ----
    item_deg = np.ones(n_items, dtype=np.float64)  # start with 1 to avoid zeros

    # ---- generate event stream ----
    src_users = rng.integers(0, n_users, size=T, dtype=np.int64)

    # strictly increasing time
    dt = rng.integers(1, 5, size=T, dtype=np.int64)
    t = np.cumsum(dt)  # 1.. increasing

    dst_items = np.empty(T, dtype=np.int64)

    # For speed, we only score a candidate pool each event.
    for idx in range(T):
        u = int(src_users[idx])

        # sample candidate items biased by popularity
        pop_w = item_deg ** args.alpha_pop
        pop_p = normalize_probs(pop_w)

        cand = rng.choice(
            n_items,
            size=min(args.cand_pool, n_items),
            replace=False,
            p=pop_p,
        )

        # score candidates (latent dot + group bias + small drift)
        g = int(user_group[u])
        drift = args.drift_scale * idx
        logits = (V[cand] @ U[u]).astype(np.float64) + (args.group_bias * (2 * g - 1)) - drift

        # sample one item using softmax(logits) (more appropriate for categorical choice)
        p = stable_softmax(logits)
        j = int(rng.choice(cand, p=p))

        dst_items[idx] = j
        item_deg[j] += 1.0

        # Optional: occasional progress
        # if (idx + 1) % 50000 == 0:
        #     print(f"[gen] {idx+1}/{T}")

    # Map bipartite node ids into one contiguous namespace:
    # users: [0, n_users-1], items: [n_users, n_users+n_items-1]
    src = src_users
    dst = dst_items + n_users
    num_nodes = n_users + n_items

    edges = pd.DataFrame({"src": src, "dst": dst, "t": t})

    # Nodes table: give group for users; items get group = -1 (ignored) by default
    node = np.arange(num_nodes, dtype=np.int64)
    group = np.full(num_nodes, -1, dtype=np.int64)
    group[:n_users] = user_group

    nodes = pd.DataFrame({"node": node, "group": group})

    edges_path = out_dir / "edges.csv"
    nodes_path = out_dir / "nodes.csv"
    meta_path = out_dir / "meta.json"

    edges.to_csv(edges_path, index=False)
    nodes.to_csv(nodes_path, index=False)

    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "num_nodes": int(num_nodes),
        "n_events": int(T),
        "group_p": float(args.group_p),
        "latent_dim": int(args.latent_dim),
        "cand_pool": int(args.cand_pool),
        "alpha_pop": float(args.alpha_pop),
        "group_bias": float(args.group_bias),
        "drift_scale": float(args.drift_scale),
        "seed": int(args.seed),
        "notes": "users have group in {0,1}; items group = -1 (ignore). src-side fairness only.",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[OK] wrote {edges_path}")
    print(f"[OK] wrote {nodes_path}")
    print(f"[OK] wrote {meta_path}")


if __name__ == "__main__":
    main()

