from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- Gradient Reversal (for adversarial debiasing) --------
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


# -------- Config --------
@dataclass
class TGNConfig:
    num_nodes: int
    emb_dim: int = 128
    msg_dim: int = 128

    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"
    train_every: int = 1

    # fairness-on-top switches
    adv_lambda: float = 0.0
    n_groups: int = 0  # required if adv_lambda>0

    reweight: str = "none"  # ["none", "inv_freq"]
    fair_penalty_lambda: float = 0.0

    seed: int = 42


# -------- Minimal TGN-style Online Link Predictor --------
class _SimpleTGN(nn.Module):
    """
    Minimal TGN-style model:
      - per-node memory vector m_v
      - message from (u_mem, v_mem, dt_embed)
      - GRU update on endpoints
      - link score = dot(Wu(m_u), Wv(m_v))
    """
    def __init__(self, num_nodes: int, emb_dim: int, msg_dim: int):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.emb_dim = int(emb_dim)
        self.msg_dim = int(msg_dim)

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * emb_dim + emb_dim, msg_dim),
            nn.ReLU(),
            nn.Linear(msg_dim, msg_dim),
        )
        self.gru = nn.GRUCell(msg_dim, emb_dim)

        self.proj = nn.Linear(emb_dim, emb_dim, bias=False)

        # simple time encoding (dt -> emb_dim) using log1p + MLP
        self.dt_mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward_score(self, mem: torch.Tensor, u: int, v_list: torch.Tensor) -> torch.Tensor:
        uvec = mem[u].unsqueeze(0)                      # [1, D]
        vvec = mem[v_list]                             # [K, D]
        su = self.proj(uvec)                           # [1, D]
        sv = self.proj(vvec)                           # [K, D]
        logits = (sv * su).sum(dim=-1)                 # [K]
        return logits

    def build_message(self, mem_u: torch.Tensor, mem_v: torch.Tensor, dt_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([mem_u, mem_v, dt_emb], dim=-1)
        return self.msg_mlp(x)

    def update_endpoints(
        self,
        mem: torch.Tensor,
        last_t: torch.Tensor,
        u: int,
        v: int,
        t: int,
    ) -> None:
        # compute dt embeddings
        tu = last_t[u].item()
        tv = last_t[v].item()
        dt_u = float(max(0.0, t - tu))
        dt_v = float(max(0.0, t - tv))

        dt_u_emb = self.dt_mlp(torch.tensor([[np.log1p(dt_u)]], device=mem.device, dtype=mem.dtype))
        dt_v_emb = self.dt_mlp(torch.tensor([[np.log1p(dt_v)]], device=mem.device, dtype=mem.dtype))

        mu = mem[u].unsqueeze(0)
        mv = mem[v].unsqueeze(0)

        msg_u = self.build_message(mu, mv, dt_u_emb)   # message to u uses (u,v)
        msg_v = self.build_message(mv, mu, dt_v_emb)   # message to v uses (v,u)

        new_mu = self.gru(msg_u, mu).squeeze(0)
        new_mv = self.gru(msg_v, mv).squeeze(0)

        mem[u] = new_mu
        mem[v] = new_mv
        last_t[u] = float(t)
        last_t[v] = float(t)


class TGNWrapper:
    """
    Online wrapper with API:
      - score(u, v_list, t, src_group=...)
      - update(u, pos_v, neg_vs, t, src_group=...)

    Supports:
      - vanilla link loss
      - adversarial debiasing head (predict group from user memory)
      - reweight (pos loss by group weights)
      - simple surrogate fairness penalty on EMA(pos_prob) across groups
    """
    def __init__(self, cfg: TGNConfig):
        self.cfg = cfg
        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))

        self.device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")

        self.model = _SimpleTGN(cfg.num_nodes, cfg.emb_dim, cfg.msg_dim).to(self.device)

        # node memories + last timestamps (buffers, but trainable memory updates happen via explicit assignment)
        self.mem = torch.zeros(cfg.num_nodes, cfg.emb_dim, device=self.device)
        self.last_t = torch.zeros(cfg.num_nodes, device=self.device)

        # adversarial head
        self.adv_lambda = float(cfg.adv_lambda)
        self.n_groups = int(cfg.n_groups)
        if self.adv_lambda > 0.0:
            if self.n_groups <= 1:
                raise ValueError("adv_lambda>0 requires n_groups>=2 (provide group labels in dataset or --n_groups).")
            self.group_head = nn.Sequential(
                nn.Linear(cfg.emb_dim, cfg.emb_dim),
                nn.ReLU(),
                nn.Linear(cfg.emb_dim, self.n_groups),
            ).to(self.device)
        else:
            self.group_head = None

        self.opt = torch.optim.Adam(
            list(self.model.parameters()) + ([] if self.group_head is None else list(self.group_head.parameters())),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

        # reweighting
        self.reweight = str(cfg.reweight)
        self.group_counts: Dict[int, int] = {}
        self.group_weights: Dict[int, float] = {}

        # fairness surrogate EMA tracking
        self.fair_penalty_lambda = float(cfg.fair_penalty_lambda)
        self.ema_pos_prob_by_group: Dict[int, float] = {}
        self.ema_alpha = 0.01  # slow EMA
        self.step = 0
        self.last_loss = float("nan")

    @torch.no_grad()
    def score(self, u: int, v_list: List[int], t: int, src_group: int = 0) -> np.ndarray:
        v = torch.tensor(v_list, device=self.device, dtype=torch.long)
        logits = self.model.forward_score(self.mem, int(u), v)  # [K]
        return logits.detach().cpu().numpy()

    def _update_group_stats(self, g: int) -> None:
        self.group_counts[g] = int(self.group_counts.get(g, 0) + 1)
        if self.reweight == "inv_freq":
            total = float(sum(self.group_counts.values()))
            # inverse frequency weights normalized to mean ~ 1
            inv = {gg: total / max(1.0, float(cc)) for gg, cc in self.group_counts.items()}
            mean_inv = float(np.mean(list(inv.values()))) if inv else 1.0
            self.group_weights = {gg: float(w / mean_inv) for gg, w in inv.items()}
        else:
            self.group_weights = {}

    def _fair_surrogate_penalty(self, g: int, p_pos: torch.Tensor) -> torch.Tensor:
        """
        Simple fairness surrogate: keep EMA(pos_prob) similar across groups.
        penalty ~ (ema_g - ema_global)^2
        """
        if self.fair_penalty_lambda <= 0.0 or self.n_groups <= 1:
            return torch.tensor(0.0, device=self.device)

        # update EMA buffers (detach to avoid weird gradients)
        p = float(p_pos.detach().item())
        old = float(self.ema_pos_prob_by_group.get(g, p))
        new = (1.0 - self.ema_alpha) * old + self.ema_alpha * p
        self.ema_pos_prob_by_group[g] = new

        # global mean EMA
        vals = list(self.ema_pos_prob_by_group.values())
        if len(vals) <= 1:
            return torch.tensor(0.0, device=self.device)
        global_mean = float(np.mean(vals))
        return (torch.tensor(new, device=self.device) - torch.tensor(global_mean, device=self.device)) ** 2

    def update(self, u: int, pos_v: int, neg_vs: List[int], t: int, src_group: int = 0) -> None:
        self.step += 1
        u = int(u); pos_v = int(pos_v); t = int(t)
        g = int(src_group)

        # group stats for reweight
        self._update_group_stats(g)
        w_pos = float(self.group_weights.get(g, 1.0))

        # build tensors
        v_all = torch.tensor([pos_v] + [int(x) for x in neg_vs], device=self.device, dtype=torch.long)
        logits = self.model.forward_score(self.mem, u, v_all)  # [1+K]

        y = torch.zeros_like(logits)
        y[0] = 1.0

        # BCE with logits; weight pos example if needed
        # weight vector: [w_pos, 1,1,...]
        weights = torch.ones_like(logits)
        weights[0] = float(w_pos)

        link_loss = F.binary_cross_entropy_with_logits(logits, y, weight=weights)

        # adversarial debias: predict group from user memory via GRL
        adv_loss = torch.tensor(0.0, device=self.device)
        if self.group_head is not None and self.adv_lambda > 0.0:
            u_rep = self.mem[u].unsqueeze(0)  # [1,D]
            u_rep_grl = grad_reverse(u_rep, self.adv_lambda)
            g_logits = self.group_head(u_rep_grl)  # [1,G]
            g_target = torch.tensor([g], device=self.device, dtype=torch.long).clamp(0, self.n_groups - 1)
            adv_loss = F.cross_entropy(g_logits, g_target)

        # fairness surrogate penalty (optional)
        p_pos = torch.sigmoid(logits[0])
        fair_pen = self._fair_surrogate_penalty(g, p_pos)
        total_loss = link_loss + (self.fair_penalty_lambda * fair_pen) + adv_loss

        if self.cfg.train_every <= 1 or (self.step % int(self.cfg.train_every) == 0):
            self.opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.group_head is not None:
                torch.nn.utils.clip_grad_norm_(self.group_head.parameters(), 1.0)
            self.opt.step()
            self.last_loss = float(total_loss.detach().item())

        # after observing positive edge, update memory of endpoints (TGN-style)
        with torch.no_grad():
            self.model.update_endpoints(self.mem, self.last_t, u, pos_v, t)

    def diagnostics(self) -> Dict[str, float]:
        d = {
            "last_loss": float(self.last_loss),
            "mem_norm_mean": float(self.mem.norm(dim=1).mean().detach().cpu().item()),
            "n_groups_seen": float(len(self.group_counts)),
        }
        if self.reweight == "inv_freq" and self.group_weights:
            d["w_pos_max"] = float(max(self.group_weights.values()))
            d["w_pos_min"] = float(min(self.group_weights.values()))
        else:
            d["w_pos_max"] = 1.0
            d["w_pos_min"] = 1.0
        return d