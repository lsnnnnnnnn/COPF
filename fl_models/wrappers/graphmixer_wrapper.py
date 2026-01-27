from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import inspect
import numpy as np
import torch
import torch.nn as nn

# IMPORTANT: this imports TGB baseline GraphMixer because TGB_ROOT is in sys.path
from models.GraphMixer import GraphMixer  # tgb_baselines/models/GraphMixer.py


@dataclass
class GraphMixerConfig:
    num_nodes: int
    emb_dim: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"
    seed: int = 42

    # some repos require extra args; keep them here if needed
    # num_layers: int = 2
    # dropout: float = 0.1


class GraphMixerWrapper:
    """
    Online wrapper for TGB GraphMixer:
      - score(u, [v...], t) -> prob scores
      - update(u, pos_v, neg_vs, t) -> one optimizer step on BCE
    """

    def __init__(self, cfg: GraphMixerConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.model = self._build_model(cfg).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.bce_logits = nn.BCEWithLogitsLoss()

        self.last_loss: float = float("nan")

    def _build_model(self, cfg: GraphMixerConfig):
        """
        Different GraphMixer implementations have different ctor signatures.
        We try a few common patterns; if all fail, raise with signature.
        """
        sig = inspect.signature(GraphMixer.__init__)
        kwargs_try = [
            dict(num_nodes=cfg.num_nodes, emb_dim=cfg.emb_dim),
            dict(num_nodes=cfg.num_nodes, node_num=cfg.num_nodes, emb_dim=cfg.emb_dim),
            dict(n_nodes=cfg.num_nodes, emb_dim=cfg.emb_dim),
            dict(num_nodes=cfg.num_nodes),
            dict(),
        ]
        for kw in kwargs_try:
            try:
                return GraphMixer(**kw)
            except TypeError:
                continue

        raise TypeError(
            f"Failed to instantiate TGB GraphMixer. Constructor signature is: {sig}. "
            f"Open tgb_baselines/models/GraphMixer.py and adjust GraphMixerConfig/_build_model."
        )

    @torch.no_grad()
    def score(self, u: int, v_list: List[int], t: int, src_group: int = 0) -> np.ndarray:
        self.model.eval()
        u_t = torch.tensor([u] * len(v_list), device=self.device, dtype=torch.long)
        v_t = torch.tensor(v_list, device=self.device, dtype=torch.long)
        ts_t = torch.tensor([t] * len(v_list), device=self.device, dtype=torch.long)

        out = self._forward(u_t, v_t, ts_t)

        out = out.detach().view(-1)
        # convert to prob if it looks like logits
        if torch.min(out) < 0.0 or torch.max(out) > 1.0:
            out = torch.sigmoid(out)
        return out.cpu().numpy()

    def update(self, u: int, pos_v: int, neg_vs: List[int], t: int, src_group: int = 0) -> None:
        self.model.train()
        # build batch: 1 pos + N neg
        v_list = [pos_v] + list(neg_vs)
        y = torch.tensor([1.0] + [0.0] * len(neg_vs), device=self.device, dtype=torch.float32)

        u_t = torch.tensor([u] * len(v_list), device=self.device, dtype=torch.long)
        v_t = torch.tensor(v_list, device=self.device, dtype=torch.long)
        ts_t = torch.tensor([t] * len(v_list), device=self.device, dtype=torch.long)

        out = self._forward(u_t, v_t, ts_t).view(-1)

        # if output already prob, convert to logits safely
        if torch.min(out) >= 0.0 and torch.max(out) <= 1.0:
            out = torch.clamp(out, 1e-4, 1 - 1e-4)
            out = torch.log(out / (1 - out))

        loss = self.bce_logits(out, y)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        self.last_loss = float(loss.detach().cpu().item())

    def diagnostics(self) -> Dict[str, Any]:
        return {"last_loss": self.last_loss, "model": "tgb_graphmixer"}

    def _forward(self, u_t: torch.Tensor, v_t: torch.Tensor, ts_t: torch.Tensor) -> torch.Tensor:
        """
        Try common forward APIs.
        """
        # common: model(u, v, t)
        try:
            return self.model(u_t, v_t, ts_t)
        except Exception:
            pass

        # sometimes: model.compute_edge_probabilities(u, v, t)
        if hasattr(self.model, "compute_edge_probabilities"):
            fn = getattr(self.model, "compute_edge_probabilities")
            return fn(u_t, v_t, ts_t)

        # sometimes: model.forward(u, v, t) explicitly
        if hasattr(self.model, "forward"):
            return self.model.forward(u_t, v_t, ts_t)

        raise RuntimeError("GraphMixer forward API not found. Inspect tgb_baselines/models/GraphMixer.py.")
