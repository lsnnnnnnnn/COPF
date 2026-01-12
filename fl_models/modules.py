# fl_models/modules.py
from __future__ import annotations

import math
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """
    Cosine time encoding used by GraphMixer/TGN-style models.

    Supports:
      - TimeEncoder(time_dim=..., parameter_requires_grad=False)
      - forward(timestamps=Tensor)  or forward(Tensor)
    Returns: (B, *, time_dim)
    """

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        super().__init__()
        self.time_dim = int(time_dim)

        # typical log-spaced frequencies
        freq = torch.logspace(0, 9, steps=self.time_dim, base=10.0) / 1e9
        phase = torch.zeros(self.time_dim)

        self.basis_freq = nn.Parameter(freq, requires_grad=bool(parameter_requires_grad))
        self.phase = nn.Parameter(phase, requires_grad=bool(parameter_requires_grad))

    def forward(self, x: torch.Tensor | None = None, *, timestamps: torch.Tensor | None = None) -> torch.Tensor:
        if timestamps is None:
            if x is None:
                raise TypeError("TimeEncoder.forward requires a tensor via positional arg or timestamps=...")
            timestamps = x

        ts = timestamps
        if not isinstance(ts, torch.Tensor):
            ts = torch.as_tensor(ts)
        ts = ts.to(dtype=torch.float32)

        # make shape (..., 1) then broadcast with (D,)
        ts = ts.unsqueeze(-1)  # (..., 1)
        out = torch.cos(ts * self.basis_freq + self.phase)  # (..., D)
        return out
