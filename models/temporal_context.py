"""Contexte temporel lent (Temporal Context Model), projection fixe."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalContext(nn.Module):
    """Vecteur qui dérive à chaque événement. ctx_dim=0 : désactivé."""

    def __init__(self, config):
        super().__init__()
        self.ctx_dim = int(getattr(config, "ctx_dim", 0))
        self.rho = float(getattr(config, "ctx_rho", 0.9))
        self.jump = float(getattr(config, "ctx_jump", 0.5))
        sem_dim = int(config.semantic_dim)
        if self.ctx_dim > 0:
            proj = torch.randn(sem_dim, self.ctx_dim) / (sem_dim ** 0.5)
        else:
            proj = torch.zeros(sem_dim, 0)
        self.register_buffer("proj", proj)
        self.register_buffer("context", torch.zeros(1, self.ctx_dim))
        self.register_buffer("last_u", torch.zeros(1, self.ctx_dim))

    def step(self, sem: torch.Tensor, advance: bool = True) -> torch.Tensor:
        """Une ligne de contexte par événement du batch, dans l'ordre."""
        if sem.dim() == 1:
            sem = sem.unsqueeze(0)
        batch = sem.size(0)
        if self.ctx_dim == 0:
            return sem.new_zeros(batch, 0)
        if not advance:
            return self.context.expand(batch, -1)
        rho = self.rho
        scale = (max(0.0, 1.0 - rho * rho)) ** 0.5
        outs = []
        ctx = self.context
        for i in range(batch):
            u = F.normalize(sem[i : i + 1].detach() @ self.proj, dim=-1, eps=1e-8)
            ctx = F.normalize(rho * ctx + scale * u, dim=-1, eps=1e-8)
            self.last_u.copy_(u)
            outs.append(ctx)
        self.context.copy_(ctx)
        return torch.cat(outs, dim=0)

    def boundary(self) -> None:
        """Saut de contexte à une frontière d'épisode."""
        if self.ctx_dim == 0:
            return
        mixed = (1.0 - self.jump) * self.context + self.jump * self.last_u
        self.context.copy_(F.normalize(mixed, dim=-1, eps=1e-8))

    def reset(self) -> None:
        self.context.zero_()
        self.last_u.zero_()
