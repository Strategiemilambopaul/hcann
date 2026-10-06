import torch
import torch.nn as nn
import torch.nn.functional as F


class DentateGyrus(nn.Module):
    """Gyrus denté : séparation de formes via expansion figée + k-WTA."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        grid_dim = len(config.grid_periods) * 4
        self.ctx_dim = int(getattr(config, "ctx_dim", 0))
        input_dim = config.ec_dim + grid_dim + self.ctx_dim
        self.register_buffer(
            "W_expansion",
            torch.randn(input_dim, config.dg_dim) / (input_dim ** 0.5),
        )
        self.W_expansion.requires_grad = False

    def _topk_sparse(self, x: torch.Tensor) -> torch.Tensor:
        k = max(1, int(self.config.dg_dim * self.config.dg_sparsity))
        topk = torch.topk(x, k, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(-1, topk.indices, topk.values)
        return F.normalize(out, dim=-1, p=2)

    def forward(
        self,
        sdr_ec: torch.Tensor,
        grid_code: torch.Tensor,
        ctx_code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concat EC + grille (+ contexte temporel) -> code DG séparé [B, dg_dim]."""
        parts = [sdr_ec, grid_code]
        if self.ctx_dim > 0:
            if ctx_code is None:
                ctx_code = sdr_ec.new_zeros(sdr_ec.size(0), self.ctx_dim)
            parts.append(ctx_code)
        combined = torch.cat(parts, dim=-1)
        expanded = F.relu(combined @ self.W_expansion)
        return self._topk_sparse(expanded)
