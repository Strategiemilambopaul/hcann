import torch
import torch.nn as nn
import torch.nn.functional as F


class DentateGyrus(nn.Module):
    """Gyrus denté : séparation de formes via expansion figée + k-WTA."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        grid_dim = len(config.grid_periods) * 4
        input_dim = config.ec_dim + grid_dim
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

    def forward(self, sdr_ec: torch.Tensor, grid_code: torch.Tensor) -> torch.Tensor:
        """Concat EC + grille -> code DG séparé [B, dg_dim]."""
        combined = torch.cat([sdr_ec, grid_code], dim=-1)
        expanded = F.relu(combined @ self.W_expansion)
        return self._topk_sparse(expanded)
