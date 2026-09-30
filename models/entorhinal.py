import torch
import torch.nn as nn
import torch.nn.functional as F


class EntorhinalCortex(nn.Module):
    """Cortex entorhinal : compression multimodale vers SDR binaires clairsemées."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.semantic_dim, config.ec_dim, bias=False)

    def _topk_binary(self, x: torch.Tensor) -> torch.Tensor:
        k = max(1, int(self.config.ec_dim * self.config.ec_sparsity))
        topk = torch.topk(x, k, dim=-1)
        sdr = torch.zeros_like(x)
        sdr.scatter_(-1, topk.indices, 1.0)
        return sdr

    def forward(self, sem: torch.Tensor) -> torch.Tensor:
        """sem [B, semantic_dim] -> sdr [B, ec_dim] binaire {0,1}."""
        activations = F.relu(self.proj(sem))
        return self._topk_binary(activations)

    def sparsity(self, sdr: torch.Tensor) -> float:
        return sdr.mean().item()
