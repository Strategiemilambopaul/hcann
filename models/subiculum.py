import torch
import torch.nn as nn


class SubiculumGateway(nn.Module):
    """Subiculum : porte de sortie hippocampe -> néocortex, mise à jour Hebb locale."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W = nn.Parameter(torch.randn(config.semantic_dim, config.dg_dim) * 0.01)

    def forward(self, hpc_state: torch.Tensor) -> torch.Tensor:
        """hpc_state [B, dg_dim] -> cortical [B, semantic_dim]."""
        return hpc_state @ self.W.T

    @torch.no_grad()
    def hebb_update(self, hpc_state: torch.Tensor, sem_target: torch.Tensor) -> float:
        """
        Règle de Hebb locale : W += η * (target - W @ hpc) ⊗ hpc
        Returns: norme moyenne du delta pour diagnostic.
        """
        eta = self.config.hebb_lr
        if hpc_state.dim() == 1:
            hpc_state = hpc_state.unsqueeze(0)
        if sem_target.dim() == 1:
            sem_target = sem_target.unsqueeze(0)

        pred = hpc_state @ self.W.T
        error = sem_target - pred
        delta = eta * (error.unsqueeze(2) * hpc_state.unsqueeze(1)).mean(dim=0)
        self.W.add_(delta)
        return delta.norm().item()
