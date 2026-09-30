import torch
import torch.nn as nn
import torch.nn.functional as F


class CA1Comparator(nn.Module):
    """CA1 : décodeur + détection match/mismatch (nouveauté)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = nn.Linear(config.dg_dim, config.semantic_dim, bias=False)

    def decode(self, ca3_state: torch.Tensor) -> torch.Tensor:
        return self.decoder(ca3_state)

    def forward(
        self,
        ca3_state: torch.Tensor,
        sem_target: torch.Tensor,
    ) -> tuple:
        """
        Returns:
            decoded_sem, novelty_score, is_novel
        """
        decoded = self.decode(ca3_state)
        sem_norm = F.normalize(sem_target, dim=-1, eps=1e-8)
        dec_norm = F.normalize(decoded, dim=-1, eps=1e-8)
        match = (sem_norm * dec_norm).sum(dim=-1)
        novelty = 1.0 - match
        is_novel = novelty > self.config.ca1_novelty_threshold
        return decoded, novelty, is_novel

    @torch.no_grad()
    def hebb_update(self, ca3_state: torch.Tensor, sem_target: torch.Tensor) -> float:
        """Apprentissage Hebb local du decodeur CA1 (alignement cosinus sur sem)."""
        eta = self.config.hebb_lr
        if ca3_state.dim() == 1:
            ca3_state = ca3_state.unsqueeze(0)
        if sem_target.dim() == 1:
            sem_target = sem_target.unsqueeze(0)
        ca3_n = F.normalize(ca3_state, dim=-1, eps=1e-8)
        sem_n = F.normalize(sem_target, dim=-1, eps=1e-8)
        delta = eta * torch.mm(sem_n.T, ca3_n)
        self.decoder.weight.add_(delta)
        return delta.norm().item()
