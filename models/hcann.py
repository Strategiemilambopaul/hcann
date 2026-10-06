import torch
import torch.nn as nn
from .entorhinal import EntorhinalCortex
from .hippocampus import TrisynapticHippocampus
from .cortex import MultimodalEncoder, WorkingMemory


class HCANN(nn.Module):
    """Architecture HCANN : boucle trisynaptique + néocortex multimodal."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = MultimodalEncoder(config)
        self.ec = EntorhinalCortex(config)
        self.hippocampus = TrisynapticHippocampus(config)
        self.wm = WorkingMemory(config)

    def fast_encode(self, images=None, texts=None, velocity=None, store=True, episode_keys=None):
        """
        Phase rapide : néocortex -> EC (SDR) -> boucle trisynaptique.

        Returns:
            ca3_state, sem, wm_state, novelty, dg_code, extras dict
        """
        sem = self.encoder(images, texts)
        wm_state, _ = self.wm(sem, write=store)
        sdr_ec = self.ec(sem)

        if velocity is None:
            velocity = torch.zeros(sem.size(0), 2, device=sem.device)

        hpc_out = self.hippocampus(
            sdr_ec, sem, velocity=velocity, store=store, episode_keys=episode_keys
        )

        return (
            hpc_out["ca3_state"],
            sem,
            wm_state,
            hpc_out["novelty"],
            hpc_out["dg_code"],
            hpc_out,
        )

    @torch.no_grad()
    def local_hebb_update(self, sem_target: torch.Tensor, hpc_state: torch.Tensor) -> float:
        """Mise à jour Hebb locale Subiculum (appelée par DreamingPhase)."""
        return self.hippocampus.subiculum.hebb_update(hpc_state, sem_target)

    def slow_consolidate(self, hpc_target, sem_input):
        """Deprecated : déléguer à DreamingPhase. Conservé pour compatibilité API."""
        delta_norm = 0.0
        for i in range(hpc_target.size(0)):
            delta_norm += self.local_hebb_update(sem_input[i], hpc_target[i])
        return delta_norm / max(hpc_target.size(0), 1)

    @property
    def subiculum(self):
        return self.hippocampus.subiculum

    @property
    def ca1(self):
        return self.hippocampus.ca1

    @property
    def ca3(self):
        return self.hippocampus.ca3
