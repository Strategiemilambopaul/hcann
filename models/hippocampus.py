import torch
import torch.nn as nn
import numpy as np

from .dentate_gyrus import DentateGyrus
from .ca3_hopfield import CA3ModernHopfield
from .ca1 import CA1Comparator
from .subiculum import SubiculumGateway


class GridCellModule(nn.Module):
    """Vector-HaSH : cellules de grille sur tore 2D."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.register_buffer("phase", torch.zeros(1, 2))

    def forward(self, velocity: torch.Tensor = None) -> torch.Tensor:
        if velocity is not None:
            if self.phase.size(0) != velocity.size(0):
                self.phase = self.phase[:1].expand(velocity.size(0), -1).clone()
            self.phase = (self.phase + velocity) % self.period
        freq = 2 * np.pi / self.period
        return torch.cat([
            torch.sin(self.phase * freq),
            torch.cos(self.phase * freq),
        ], dim=-1)

    def reset_phase(self):
        self.phase.zero_()


class TrisynapticHippocampus(nn.Module):
    """Boucle trisynaptique : DG -> CA3 -> CA1 -> Subiculum + échafaudage spatial."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.grid_modules = nn.ModuleList([GridCellModule(p) for p in config.grid_periods])
        self.dg = DentateGyrus(config)
        self.ca3 = CA3ModernHopfield(config)
        self.ca1 = CA1Comparator(config)
        self.subiculum = SubiculumGateway(config)

    def encode_grid(self, velocity: torch.Tensor = None) -> torch.Tensor:
        grid_states = [mod(velocity) for mod in self.grid_modules]
        return torch.cat(grid_states, dim=-1)

    def forward(
        self,
        sdr_ec: torch.Tensor,
        sem_dense: torch.Tensor,
        velocity: torch.Tensor = None,
        store: bool = True,
        episode_keys: list | None = None,
    ) -> dict:
        grid_code = self.encode_grid(velocity)
        dg_code = self.dg(sdr_ec, grid_code)

        if store:
            if episode_keys:
                for i, key in enumerate(episode_keys):
                    if key is not None:
                        self.ca3.store(dg_code[i], key=str(key))
                    else:
                        self.ca3.store(dg_code[i : i + 1])
            else:
                self.ca3.store(dg_code)

        ca3_state = self.ca3.complete(dg_code)
        decoded, novelty, is_novel = self.ca1(ca3_state, sem_dense)
        cortical = self.subiculum(ca3_state)

        return {
            "ca3_state": ca3_state,
            "dg_code": dg_code,
            "grid_code": grid_code,
            "decoded": decoded,
            "novelty": novelty,
            "is_novel": is_novel,
            "cortical": cortical,
        }

    def reset_phases(self):
        for mod in self.grid_modules:
            mod.reset_phase()

    def reset_memory(self):
        self.ca3.reset()


# Alias rétrocompatibilité
HippocampalScaffold = TrisynapticHippocampus
