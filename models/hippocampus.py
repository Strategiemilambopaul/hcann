import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GridCellModule(nn.Module):
    """Vector-HaSH: Module de cellules de grille sur tore 2D"""
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        # Phase 2D pour chaque module (x, y)
        self.register_buffer('phase', torch.zeros(1, 2))
        
    def forward(self, velocity: torch.Tensor = None) -> torch.Tensor:
        if velocity is not None:
            self.phase = (self.phase + velocity) % self.period
        freq = 2 * np.pi / self.period
        # Sinus et cosinus pour chaque dimension de la phase (2D → 4D)
        return torch.cat([
            torch.sin(self.phase * freq), 
            torch.cos(self.phase * freq)
        ], dim=-1)  # Shape: [1, 4]

class HippocampalScaffold(nn.Module):
    """Vector-HaSH: Scaffold fixe + projections aléatoires + retour appris"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.grid_modules = nn.ModuleList([GridCellModule(p) for p in config.grid_periods])
        
        # Chaque module retourne 4 dimensions (sin/cos pour x/y)
        grid_dim = len(config.grid_periods) * 4
        # Projection fixe aléatoire Grille → Hippocampe
        self.W_grid_to_hpc = nn.Linear(grid_dim, config.hpc_size, bias=False)
        self._init_fixed_projection()
        
        # Retour appris une fois puis figé (Hebb-like)
        self.W_hpc_to_grid = nn.Linear(config.hpc_size, grid_dim, bias=False)
        
        # Attracteurs symétriques (VLEM)
        self.W_attractor = nn.Parameter(torch.randn(config.hpc_size, config.hpc_size))
        self.W_attractor = nn.Parameter((self.W_attractor + self.W_attractor.T) / 2)
        
    def _init_fixed_projection(self):
        with torch.no_grad():
            nn.init.normal_(self.W_grid_to_hpc.weight, std=1.0)
            self.W_grid_to_hpc.weight.requires_grad = False
            
    def forward(self, velocity: torch.Tensor = None, steps: int = 3) -> torch.Tensor:
        # Path integration
        grid_states = [mod(velocity) for mod in self.grid_modules]
        g = torch.cat(grid_states, dim=-1)
        
        # Projection vers hippocampe
        h = F.relu(self.W_grid_to_hpc(g) - self.config.hpc_threshold)
        
        # Dynamique d'attracteur (nettoyage)
        for _ in range(steps):
            h = torch.tanh(h @ self.W_attractor)
            
        return h, g
    
    def reset_phases(self):
        for mod in self.grid_modules:
            mod.phase.zero_()