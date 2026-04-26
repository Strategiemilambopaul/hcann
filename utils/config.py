from dataclasses import dataclass
from typing import List

@dataclass
class HCANNConfig:
    # 🌍 Multimodal & Sémantique (VLEM)
    use_vision: bool = True
    use_language: bool = True
    semantic_dim: int = 256
    clip_model: str = "openai/clip-vit-base-patch32"
    
    # 🧠 Hippocampe & Scaffold (Vector-HaSH)
    grid_periods: List[int] = None
    hpc_size: int = 256
    hpc_threshold: float = 0.5
    scaffold_lr: float = 0.01
    hetero_lr: float = 0.001
    
    # 🧩 Cortex & Attracteurs (VLEM)
    wm_slots: int = 7
    wm_dim: int = 256
    entorhinal_dim: int = 256
    attractor_dim: int = 128  # par attribut (where/what/when)
    attractor_steps: int = 5
    
    # 🕸️ Consolidation & Graphe (HeLa-Mem)
    hebbian_lr: float = 0.02
    hebbian_decay: float = 0.995
    spreading_strength: float = 0.1
    hub_threshold: int = 10
    prune_weight_thresh: float = 0.1
    prune_age_thresh: float = 86400 * 7  # 7 jours
    
    # ⚙️ Entraînement
    batch_size: int = 32
    epochs_per_task: int = 5
    consolidation_freq: int = 10
    
    def __post_init__(self):
        if self.grid_periods is None:
            self.grid_periods = [3, 5, 7]