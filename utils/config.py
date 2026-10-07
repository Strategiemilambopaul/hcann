from dataclasses import dataclass
from typing import List

@dataclass
class HCANNConfig:
    # Multimodal & neocortex
    use_vision: bool = True
    use_language: bool = True
    semantic_dim: int = 256
    clip_model: str = "openai/clip-vit-base-patch32"
    use_mock_encoder: bool = True

    # Échafaudage spatial (Vector-HaSH)
    grid_periods: List[int] = None
    hpc_size: int = 256  # alias dg_dim pour compat graphe hebbien

    # Cortex entorhinal (EC) — SDR
    ec_dim: int = 512
    ec_sparsity: float = 0.04  # fraction de neurones actifs

    # Gyrus denté (DG) — pattern separation
    dg_dim: int = 256
    dg_sparsity: float = 0.08
    dg_expansion_factor: int = 4

    # CA3 Modern Hopfield
    hopfield_beta: float = 8.0
    hopfield_steps: int = 5
    max_patterns: int = 2000

    # CA1 — match/mismatch
    ca1_novelty_threshold: float = 0.35

    # Subiculum — apprentissage local
    hebb_lr: float = 0.01
    stdp_lr: float = 0.005

    # Mémoire de travail (néocortex)
    wm_slots: int = 7
    wm_dim: int = 256

    # Graphe hebbien
    hebbian_lr: float = 0.02
    hebbian_decay: float = 0.995
    spreading_strength: float = 0.1
    hub_threshold: int = 10
    prune_weight_thresh: float = 0.1
    prune_age_thresh: float = 86400 * 7
    max_nodes: int = 2000

    # Entraînement & dreaming
    batch_size: int = 32
    epochs_per_task: int = 5
    consolidation_freq: int = 10
    dream_cycles: int = 3
    dream_batch_size: int = 16
    enable_dreaming: bool = True

    # Mémoire épisodique (défauts = comportement historique de models/)
    episodic_mode: bool = False
    ctx_dim: int = 0
    ctx_rho: float = 0.9
    ctx_jump: float = 0.5
    mem_novelty_threshold: float = 0.25
    seg_k: float = 1.5
    seg_ema: float = 0.9
    min_event_len: int = 3
    seg_signal: str = "dg"
    reencode_boundary: bool = True
    ca1_decay: float = 0.999
    dream_seq_len: int = 4

    def __post_init__(self):
        if self.grid_periods is None:
            self.grid_periods = [3, 5, 7]
        if self.dg_dim != self.hpc_size:
            self.hpc_size = self.dg_dim
        if self.wm_dim != self.semantic_dim:
            self.wm_dim = self.semantic_dim
