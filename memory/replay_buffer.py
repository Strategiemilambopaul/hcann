import torch
import random
import time
from collections import deque
from typing import Dict, List, Optional
import numpy as np

class EpisodicBuffer:
    """Stockage épisodes multimodaux pour consolidation différée"""
    def __init__(self, capacity: int = 2000):
        self.buffer: deque = deque(maxlen=capacity)
        
    def push(self, sem_embed: torch.Tensor, hpc_state: torch.Tensor, metadata: Optional[Dict] = None):
        """Ajoute un épisode au buffer"""
        self.buffer.append({
            'sem': sem_embed.detach().cpu(),
            'hpc': hpc_state.detach().cpu(),
            'meta': metadata or {},
            'ts': time.time()
        })
        
    def sample(self, n: int = 32) -> Optional[Dict[str, torch.Tensor]]:
        """Échantillonne un batch d'épisodes pour la consolidation"""
        if len(self.buffer) == 0:
            return None
        n = min(n, len(self.buffer))
        batch = random.sample(list(self.buffer), n)
        return {
            'sem': torch.stack([b['sem'] for b in batch]),
            'hpc': torch.stack([b['hpc'] for b in batch]),
            'meta': [b['meta'] for b in batch],
            'ts': [b['ts'] for b in batch]
        }
    
    def sample_sequences(self, n_seq: int, seq_len: int) -> List[List[Dict]]:
        """Fenêtres contiguës dans l'ordre d'insertion (pas un tirage i.i.d.)."""
        if n_seq <= 0 or seq_len <= 0:
            return []
        items = list(self.buffer)
        if len(items) < seq_len:
            return []
        starts = list(range(len(items) - seq_len + 1))
        if len(starts) > n_seq:
            starts = sorted(random.sample(starts, n_seq))
        return [items[s : s + seq_len] for s in starts]

    def get_recent(self, n: int = 10) -> List[Dict]:
        """Retourne les n épisodes les plus récents"""
        return list(self.buffer)[-n:]
    
    def clear(self):
        """Vide le buffer"""
        self.buffer.clear()
        
    def __len__(self) -> int:
        return len(self.buffer)


class ExperienceReplayBuffer(EpisodicBuffer):
    """Alias pour compatibilité avec l'ancien code"""
    pass