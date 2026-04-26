import torch
from collections import deque

class EpisodicBuffer:
    """Stockage épisodes multimodaux pour consolidation"""
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, sem_embed, hpc_state, metadata=None):
        self.buffer.append({
            'sem': sem_embed.detach().cpu(),
            'hpc': hpc_state.detach().cpu(),
            'meta': metadata or {},
            'ts': time.time()
        })
        
    def sample(self, n=32):
        if len(self.buffer) < n: return None
        batch = random.sample(self.buffer, n)
        return {
            'sem': torch.stack([b['sem'] for b in batch]),
            'hpc': torch.stack([b['hpc'] for b in batch])
        }