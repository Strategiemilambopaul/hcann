import torch
import torch.nn as nn
import time

class HebbianMemoryGraph(nn.Module):
    """HeLa-Mem: Graphe hebbien dynamique + consolidation"""
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.max_nodes = 2000
        self.nodes = torch.zeros(self.max_nodes, config.hpc_size, device=device)
        self.edges = torch.zeros(self.max_nodes, self.max_nodes, device=device)
        self.timestamps = torch.zeros(self.max_nodes, device=device)
        self.access_counts = torch.zeros(self.max_nodes, device=device)
        self.count = 0
        
    def add_node(self, embedding: torch.Tensor) -> int:
        if self.count < self.max_nodes:
            self.nodes[self.count] = embedding.detach()
            self.timestamps[self.count] = time.time()
            idx = self.count
            self.count += 1
            return idx
        return -1
        
    def update_hebbian(self, current_idx: int, retrieved: list):
        """Règle de Hebb: w(t+1) = (1-λ)w + η·I(co-activation)"""
        if current_idx < 0: return
        self.edges *= self.config.hebbian_decay
        for r in retrieved:
            if 0 <= r < self.count:
                self.edges[current_idx, r] += self.config.hebbian_lr
                self.edges[r, current_idx] += self.config.hebbian_lr
                
    def spreading_activation(self, query: torch.Tensor, k: int = 5) -> list:
        """Récupération base + propagation hebbienne"""
        if self.count == 0: return []
        active = self.nodes[:self.count]
        sim = F.cosine_similarity(query.unsqueeze(0), active, dim=-1)
        base_scores = sim.clone()
        
        # Propagation
        boosted = sim.clone()
        for i in range(self.count):
            if sim[i] > 0.2:
                neighbors = torch.where(self.edges[i, :self.count] > 0.05)[0]
                for n in neighbors:
                    boosted[n] += self.config.spreading_strength * self.edges[i, n]
                    
        top_k = torch.topk(boosted, min(k, self.count)).indices.tolist()
        for idx in top_k: self.access_counts[idx] += 1
        return top_k
        
    def detect_hubs(self) -> list:
        degrees = torch.sum(self.edges[:self.count, :self.count] > 0.05, dim=0)
        return torch.where(degrees > self.config.hub_threshold)[0].tolist()
        
    def adaptive_forgetting(self):
        now = time.time()
        to_remove = []
        for i in range(self.count):
            w_total = torch.sum(self.edges[i, :self.count])
            age = now - self.timestamps[i].item()
            if w_total < self.config.prune_weight_thresh and age > self.config.prune_age_thresh and self.access_counts[i] == 0:
                to_remove.append(i)
        for i in sorted(to_remove, reverse=True):
            self.nodes[i].zero_()
            self.edges[i, :].zero_()
            self.edges[:, i].zero_()