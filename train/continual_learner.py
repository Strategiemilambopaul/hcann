import torch
from tqdm import tqdm
from models.hcann import HCANN
from memory.consolidation import HebbianMemoryGraph
from memory.replay_buffer import EpisodicBuffer

class ContinualLearner:
    def __init__(self, model: HCANN, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.graph = HebbianMemoryGraph(config, device)
        self.buffer = EpisodicBuffer()
        self.step_count = 0
        
    def train_step(self, images=None, texts=None, velocity=None):
        """Boucle bio-plausible : encodage rapide → mise à jour hebbienne → consolidation lente"""
        self.model.train()
        
        # 1. Encodage rapide
        hpc, sem, wm = self.model.fast_encode(images, texts, velocity)
        
        # 2. Mise à jour graphe hebbien
        retrieved = self.graph.spreading_activation(hpc[0], k=3)
        idx = self.graph.add_node(hpc[0])
        if idx >= 0:
            self.graph.update_hebbian(idx, retrieved)
            self.buffer.push(sem[0], hpc[0])
            
        # 3. Consolidation périodique (cortex)
        self.step_count += 1
        if self.step_count % self.config.consolidation_freq == 0:
            batch = self.buffer.sample(32)
            if batch:
                loss = self.model.slow_consolidate(batch['hpc'].to(self.device), 
                                                   batch['sem'].to(self.device))
                self.model.cortex_opt.zero_grad()
                loss.backward()
                self.model.cortex_opt.step()
                
                # Oubli adaptatif
                self.graph.adaptive_forgetting()
                
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        correct, total = 0, 0
        for batch in dataloader:
            imgs, lbls = batch.get('images'), batch.get('labels')
            if imgs is not None:
                imgs = imgs.to(self.device)
            hpc, _, _ = self.model.fast_encode(images=imgs)
            # Classification simple via projection
            logits = self.model.W_hpc_to_sens(hpc) @ self.model.encoder.text_proj.weight.T
            preds = logits.argmax(dim=-1)
            correct += (preds == lbls.to(self.device)).sum().item()
            total += lbls.size(0)
        return correct / total if total > 0 else 0.0