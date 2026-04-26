import torch
import torch.nn as nn
from .hippocampus import HippocampalScaffold
from .cortex import MultimodalEncoder, WorkingMemory, EntorhinalGateway

class HCANN(nn.Module):
    """Architecture HCANN Multimodale Bio-inspirée"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.encoder = MultimodalEncoder(config)
        self.hippocampus = HippocampalScaffold(config)
        self.wm = WorkingMemory(config)
        self.entorhinal = EntorhinalGateway(config)
        
        # Hétéro-association (Vector-HaSH)
        self.W_sens_to_hpc = nn.Linear(config.semantic_dim, config.hpc_size)
        self.W_hpc_to_sens = nn.Linear(config.hpc_size, config.semantic_dim)
        
        # Optimiseurs séparés (CLS theory)
        self.scaffold_opt = torch.optim.SGD(list(self.hippocampus.parameters()) + 
                                            [self.W_sens_to_hpc.weight, self.W_hpc_to_sens.weight], 
                                            lr=config.scaffold_lr)
        self.cortex_opt = torch.optim.Adam([p for n, p in self.named_parameters() 
                                            if 'hippocampus' not in n and 'encoder' not in n], 
                                           lr=config.hetero_lr)
                                           
    def fast_encode(self, images=None, texts=None, velocity=None):
        """Phase rapide : encodage multimodal + activation scaffold"""
        sem = self.encoder(images, texts)
        wm_state, _ = self.wm(sem)
        sens_to_hpc = self.W_sens_to_hpc(sem)
        hpc_state, _ = self.hippocampus(velocity)
        # Fusion additive rapide
        return hpc_state + sens_to_hpc, sem, wm_state
    
    def slow_consolidate(self, hpc_target, sem_input):
        """Phase lente : hétéro-association cortex↔hippocampe"""
        pred_sem = self.W_hpc_to_sens(hpc_target)
        loss_hetero = nn.functional.mse_loss(pred_sem, sem_input.detach())
        return loss_hetero