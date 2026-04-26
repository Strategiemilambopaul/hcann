import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class MultimodalEncoder(nn.Module):
    """VLEM: Encodeurs sémantiques gelés (CLIP vision & texte)"""
    def __init__(self, config):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(config.clip_model)
        self.clip_proc = CLIPProcessor.from_pretrained(config.clip_model)
        self.vision_proj = nn.Linear(512, config.semantic_dim)
        self.text_proj = nn.Linear(512, config.semantic_dim)
        self._freeze_clip()
        
    def _freeze_clip(self):
        for p in self.clip.parameters(): p.requires_grad = False
        
    def forward(self, images=None, texts=None):
        embeddings = []
        if images is not None:
            v = self.clip.get_image_features(pixel_values=images)
            embeddings.append(F.normalize(self.vision_proj(v.float()), dim=-1))
        if texts is not None:
            inputs = self.clip_proc.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
            t = self.clip.get_text_features(**inputs)
            embeddings.append(F.normalize(self.text_proj(t.float()), dim=-1))
        if not embeddings: raise ValueError("Aucune modalité fournie")
        return torch.stack(embeddings).mean(dim=0) if len(embeddings) > 1 else embeddings[0]

class WorkingMemory(nn.Module):
    """VLEM: 7 slots RNN + perte de diversité"""
    def __init__(self, config):
        super().__init__()
        self.slots = nn.Parameter(torch.zeros(config.wm_slots, config.wm_dim))
        self.scale = config.wm_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> tuple:
        attn = torch.matmul(x, self.slots.T) * self.scale
        w = F.softmax(attn, dim=-1)
        read = torch.matmul(w, self.slots)
        # Écriture douce (mise à jour du slot le plus proche)
        with torch.no_grad():
            best = torch.argmax(w, dim=-1)
            for i in range(x.size(0)):
                self.slots[best[i]] = 0.95 * self.slots[best[i]] + 0.05 * x[i]
        return read, w
    
    def diversity_loss(self) -> torch.Tensor:
        norm = F.normalize(self.slots, dim=-1)
        sim = torch.matmul(norm, norm.T) - torch.eye(self.slots.size(0), device=self.slots.device)
        return (sim ** 2).mean()

class EntorhinalGateway(nn.Module):
    """VLEM: Passerelle cortex↔hippocampe par cross-attention"""
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.attractor_dim * 3, config.entorhinal_dim)
        self.kv_proj = nn.Linear(config.wm_dim, config.entorhinal_dim)
        self.out_proj = nn.Linear(config.entorhinal_dim, config.hpc_size)
        
    def forward(self, attractor_state: torch.Tensor, wm_state: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(attractor_state)
        k, v = self.kv_proj(wm_state).chunk(2, dim=-1)
        attn = F.softmax(torch.matmul(q, k.T) / (k.size(-1) ** 0.5), dim=-1)
        return self.out_proj(torch.matmul(attn, v))