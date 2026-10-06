import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class MultimodalEncoder(nn.Module):
    """Néocortex : encodeurs sémantiques gelés (CLIP vision & texte)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_mock = getattr(config, 'use_mock_encoder', False)
        
        if not self.use_mock:
            local_only = getattr(config, "clip_local_files_only", False)
            try:
                self.clip = CLIPModel.from_pretrained(config.clip_model, local_files_only=local_only)
                self.clip_proc = CLIPProcessor.from_pretrained(config.clip_model, local_files_only=local_only)
            except Exception as exc:
                if getattr(config, "fail_on_clip_load_error", False):
                    raise RuntimeError(f"Impossible de charger CLIP '{config.clip_model}'") from exc
                print(f"[MultimodalEncoder] Chargement CLIP échoué, fallback mock activé: {exc}")
                self.use_mock = True
                self.clip = None
                self.clip_proc = None
            self._freeze_clip()
        else:
            # Mock encoder pour tests sans téléchargement
            self.clip = None
            self.clip_proc = None
            
        self.vision_proj = nn.Linear(512, config.semantic_dim)
        self.text_proj = nn.Linear(512, config.semantic_dim)
        
    def _freeze_clip(self):
        if self.clip:
            for p in self.clip.parameters(): p.requires_grad = False

    @staticmethod
    def _as_feature_tensor(features) -> torch.Tensor:
        """Compat transformers recent : tensor ou ModelOutput."""
        if isinstance(features, torch.Tensor):
            return features
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            val = getattr(features, attr, None)
            if val is not None:
                return val
        hidden = getattr(features, "last_hidden_state", None)
        if hidden is not None:
            return hidden[:, 0] if hidden.dim() == 3 else hidden
        raise TypeError(f"Sortie CLIP non reconnue: {type(features)}")

    def _denorm_imagenet(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (tensor * std + mean).clamp(0.0, 1.0)

    def _prepare_texts(self, texts):
        if texts is None:
            return None
        if isinstance(texts, str):
            return [texts]
        if isinstance(texts, (list, tuple)):
            if len(texts) == 1 and isinstance(texts[0], (list, tuple)):
                return [str(t) for t in texts[0]]
            return [str(t) for t in texts]
        return [str(texts)]

    def _prepare_image_inputs(self, images, device):
        if images is None:
            return None

        from PIL import Image

        # Tenseurs du dataloader (ImageNet norm) -> PIL -> preprocessing CLIP
        if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[1] == 3:
            images = self._denorm_imagenet(images.detach().cpu())
            prepared_images = []
            for im in images:
                arr = (im.permute(1, 2, 0).numpy() * 255).astype("uint8")
                prepared_images.append(Image.fromarray(arr))
            pixel_values = self.clip_proc(images=prepared_images, return_tensors="pt")["pixel_values"]
            return pixel_values.to(device)

        prepared_images = images
        if isinstance(images, torch.Tensor):
            img = images.detach().cpu()
            if img.ndim == 3:
                # [C,H,W] ou [H,W,C]
                if img.shape[0] in (1, 3):
                    prepared_images = [img]
                elif img.shape[-1] in (1, 3):
                    prepared_images = [img.permute(2, 0, 1)]
                else:
                    raise ValueError(f"Format image tensor non supporté: {tuple(img.shape)}")
            elif img.ndim == 4:
                # [B,C,H,W] ou [B,H,W,C]
                if img.shape[1] in (1, 3):
                    prepared_images = [im for im in img]
                elif img.shape[-1] in (1, 3):
                    prepared_images = [im.permute(2, 0, 1) for im in img]
                else:
                    raise ValueError(f"Format batch image tensor non supporté: {tuple(img.shape)}")
            else:
                raise ValueError(f"Rank image tensor non supporté: {img.ndim}")

        pixel_values = self.clip_proc(images=prepared_images, return_tensors="pt")["pixel_values"]
        return pixel_values.to(device)
        
    def forward(self, images=None, texts=None):
        if self.use_mock:
            return self._mock_forward(images, texts)
            
        embeddings = []
        device = next(self.parameters()).device
        if images is not None:
            pixel_values = self._prepare_image_inputs(images, device)
            v = self._as_feature_tensor(self.clip.get_image_features(pixel_values=pixel_values))
            embeddings.append(F.normalize(self.vision_proj(v.float()), dim=-1))
        if texts is not None:
            text_batch = self._prepare_texts(texts)
            inputs = self.clip_proc(text=text_batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            t = self._as_feature_tensor(self.clip.get_text_features(**inputs))
            embeddings.append(F.normalize(self.text_proj(t.float()), dim=-1))
        if not embeddings: raise ValueError("Aucune modalité fournie")
        return torch.stack(embeddings).mean(dim=0) if len(embeddings) > 1 else embeddings[0]
    
    def _mock_forward(self, images=None, texts=None):
        """Mock encoder pour tests - retourne des embeddings aléatoires normalisés"""
        embeddings = []
        target_batch_size = None
        if images is not None:
            target_batch_size = images.shape[0] if isinstance(images, torch.Tensor) else 1
            v = torch.randn(target_batch_size, 512, device=self.vision_proj.weight.device)
            embeddings.append(F.normalize(self.vision_proj(v), dim=-1))
        if texts is not None:
            if isinstance(texts, (list, tuple)):
                text_batch_size = len(texts)
                # Cas collate étrange: [['t1', 't2', ...]]
                if text_batch_size == 1 and isinstance(texts[0], (list, tuple)):
                    text_batch_size = len(texts[0])
            else:
                text_batch_size = 1

            if target_batch_size is not None:
                text_batch_size = target_batch_size

            t = torch.randn(text_batch_size, 512, device=self.text_proj.weight.device)
            embeddings.append(F.normalize(self.text_proj(t), dim=-1))
        if not embeddings:
            return torch.randn(1, self.config.semantic_dim, device=self.vision_proj.weight.device)
        return torch.stack(embeddings).mean(dim=0) if len(embeddings) > 1 else embeddings[0]

class WorkingMemory(nn.Module):
    """Mémoire de travail : slots + perte de diversité."""
    def __init__(self, config):
        super().__init__()
        self.slots = nn.Parameter(torch.zeros(config.wm_slots, config.wm_dim))
        self.scale = config.wm_dim ** -0.5
        
    def forward(self, x: torch.Tensor, write: bool = True) -> tuple:
        attn = torch.matmul(x, self.slots.T) * self.scale
        w = F.softmax(attn, dim=-1)
        read = torch.matmul(w, self.slots)
        if write:
            with torch.no_grad():
                best = torch.argmax(w, dim=-1)
                x_det = x.detach()
                slots = self.slots.data
                for i in range(x_det.size(0)):
                    b = int(best[i].item())
                    slots[b].mul_(0.95).add_(x_det[i], alpha=0.05)
        return read, w
    
    def diversity_loss(self) -> torch.Tensor:
        norm = F.normalize(self.slots, dim=-1)
        sim = torch.matmul(norm, norm.T) - torch.eye(self.slots.size(0), device=self.slots.device)
        return (sim ** 2).mean()