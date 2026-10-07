import torch
import torch.nn as nn
import torch.nn.functional as F


class CA3ModernHopfield(nn.Module):
    """CA3 : Modern Hopfield (Ramsauer et al.) — store one-shot + pattern completion."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pattern_dim = config.dg_dim
        self.register_buffer("patterns", torch.zeros(0, self.pattern_dim))
        self._count = 0
        self._key_to_idx: dict[str, int] = {}

    @property
    def num_patterns(self) -> int:
        return self.patterns.size(0)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=1e-8)

    def store(self, pattern: torch.Tensor, key: str | None = None) -> None:
        """Mémorisation one-shot ; upsert si key deja vue (re-encodage)."""
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)
        normed = self._normalize(pattern.detach())
        device = normed.device
        if self.patterns.numel() == 0:
            self.patterns = torch.zeros(0, self.pattern_dim, device=device)

        for i in range(normed.size(0)):
            row = normed[i : i + 1]
            item_key = key if normed.size(0) == 1 else None

            if item_key is not None and item_key in self._key_to_idx:
                idx = self._key_to_idx[item_key]
                if idx < self.patterns.size(0):
                    self.patterns[idx] = row.squeeze(0)
                    continue

            if self.num_patterns >= self.config.max_patterns:
                self.patterns = self.patterns[1:]
                self._key_to_idx = {
                    k: v - 1 for k, v in self._key_to_idx.items() if v > 0
                }

            idx = self.num_patterns
            self.patterns = torch.cat([self.patterns, row], dim=0)
            if item_key is not None:
                self._key_to_idx[item_key] = idx
            self._count += 1

    def complete(self, query: torch.Tensor, steps: int = None) -> torch.Tensor:
        """Pattern completion par dynamique Modern Hopfield."""
        if self.num_patterns == 0:
            return query if query.dim() == 2 else query.unsqueeze(0)

        steps = steps or self.config.hopfield_steps
        beta = self.config.hopfield_beta

        if query.dim() == 1:
            query = query.unsqueeze(0)

        xi = self._normalize(query)
        P = self.patterns  # [N, D]

        for _ in range(steps):
            # sim [B, N] = xi @ P.T
            sim = beta * (xi @ P.T)
            weights = F.softmax(sim, dim=-1)
            xi = self._normalize(weights @ P)

        return xi

    def complete_among(
        self,
        query: torch.Tensor,
        keys: list[str],
        margin: float = 0.05,
    ) -> tuple[str | None, dict[str, float]]:
        """Complétion restreinte aux souvenirs nommés. None si l'attracteur ne départage pas."""
        present = [(str(k), self._key_to_idx[str(k)]) for k in keys if str(k) in self._key_to_idx]
        if len(present) < 2 or self.num_patterns == 0:
            return None, {}
        idxs = [idx for _, idx in present]
        P = self._normalize(self.patterns[idxs])
        if query.dim() == 1:
            query = query.unsqueeze(0)
        xi = self._normalize(query.to(P.device, dtype=P.dtype))
        steps = self.config.hopfield_steps
        beta = self.config.hopfield_beta
        for _ in range(steps):
            weights = F.softmax(beta * (xi @ P.T), dim=-1)
            xi = self._normalize(weights @ P)
        cos = (xi @ P.T).squeeze(0)
        if cos.dim() == 0:
            return None, {}
        scores = {present[i][0]: float(cos[i].item()) for i in range(len(present))}
        ranked = sorted(scores, key=scores.get, reverse=True)
        if scores[ranked[0]] - scores[ranked[1]] >= margin:
            return ranked[0], scores
        return None, scores

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """Énergie Modern Hopfield (diagnostic)."""
        if self.num_patterns == 0:
            return torch.zeros(state.size(0), device=state.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        beta = self.config.hopfield_beta
        sim = beta * (state @ self.patterns.T)
        return -torch.logsumexp(sim, dim=-1)

    def max_pattern_similarity(self, query: torch.Tensor) -> torch.Tensor:
        """Similarite max avec les patterns stockes (haute = familier)."""
        if self.num_patterns == 0:
            return torch.zeros(query.size(0), device=query.device)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        q = self._normalize(query)
        return (q @ self._normalize(self.patterns).T).max(dim=-1).values

    def nearest(self, state: torch.Tensor, k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
        """Indices et cosinus des k patterns les plus proches."""
        if self.num_patterns == 0:
            empty = torch.zeros(0, dtype=torch.long, device=state.device)
            return empty, torch.zeros(0, device=state.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        q = self._normalize(state)
        cos = (q @ self._normalize(self.patterns).T).squeeze(0)
        k_eff = min(int(k), self.num_patterns)
        vals, idxs = torch.topk(cos, k_eff)
        return idxs, vals

    def key_of(self, idx: int) -> str | None:
        """Clé d'épisode associée à un indice de pattern, si elle existe."""
        for key, value in self._key_to_idx.items():
            if value == int(idx):
                return key
        return None

    @torch.no_grad()
    def reconsolidate(self, key: str, target: torch.Tensor, lr: float) -> tuple[float, float]:
        """Mélange le pattern nommé vers target. Renvoie (cos avant, cos après)."""
        key = str(key)
        if key not in self._key_to_idx:
            return 0.0, 0.0
        idx = self._key_to_idx[key]
        if target.dim() == 1:
            target = target.unsqueeze(0)
        target_n = self._normalize(target.detach())[0]
        before = float((self.patterns[idx] * target_n).sum().item())
        mixed = (1.0 - float(lr)) * self.patterns[idx] + float(lr) * target_n
        self.patterns[idx] = self._normalize(mixed.unsqueeze(0)).squeeze(0)
        after = float((self.patterns[idx] * target_n).sum().item())
        return before, after

    def reset(self) -> None:
        self.patterns = torch.zeros(0, self.pattern_dim, device=self.patterns.device)
        self._count = 0
        self._key_to_idx = {}
