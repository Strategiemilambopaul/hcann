"""Encodeur de test : un texte correspond toujours au même vecteur unitaire."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest


def bind_text_table(model, mapping: dict) -> dict[str, torch.Tensor]:
    """Remplace encoder.forward par une table texte -> vecteur unitaire fixe."""
    dim = int(model.config.semantic_dim)
    device = next(model.parameters()).device
    table: dict[str, torch.Tensor] = {}
    for key, value in mapping.items():
        vec = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
        if vec.numel() != dim:
            raise ValueError(f"{key}: dimension {vec.numel()} != {dim}")
        table[str(key)] = F.normalize(vec, dim=0)

    def forward(images=None, texts=None):
        if texts is None:
            n = 1
            if images is not None and hasattr(images, "shape"):
                n = int(images.shape[0])
            return torch.zeros(n, dim, device=device)
        batch = [texts] if isinstance(texts, str) else list(texts)
        rows = []
        for text in batch:
            row = table.get(str(text))
            if row is None:
                row = torch.zeros(dim, device=device)
            rows.append(row)
        return torch.stack(rows, dim=0)

    model.encoder.forward = forward
    return table


@pytest.fixture
def bind_encoder():
    return bind_text_table
