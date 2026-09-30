"""Baseline RAG : index CLIP cosine (deja utilise dans benchmark_rag_baseline)."""

from __future__ import annotations

import numpy as np
import torch


class RagBaseline:
    name = "rag_clip"
    family = "rag"

    def __init__(self):
        self.ids: list[str] = []
        self._chunks: list[np.ndarray] = []

    def reset(self) -> None:
        self.ids = []
        self._chunks = []

    def add_episode(
        self,
        episode_id: str,
        sem: torch.Tensor,
        dg=None,
        hpc=None,
        metadata=None,
    ) -> None:
        self.ids.append(str(episode_id))
        v = sem.detach().cpu().numpy().astype(np.float32).reshape(-1)
        n = np.linalg.norm(v)
        self._chunks.append(v / max(n, 1e-8))

    def build_index(self) -> None:
        pass

    @property
    def _matrix(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack(self._chunks, axis=0)

    def retrieve_ids(
        self,
        query_sem: torch.Tensor,
        query_dg=None,
        k: int = 5,
        allowed_ids: set[str] | None = None,
    ) -> list[str]:
        mat = self._matrix
        if mat.size == 0:
            return []
        q = query_sem.detach().cpu().numpy().astype(np.float32).reshape(-1)
        q = q / max(np.linalg.norm(q), 1e-8)
        scores = mat @ q
        if allowed_ids is not None:
            for i, eid in enumerate(self.ids):
                if eid not in allowed_ids:
                    scores[i] = -1e9
        k = min(k, len(self.ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [self.ids[int(i)] for i in top]

    def stats(self) -> dict:
        return {"n_episodes": len(self.ids), "backend": "cosine_faiss_optional"}
