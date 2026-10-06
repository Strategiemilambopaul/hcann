"""Phase 1.5 : seuil de spreading, signal de segmentation, événement-frontière."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.consolidation import HebbianMemoryGraph
from utils.config import HCANNConfig


def _naive_boosted(g: HebbianMemoryGraph, query: np.ndarray, allowed_ids: set | None):
    """Oracle : ancienne boucle, arête > 0,05, source si base > 0,2."""
    n = g.count
    q = torch.from_numpy(query).to(g.device).float()
    active = g.node_embeddings[:n]
    base = torch.matmul(
        torch.nn.functional.normalize(q.unsqueeze(0), dim=-1),
        torch.nn.functional.normalize(active, dim=-1).T,
    ).squeeze(0)
    base = g._mask_invalid(base, allowed_ids)
    boosted = base.clone()
    strength = g.config.spreading_strength
    for i in range(n):
        if float(base[i]) <= 0.2:
            continue
        for j in range(n):
            weight = float(g.edges[i, j])
            if weight <= 0.05:
                continue
            nid = g.node_ids[j]
            if allowed_ids is not None and nid not in allowed_ids:
                continue
            boosted[j] = boosted[j] + strength * weight
    return g._mask_invalid(boosted, allowed_ids)


def test_spreading_matches_edge_threshold_oracle():
    cfg = HCANNConfig()
    cfg.max_nodes = 12
    cfg.hpc_size = 8
    cfg.dg_dim = 8
    cfg.semantic_dim = 8
    cfg.spreading_strength = 0.1
    g = HebbianMemoryGraph(cfg, torch.device("cpu"))
    rng = np.random.default_rng(0)
    vecs = []
    for i in range(8):
        v = rng.normal(size=8).astype(np.float32)
        v = v / np.linalg.norm(v)
        vecs.append(v)
        g.add_node(f"n{i}", {"i": i}, v, sem_embedding=v, dg_embedding=v)
    for i in range(8):
        for j in range(i + 1, 8):
            weight = 0.02 if (i + j) % 2 == 0 else 0.3
            g.update_hebbian_weights(f"n{i}", f"n{j}", weight)
    query = vecs[0]
    for allowed in (None, {"n0", "n1", "n3", "n5"}):
        got = g.spreading_activation(query, k=8, allowed_ids=allowed, touch=False)
        oracle_scores = _naive_boosted(g, query, allowed)
        finite = torch.isfinite(oracle_scores)
        k = int(finite.sum().item())
        vals, idxs = torch.topk(oracle_scores, k)
        expected = []
        for value, idx in zip(vals.tolist(), idxs.tolist()):
            if not np.isfinite(value):
                continue
            expected.append((g.node_ids[idx], value))
        assert [h["id"] for h in got] == [e[0] for e in expected]
        for hit, (_, score) in zip(got, expected):
            assert abs(hit["score"] - score) < 1e-6
