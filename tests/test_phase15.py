"""Phase 1.5 : seuil de spreading, signal de segmentation, événement-frontière."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.consolidation import HebbianMemoryGraph
from models.hcann import HCANN
from tests.conftest import bind_text_table
from train.continual_learner import ContinualLearner
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


def _scene_cfg(**overrides) -> HCANNConfig:
    cfg = HCANNConfig()
    cfg.use_mock_encoder = True
    cfg.enable_dreaming = False
    cfg.episodic_mode = True
    cfg.ec_dim = 32
    cfg.dg_dim = 16
    cfg.hpc_size = 16
    cfg.semantic_dim = 16
    cfg.wm_dim = 16
    cfg.max_patterns = 80
    cfg.max_nodes = 80
    cfg.ctx_dim = 16
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _scene_table(dim: int, n_scenes: int, per_scene: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    centers = torch.nn.functional.normalize(torch.randn(n_scenes, dim, generator=gen), dim=-1)
    mapping = {}
    ids = []
    for s in range(n_scenes):
        for j in range(per_scene):
            eid = f"s{s}e{j}"
            mapping[eid] = centers[s].clone()
            ids.append(eid)
    return mapping, ids


def _count_segments(signal: str) -> int:
    cfg = _scene_cfg(seg_signal=signal)
    device = torch.device("cpu")
    model = HCANN(cfg).to(device)
    mapping, ids = _scene_table(cfg.semantic_dim, 3, 10, seed=2)
    bind_text_table(model, mapping)
    learner = ContinualLearner(model, cfg, device)
    for eid in ids:
        learner.train_step(texts=[eid], episode_id=[eid])
    return len(learner.episodes)


def test_three_scene_segmentation_for_each_signal(capsys):
    counts = {signal: _count_segments(signal) for signal in ("dg", "sem", "mix")}
    print("seg_signal counts", counts)
    for signal, count in counts.items():
        assert abs(count - 3) <= 1, signal
    closest = min(counts, key=lambda name: abs(counts[name] - 3))
    print("seg_signal le plus proche de 3 scenes:", closest)


def test_boundary_event_uses_the_new_context_without_extra_pattern():
    cfg = _scene_cfg(seg_signal="sem", reencode_boundary=True)
    device = torch.device("cpu")
    model = HCANN(cfg).to(device)
    mapping, ids = _scene_table(cfg.semantic_dim, 2, 6, seed=4)
    bind_text_table(model, mapping)
    learner = ContinualLearner(model, cfg, device)
    before = model.ca3.num_patterns
    for eid in ids:
        learner.train_step(texts=[eid], episode_id=[eid])
        assert model.ca3.num_patterns == before + 1 or model.ca3.num_patterns == learner.graph.num_valid
        before = model.ca3.num_patterns
    assert model.ca3.num_patterns == len(ids)
    g = learner.graph.graph
    boundary_id = None
    for eid in ids:
        if g.nodes[eid]["data"].get("event_idx") == 0 and g.nodes[eid]["data"].get("segment_id") != "seg_0":
            boundary_id = eid
            break
    assert boundary_id is not None
    bctx = torch.tensor(g.nodes[boundary_id]["data"]["ctx_code"], dtype=torch.float32)
    prev_ids = [eid for eid in ids if g.nodes[eid]["data"]["segment_id"] == "seg_0"][-3:]
    members = learner.episodes.members(g.nodes[boundary_id]["data"]["segment_id"])
    next_ids = [eid for eid in members if eid != boundary_id][:3]
    assert len(prev_ids) == 3 and len(next_ids) == 3

    def _cos(other_id: str) -> float:
        other = torch.tensor(g.nodes[other_id]["data"]["ctx_code"], dtype=torch.float32)
        return float(torch.nn.functional.cosine_similarity(bctx, other, dim=0))

    near = sum(_cos(eid) for eid in next_ids) / 3
    far = sum(_cos(eid) for eid in prev_ids) / 3
    assert near > far
