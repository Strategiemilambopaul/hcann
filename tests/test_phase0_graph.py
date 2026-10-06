"""Phase 0 : graphe plein, touch, auto-lien, séquences de rêve, mémoire de travail."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.consolidation import HebbianConsolidation, HebbianMemoryGraph
from memory.dreaming import DreamingPhase
from memory.replay_buffer import EpisodicBuffer
from models.cortex import WorkingMemory
from models.hcann import HCANN
from train.continual_learner import ContinualLearner
from utils.config import HCANNConfig


def _graph_config(max_nodes: int = 10) -> HCANNConfig:
    cfg = HCANNConfig()
    cfg.max_nodes = max_nodes
    cfg.hpc_size = 8
    cfg.dg_dim = 8
    cfg.semantic_dim = 8
    cfg.wm_dim = 8
    cfg.use_mock_encoder = True
    cfg.enable_dreaming = False
    return cfg


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-8)


def test_full_graph_reuses_slots_and_keeps_last_insert():
    cfg = _graph_config(10)
    g = HebbianMemoryGraph(cfg, torch.device("cpu"))
    for i in range(25):
        emb = _vec(i)
        g.add_node(f"n{i}", {"i": i}, emb, sem_embedding=emb, dg_embedding=emb)
    assert g.num_valid <= 10
    assert "n24" in g.nodes
    hits = g.retrieve_semantic(_vec(24), k=50)
    assert hits
    assert all(h["id"] is not None for h in hits)
    assert len(hits) <= g.num_valid


def test_touch_false_does_not_bump_access_count():
    cfg = _graph_config(4)
    g = HebbianMemoryGraph(cfg, torch.device("cpu"))
    emb = _vec(1)
    g.add_node("a", {"text": "a"}, emb, sem_embedding=emb, dg_embedding=emb)
    g.spreading_activation(emb, k=1, touch=False)
    assert g.nodes["a"]["access_count"] == 0
    assert float(g.access_counts[g.id_to_index["a"]].item()) == 0.0
    g.spreading_activation(emb, k=1, touch=True)
    assert g.nodes["a"]["access_count"] == 1
    assert float(g.access_counts[g.id_to_index["a"]].item()) == 1.0


def test_reencode_does_not_self_link():
    cfg = _graph_config(8)
    cfg.ec_dim = 32
    cfg.dg_dim = 16
    cfg.hpc_size = 16
    cfg.semantic_dim = 16
    cfg.wm_dim = 16
    device = torch.device("cpu")
    model = HCANN(cfg).to(device)
    learner = ContinualLearner(model, cfg, device)
    learner.train_step(texts=["same visit"], episode_id=["ep"])
    learner.train_step(texts=["same visit"], episode_id=["ep"])
    g = learner.graph.graph
    idx = g.id_to_index["ep"]
    assert float(g.edges[idx, idx].item()) == 0.0


def test_sample_sequences_are_contiguous_and_sleep_links_only_those():
    buf = EpisodicBuffer()
    for i in range(8):
        buf.push(torch.zeros(8), torch.zeros(8), {"episode_id": f"e{i}"})
    windows = buf.sample_sequences(4, 3)
    assert windows
    for window in windows:
        nums = [int(item["meta"]["episode_id"][1:]) for item in window]
        assert nums == list(range(nums[0], nums[0] + 3))

    cfg = _graph_config(16)
    cfg.ec_dim = 32
    cfg.dg_dim = 8
    cfg.hpc_size = 8
    cfg.semantic_dim = 8
    cfg.wm_dim = 8
    cfg.dream_cycles = 1
    cfg.dream_batch_size = 2
    cfg.dream_seq_len = 3
    cfg.episodic_mode = False
    device = torch.device("cpu")
    model = HCANN(cfg).to(device)
    graph = HebbianConsolidation(cfg, device)
    order = []
    for i in range(8):
        emb = _vec(100 + i)
        eid = f"e{i}"
        order.append(eid)
        graph.add_node(eid, {"text": eid}, emb, sem_embedding=emb, dg_embedding=emb)
    torch.manual_seed(0)
    DreamingPhase.run_sleep_cycle(buf, model, graph, cfg, n_cycles=1)
    g = graph.graph
    for i, src in enumerate(order):
        si = g.id_to_index[src]
        for j, dst in enumerate(order):
            if abs(i - j) == 1:
                continue
            assert float(g.edges[si, g.id_to_index[dst]].item()) == 0.0


def test_fast_encode_store_false_is_side_effect_free_and_wm_backward_works():
    cfg = _graph_config(8)
    cfg.ec_dim = 32
    cfg.dg_dim = 16
    cfg.hpc_size = 16
    cfg.semantic_dim = 16
    cfg.wm_dim = 16
    cfg.ctx_dim = 0
    model = HCANN(cfg)
    model.fast_encode(texts=["hello"], store=True)
    slots = model.wm.slots.detach().clone()
    ctx = model.hippocampus.temporal_ctx.context.detach().clone()
    model.fast_encode(texts=["hello"], store=False)
    assert torch.equal(model.wm.slots, slots)
    assert torch.equal(model.hippocampus.temporal_ctx.context, ctx)

    wm = WorkingMemory(cfg)
    x = torch.randn(2, cfg.wm_dim, requires_grad=True)
    read, _ = wm(x, write=True)
    read.sum().backward()
    assert x.grad is not None
