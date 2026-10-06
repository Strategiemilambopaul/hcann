"""Phase 1 : nouveauté mémoire, contexte temporel, segments, clés, successeurs."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.episodes import EpisodeIndex
from models.hcann import HCANN
from models.hippocampus import TrisynapticHippocampus
from models.temporal_context import TemporalContext
from train.continual_learner import ContinualLearner
from utils.config import HCANNConfig


def _cfg(**overrides) -> HCANNConfig:
    cfg = HCANNConfig()
    cfg.use_mock_encoder = True
    cfg.enable_dreaming = False
    cfg.ec_dim = 32
    cfg.dg_dim = 16
    cfg.hpc_size = 16
    cfg.semantic_dim = 16
    cfg.wm_dim = 16
    cfg.max_patterns = 50
    cfg.max_nodes = 50
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_novelty_mem_is_computed_before_store():
    cfg = _cfg()
    hpc = TrisynapticHippocampus(cfg)
    sdr = torch.zeros(1, cfg.ec_dim)
    sdr[0, :8] = 1.0
    sem = F.normalize(torch.randn(1, cfg.semantic_dim), dim=-1)
    first = hpc(sdr, sem, store=True)
    assert float(first["novelty_mem"].item()) > 0.5
    second = hpc(sdr, sem, store=True)
    assert float(second["novelty_mem"].item()) < 0.1
    other = torch.zeros(1, cfg.ec_dim)
    other[0, 16:28] = 1.0
    rand_sem = F.normalize(torch.randn(1, cfg.semantic_dim), dim=-1)
    third = hpc(other, rand_sem, store=False)
    assert float(third["novelty_mem"].item()) > 0.5


def test_fast_encode_shapes_unchanged_when_context_off():
    cfg = _cfg(ctx_dim=0, episodic_mode=False)
    model = HCANN(cfg)
    grid_dim = len(cfg.grid_periods) * 4
    assert model.hippocampus.dg.W_expansion.shape[0] == cfg.ec_dim + grid_dim
    ca3, sem, wm, novelty, dg, extras = model.fast_encode(texts=["episode"])
    assert ca3.shape == (1, cfg.dg_dim)
    assert sem.shape == (1, cfg.semantic_dim)
    assert wm.shape == (1, cfg.wm_dim)
    assert novelty.shape == (1,)
    assert dg.shape == (1, cfg.dg_dim)
    assert "is_novel" in extras
    assert "novelty" in extras
    assert extras["ctx_code"].shape == (1, 0)


def test_temporal_context_decays_and_jumps_at_boundary():
    cfg = _cfg(ctx_dim=16)
    ctx = TemporalContext(cfg)
    torch.manual_seed(0)
    states = []
    for _ in range(21):
        sem = F.normalize(torch.randn(1, cfg.semantic_dim), dim=-1)
        states.append(ctx.step(sem, advance=True).detach().clone())
    near = F.cosine_similarity(states[19], states[20], dim=-1).item()
    far = F.cosine_similarity(states[0], states[20], dim=-1).item()
    assert near > far
    before = ctx.context.detach().clone()
    ctx.boundary()
    after = F.cosine_similarity(before, ctx.context, dim=-1).item()
    assert after < near


def test_three_scenes_become_three_segments():
    cfg = _cfg(min_event_len=3, seg_k=1.5, mem_novelty_threshold=0.25)
    index = EpisodeIndex(cfg)
    pattern = [0.05] * 10 + [0.9] + [0.05] * 9 + [0.9] + [0.05] * 9
    for i, novelty in enumerate(pattern):
        index.add_event(f"e{i}", novelty)
    assert abs(len(index) - 3) <= 1
    for sid in list(index._segments):
        assert len(index.members(sid)) >= cfg.min_event_len


def test_same_key_is_an_upsert():
    cfg = _cfg(episodic_mode=True)
    device = torch.device("cpu")
    model = HCANN(cfg).to(device)
    learner = ContinualLearner(model, cfg, device)
    learner.train_step(texts=["visit"], episode_id=["visit-1"])
    patterns = model.ca3.num_patterns
    valid = learner.graph.num_valid
    learner.train_step(texts=["visit again"], episode_id=["visit-1"])
    assert model.ca3.num_patterns == patterns
    assert learner.graph.num_valid == valid


def test_successors_follow_encoding_order():
    cfg = _cfg(episodic_mode=True)
    device = torch.device("cpu")
    model = HCANN(cfg).to(device)
    learner = ContinualLearner(model, cfg, device)
    for name in ("A", "B", "C"):
        learner.train_step(texts=[name], episode_id=[name])
    succ_a = learner.graph.successors("A")
    succ_b = learner.graph.successors("B")
    succ_c = learner.graph.successors("C")
    assert succ_a[0]["id"] == "B"
    assert succ_b[0]["id"] == "C"
    assert not succ_c or succ_c[0]["id"] != "B"
