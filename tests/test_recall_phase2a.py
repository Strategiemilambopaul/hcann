"""Phase 2a : rappel ordonné, confiance, reconsolidation."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hcann import HCANN
from tests.conftest import bind_text_table
from train.continual_learner import ContinualLearner
from utils.config import HCANNConfig


def _cfg(**overrides) -> HCANNConfig:
    cfg = HCANNConfig()
    cfg.use_mock_encoder = True
    cfg.enable_dreaming = False
    cfg.episodic_mode = True
    cfg.ec_dim = 128
    cfg.dg_dim = 128
    cfg.hpc_size = 128
    cfg.semantic_dim = 64
    cfg.wm_dim = 64
    cfg.max_patterns = 500
    cfg.max_nodes = 500
    cfg.ctx_dim = 0
    cfg.seg_signal = "sem"
    cfg.hopfield_beta = 40.0
    cfg.hopfield_steps = 8
    cfg.ec_sparsity = 0.08
    cfg.dg_sparsity = 0.08
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _auroc(scores: list[float], labels: list[int]) -> float:
    pairs = 0.0
    correct = 0.0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if labels[i] == labels[j]:
                continue
            pairs += 1
            if scores[i] == scores[j]:
                correct += 0.5
            elif (labels[i] > labels[j]) == (scores[i] > scores[j]):
                correct += 1
    return correct / max(pairs, 1.0)


def _ortho_scenes(n_scenes: int, per_scene: int, dim: int):
    """Scènes orthogonales ; chaque événement a une petite composante unique."""
    need = n_scenes + per_scene
    assert dim >= need
    mapping = {}
    ids = []
    for s in range(n_scenes):
        for j in range(per_scene):
            eid = f"s{s}e{j}"
            vec = torch.zeros(dim)
            vec[s] = 1.0
            vec[n_scenes + j] = 0.2
            mapping[eid] = F.normalize(vec, dim=0)
            ids.append(eid)
    return mapping, ids


def _learner(cfg):
    device = torch.device("cpu")
    model = HCANN(cfg).to(device)
    return ContinualLearner(model, cfg, device), model


def _stored(model, eid: str) -> torch.Tensor:
    return model.ca3.patterns[model.ca3._key_to_idx[eid]].detach().clone()


def test_partial_cue_recovers_stored_pattern():
    cfg = _cfg()
    learner, model = _learner(cfg)
    torch.manual_seed(0)
    ids = [f"e{i}" for i in range(200)]
    mapping = {
        eid: F.normalize(torch.randn(cfg.semantic_dim), dim=0) for eid in ids
    }
    bind_text_table(model, mapping)
    for eid in ids:
        learner.train_step(texts=[eid], episode_id=[eid])
    hits = 0
    probes = ids[::2]
    gen = torch.Generator().manual_seed(1)
    for eid in probes:
        stored = _stored(model, eid)
        # 50 % des dimensions actives (k-WTA) : masque uniforme hors support trop dur.
        active = (stored.abs() > 1e-6).nonzero(as_tuple=False).reshape(-1)
        if active.numel() == 0:
            continue
        perm = active[torch.randperm(active.numel(), generator=gen)]
        keep = perm[: max(1, active.numel() // 2)]
        mask = torch.zeros_like(stored)
        mask[keep] = 1.0
        cue = F.normalize(stored * mask + 0.01 * torch.randn(stored.shape, generator=gen), dim=0)
        result = learner.recall(cue_dg=cue, touch=False)
        if result.key == eid:
            hits += 1
    assert hits / len(probes) >= 0.9


def test_segment_and_chain_order():
    cfg = _cfg()
    learner, model = _learner(cfg)
    mapping, ids = _ortho_scenes(5, 8, cfg.semantic_dim)
    bind_text_table(model, mapping)
    for eid in ids:
        learner.train_step(texts=[eid], episode_id=[eid])
    # Pose les segments attendus : le test cible l'ordre du rappel, pas le segmenter.
    from memory.episodes import EpisodeIndex

    index = EpisodeIndex(cfg)
    index._segments = {}
    for s in range(5):
        sid = f"seg_{s}"
        index._segments[sid] = []
        for j in range(8):
            eid = f"s{s}e{j}"
            index._segments[sid].append(eid)
            data = learner.graph.graph.nodes[eid]["data"]
            data["segment_id"] = sid
            data["event_idx"] = j
    index._seg_count = 4
    index._current = "seg_4"
    learner.episodes = index
    learner._recall.episodes = index

    cue_id = "s2e3"
    result = learner.recall(cue_dg=_stored(model, cue_id), scope="segment", touch=False)
    got = [e.node_id for e in result.events]
    assert got == [f"s2e{j}" for j in range(3, 8)]
    chain = learner.recall(cue_dg=_stored(model, cue_id), scope="chain", follow=6, touch=False)
    chain_ids = [e.node_id for e in chain.events]
    assert chain_ids[0] == "s2e3"
    assert "s2e7" in chain_ids
    assert any(eid.startswith("s3") for eid in chain_ids)


def test_abstention_on_unseen_cues():
    cfg = _cfg()
    learner, model = _learner(cfg)
    mapping, ids = _ortho_scenes(5, 8, cfg.semantic_dim)
    bind_text_table(model, mapping)
    for eid in ids:
        learner.train_step(texts=[eid], episode_id=[eid])
    unknown = 0
    torch.manual_seed(0)
    for _ in range(40):
        cue = F.normalize(torch.randn(cfg.dg_dim), dim=0)
        result = learner.recall(cue_dg=cue, touch=False)
        if result.level == "inconnu" and not result.events and result.confidence < cfg.recall_lo:
            unknown += 1
    assert unknown / 40 >= 0.95


def test_calibration_raises_auroc():
    cfg = _cfg()
    learner, model = _learner(cfg)
    mapping, ids = _ortho_scenes(10, 8, cfg.semantic_dim)
    bind_text_table(model, mapping)
    for eid in ids:
        learner.train_step(texts=[eid], episode_id=[eid])
    features = []
    labels = []
    torch.manual_seed(1)
    for eid in ids:
        stored = _stored(model, eid)
        for amp in (0.0, 0.1, 0.3, 0.8, 1.5):
            cue = F.normalize(stored + amp * torch.randn_like(stored), dim=0)
            result = learner.recall(cue_dg=cue, touch=False)
            features.append(result.features)
            labels.append(1 if result.key == eid else 0)
    for _ in range(40):
        cue = F.normalize(torch.randn(cfg.dg_dim), dim=0)
        result = learner.recall(cue_dg=cue, touch=False)
        features.append(result.features)
        labels.append(0)
    learner._recall.fit_calibration(features, labels, steps=800)
    scores = [learner._recall._confidence(feat) for feat in features]
    assert _auroc(scores, labels) >= 0.85


def test_reconsolidation_stability_and_cue_shift():
    cfg = _cfg(recon_target="attractor", recon_lr=0.05)
    learner, model = _learner(cfg)
    mapping, ids = _ortho_scenes(16, 1, cfg.semantic_dim)
    bind_text_table(model, mapping)
    for eid in ids:
        learner.train_step(texts=[eid], episode_id=[eid])
    target = "s1e0"
    original = _stored(model, target)
    idxs, cos_before = model.ca3.nearest(original, k=2)
    margin_before = float(cos_before[0] - (cos_before[1] if cos_before.numel() > 1 else 0))
    for _ in range(50):
        noisy = F.normalize(original + 0.1 * torch.randn_like(original), dim=0)
        learner.recall(cue_dg=noisy, touch=True)
    after = _stored(model, target)
    assert float((after * original).sum()) >= 0.9
    top, cos_after = model.ca3.nearest(original, k=2)
    assert model.ca3.key_of(int(top[0].item())) == target
    margin_after = float(cos_after[0] - (cos_after[1] if cos_after.numel() > 1 else 0))
    assert margin_after >= 0.9 * margin_before

    cfg2 = _cfg(recon_target="cue", recon_lr=0.35, recon_min_conf=0.0)
    learner2, model2 = _learner(cfg2)
    bind_text_table(model2, mapping)
    for eid in ids:
        learner2.train_step(texts=[eid], episode_id=[eid])
    original2 = _stored(model2, target)
    cue = F.normalize(original2 + 0.5 * torch.randn_like(original2), dim=0)
    before_pat = _stored(model2, target).clone()
    before_cos = float((before_pat * cue).sum())
    # Forcer une reconsolidation même si le niveau est flou.
    learner2._recall._reconsolidate(target, cue.unsqueeze(0), cue.unsqueeze(0), [target])
    after_pat = _stored(model2, target)
    after_cos = float((after_pat * cue).sum())
    assert after_cos > before_cos + 1e-4
    assert not torch.allclose(before_pat, after_pat)
    top2, _ = model2.ca3.nearest(original2, k=1)
    assert model2.ca3.key_of(int(top2[0].item())) == target


def test_touch_true_and_false_side_effects():
    cfg = _cfg()
    learner, model = _learner(cfg)
    mapping, ids = _ortho_scenes(3, 6, cfg.semantic_dim)
    bind_text_table(model, mapping)
    for eid in ids:
        learner.train_step(texts=[eid], episode_id=[eid])
    target = "s0e1"
    patterns_before = model.ca3.patterns.detach().clone()
    access_before = learner.graph.graph.nodes[target]["access_count"]
    learner.recall(cue_dg=_stored(model, target), touch=False)
    assert torch.equal(model.ca3.patterns, patterns_before)
    assert learner.graph.graph.nodes[target]["access_count"] == access_before
    learner.recall(cue_dg=_stored(model, target), touch=True)
    assert learner.graph.graph.nodes[target]["access_count"] == access_before + 1
    assert learner.graph.graph.nodes[target]["data"].get("recall_count", 0) >= 1


def test_episodic_mode_false_keeps_learner_without_recall_engine():
    cfg = _cfg(episodic_mode=False)
    learner, model = _learner(cfg)
    assert learner._recall is None
    bind_text_table(model, {"a": torch.ones(cfg.semantic_dim)})
    learner.train_step(texts=["a"], episode_id=["a"])
    result = learner.recall(cue_texts=["a"])
    assert result.level == "inconnu"
    assert result.events == []
