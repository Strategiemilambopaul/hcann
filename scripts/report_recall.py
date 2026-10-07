#!/usr/bin/env python3
"""Rapport phase 2a : rappel, abstention, calibration, reconsolidation, fam_floor.

Aucun correctif — chiffres seulement.
Usage : python scripts/report_recall.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _stored(model, eid: str) -> torch.Tensor:
    return model.ca3.patterns[model.ca3._key_to_idx[eid]].detach().clone()


def _partial_cue(stored: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    active = (stored.abs() > 1e-6).nonzero(as_tuple=False).reshape(-1)
    if active.numel() == 0:
        return F.normalize(stored + 0.01 * torch.randn(stored.shape, generator=gen), dim=0)
    perm = active[torch.randperm(active.numel(), generator=gen)]
    keep = perm[: max(1, active.numel() // 2)]
    mask = torch.zeros_like(stored)
    mask[keep] = 1.0
    return F.normalize(stored * mask + 0.01 * torch.randn(stored.shape, generator=gen), dim=0)


def main() -> int:
    cfg = _cfg()
    device = torch.device("cpu")
    model = HCANN(cfg).to(device)
    learner = ContinualLearner(model, cfg, device)
    mapping, ids = _ortho_scenes(10, 8, cfg.semantic_dim)
    bind_text_table(model, mapping)
    for eid in ids:
        learner.train_step(texts=[eid], episode_id=[eid])

    # Top-1 indices partiels
    gen = torch.Generator().manual_seed(1)
    probes = ids[::2]
    hits = 0
    for eid in probes:
        cue = _partial_cue(_stored(model, eid), gen)
        result = learner.recall(cue_dg=cue, touch=False)
        if result.key == eid:
            hits += 1
    top1 = hits / max(len(probes), 1)
    print(f"top1_partial_cues: {100 * top1:.1f}% ({hits}/{len(probes)})")

    # Abstention sur vecteurs inconnus
    unknown = 0
    torch.manual_seed(0)
    n_unk = 40
    for _ in range(n_unk):
        cue = F.normalize(torch.randn(cfg.dg_dim), dim=0)
        result = learner.recall(cue_dg=cue, touch=False)
        if result.level == "inconnu" and not result.events:
            unknown += 1
    abst = unknown / n_unk
    print(f"abstention_unknown: {100 * abst:.1f}% ({unknown}/{n_unk})")

    # AUROC après calibration
    features, labels = [], []
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
    before = _auroc([learner._recall._confidence(f) for f in features], labels)
    learner._recall.fit_calibration(features, labels, steps=800)
    after = _auroc([learner._recall._confidence(f) for f in features], labels)
    print(f"auroc_confidence: before={before:.3f} after_calibration={after:.3f}")

    # Dérive reconsolidation (50 rappels)
    target = "s1e0"
    original = _stored(model, target)
    _, cos_before = model.ca3.nearest(original, k=2)
    margin_before = float(cos_before[0] - (cos_before[1] if cos_before.numel() > 1 else 0))
    for _ in range(50):
        noisy = F.normalize(original + 0.1 * torch.randn_like(original), dim=0)
        learner.recall(cue_dg=noisy, touch=True)
    after_pat = _stored(model, target)
    cos_pat = float((after_pat * original).sum())
    _, cos_after = model.ca3.nearest(original, k=2)
    margin_after = float(cos_after[0] - (cos_after[1] if cos_after.numel() > 1 else 0))
    print(
        f"reconsolidation: cos_pattern_original={cos_pat:.4f} "
        f"margin_before={margin_before:.4f} margin_after={margin_after:.4f} "
        f"margin_ratio={margin_after / max(margin_before, 1e-9):.4f}"
    )

    # Effet recall_fam_floor
    def _true_reject_rate(floor: float) -> float:
        cfg_f = _cfg(recall_fam_floor=floor)
        m = HCANN(cfg_f).to(device)
        lr = ContinualLearner(m, cfg_f, device)
        bind_text_table(m, mapping)
        for eid in ids:
            lr.train_step(texts=[eid], episode_id=[eid])
        rejected = 0
        total = 0
        gen_f = torch.Generator().manual_seed(2)
        for eid in ids:
            cue = _partial_cue(_stored(m, eid), gen_f)
            # Familiarité brute sans plancher
            dg = cue.unsqueeze(0)
            fam = float(m.ca3.max_pattern_similarity(dg)[0].item())
            result = lr.recall(cue_dg=cue, touch=False)
            total += 1
            if fam >= floor and result.key != eid:
                continue
            if fam < floor and result.level == "inconnu":
                # vrai souvenir potentiellement rejeté par le plancher
                idxs, _ = m.ca3.nearest(m.ca3.complete(dg), k=1)
                if idxs.numel() and m.ca3.key_of(int(idxs[0].item())) == eid:
                    rejected += 1
            elif result.key != eid and result.level == "inconnu":
                # check nearest would have been correct
                completed = m.ca3.complete(dg)
                idxs, _ = m.ca3.nearest(completed, k=1)
                if idxs.numel() and m.ca3.key_of(int(idxs[0].item())) == eid:
                    rejected += 1
        return rejected / max(total, 1)

    # Mesure directe : parmi les cues partiels dont le nearest est correct,
    # combien passent en "inconnu" à cause du plancher.
    def _floor_reject(floor: float) -> tuple[float, int, int]:
        cfg_f = _cfg(recall_fam_floor=floor)
        m = HCANN(cfg_f).to(device)
        lr = ContinualLearner(m, cfg_f, device)
        bind_text_table(m, mapping)
        for eid in ids:
            lr.train_step(texts=[eid], episode_id=[eid])
        true_hits = 0
        rejected = 0
        gen_f = torch.Generator().manual_seed(3)
        for eid in ids:
            cue = _partial_cue(_stored(m, eid), gen_f)
            dg = cue.unsqueeze(0)
            completed = m.ca3.complete(dg)
            idxs, _ = m.ca3.nearest(completed, k=1)
            if not idxs.numel() or m.ca3.key_of(int(idxs[0].item())) != eid:
                continue
            true_hits += 1
            fam = float(m.ca3.max_pattern_similarity(dg)[0].item())
            result = lr.recall(cue_dg=cue, touch=False)
            if fam < floor and result.level == "inconnu":
                rejected += 1
        return rejected / max(true_hits, 1), rejected, true_hits

    r0, n0, t0 = _floor_reject(0.0)
    r_def, n_def, t_def = _floor_reject(float(cfg.recall_fam_floor))
    print(
        f"recall_fam_floor: without={100 * r0:.1f}% rejected of true nearest "
        f"({n0}/{t0}); with_floor={100 * r_def:.1f}% ({n_def}/{t_def}) "
        f"floor={cfg.recall_fam_floor}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
