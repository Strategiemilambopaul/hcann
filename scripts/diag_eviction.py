#!/usr/bin/env python3
"""Diagnostic d'éviction du graphe. Aucun correctif — chiffres seulement.

Usage :
  python scripts/diag_eviction.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.hcann import HCANN
from tests.conftest import bind_text_table
from train.continual_learner import ContinualLearner
from utils.config import HCANNConfig


def main() -> int:
    max_nodes = 50
    cfg = HCANNConfig()
    cfg.use_mock_encoder = True
    cfg.episodic_mode = True
    cfg.enable_dreaming = True
    cfg.consolidation_freq = 10
    cfg.dream_cycles = 1
    cfg.dream_batch_size = 8
    cfg.max_nodes = max_nodes
    cfg.max_patterns = 400
    cfg.ec_dim = 32
    cfg.dg_dim = 16
    cfg.hpc_size = 16
    cfg.semantic_dim = 16
    cfg.wm_dim = 16
    cfg.ctx_dim = 0
    cfg.seg_signal = "sem"
    cfg.prune_age_thresh = 0.0
    cfg.prune_weight_thresh = 1e9

    device = torch.device("cpu")
    model = HCANN(cfg).to(device)
    n_events = 5 * max_nodes
    gen = torch.Generator().manual_seed(0)
    mapping = {
        f"e{i}": torch.nn.functional.normalize(torch.randn(cfg.semantic_dim, generator=gen), dim=0)
        for i in range(n_events)
    }
    bind_text_table(model, mapping)
    learner = ContinualLearner(model, cfg, device)
    inserted_at = {}
    for i in range(n_events):
        eid = f"e{i}"
        learner.train_step(texts=[eid], episode_id=[eid])
        inserted_at[eid] = i

    g = learner.graph.graph
    survivors = [nid for nid in g.nodes]
    ages = [n_events - 1 - inserted_at[nid] for nid in survivors]
    recent_cut = int(0.8 * (n_events - 1))
    recent_ids = {f"e{i}" for i in range(recent_cut, n_events)}
    recent_alive = sum(1 for nid in survivors if nid in recent_ids) / max(len(recent_ids), 1)

    deg_surv = []
    for nid in survivors:
        idx = g.id_to_index[nid]
        deg_surv.append(float((g.edges[idx, : g.count] > 0.05).sum().item()))
    evicted = [f"e{i}" for i in range(n_events) if f"e{i}" not in g.nodes]
    # degré au moment de l'éviction inconnu : on approxime 0 pour les absents
    deg_evicted_proxy = 0.0

    hist = Counter()
    for age in ages:
        bucket = 10 * (age // 10)
        hist[bucket] += 1

    print(f"max_nodes={max_nodes} n_events={n_events} survivors={len(survivors)}")
    print("histogramme ages (bucket -> count):")
    for bucket in sorted(hist):
        print(f"  [{bucket}, {bucket + 10}) : {hist[bucket]}")
    print(f"part des 20% les plus recents encore presents : {100 * recent_alive:.1f}%")
    print(f"degre moyen survivants : {np.mean(deg_surv):.2f}")
    print(f"degre moyen evinces (proxy=0, non observé) : {deg_evicted_proxy:.2f}")
    print(f"n_evinces : {len(evicted)}")
    mean_age = float(np.mean(ages)) if ages else 0.0
    # Si les survivants sont surtout jeunes (âge faible), biais vers les récents.
    # Biais vers les anciens = survivants plutôt âgés et 20% récents peu présents.
    bias_old = recent_alive < 0.35 and mean_age > 0.4 * n_events
    print(
        "Verdict : biais vers les anciens : "
        + ("oui" if bias_old else "non")
        + f" (âge moyen survivants={mean_age:.1f}, récents vivants={100 * recent_alive:.1f}%)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
