#!/usr/bin/env python3
"""Diagnostic d'éviction : legacy vs oubli unifié. Aucun correctif — chiffres.

Usage :
  python scripts/diag_eviction.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.hcann import HCANN
from tests.conftest import bind_text_table
from train.continual_learner import ContinualLearner
from utils.config import HCANNConfig


def _run(label: str, unified: bool, max_nodes: int = 50) -> dict:
    cfg = HCANNConfig()
    cfg.use_mock_encoder = True
    cfg.episodic_mode = True
    cfg.enable_dreaming = True
    cfg.consolidation_freq = 10
    cfg.dream_cycles = 1
    cfg.dream_batch_size = 8
    cfg.max_nodes = max_nodes
    cfg.max_patterns = max_nodes
    cfg.ec_dim = 32
    cfg.dg_dim = 16
    cfg.hpc_size = 16
    cfg.semantic_dim = 16
    cfg.wm_dim = 16
    cfg.ctx_dim = 0
    cfg.seg_signal = "sem"
    cfg.prune_age_thresh = 0.0
    cfg.prune_weight_thresh = 1e9
    cfg.schema_every = 0  # pas de schémas ici : comparer l'éviction seule
    if not unified:
        # Legacy : pas de MemoryManager (éviction graphe interne + FIFO CA3)
        # On désactive le câblage en passant par un flag local après init.
        pass

    device = torch.device("cpu")
    model = HCANN(cfg).to(device)
    n_events = 5 * max_nodes
    gen = torch.Generator().manual_seed(0)
    mapping = {
        f"e{i}": F.normalize(torch.randn(cfg.semantic_dim, generator=gen), dim=0)
        for i in range(n_events)
    }
    # 10 % "importants" : cluster rare (axe dédié) + rappels
    important = {f"e{i}" for i in range(0, n_events, 10)}
    for eid in important:
        vec = torch.zeros(cfg.semantic_dim)
        vec[0] = 1.0
        vec[1] = 0.15 * (hash(eid) % 7)
        mapping[eid] = F.normalize(vec, dim=0)

    bind_text_table(model, mapping)
    learner = ContinualLearner(model, cfg, device)
    if not unified:
        # Retirer les handlers unifiés → comportement legacy
        learner.graph.graph.eviction_policy = None
        model.ca3.set_eviction_handler(None)
        learner.memory_manager = None

    inserted_at = {}
    for i in range(n_events):
        eid = f"e{i}"
        learner.train_step(texts=[eid], episode_id=[eid])
        inserted_at[eid] = i
        if eid in important and eid in learner.graph.graph.nodes:
            # 3 rappels pour valoriser (unifié) / access (legacy)
            pat = model.ca3.patterns[model.ca3._key_to_idx[eid]].detach() if eid in model.ca3._key_to_idx else None
            if pat is not None and learner._recall is not None:
                for _ in range(3):
                    learner.recall(cue_dg=pat, touch=True)

    g = learner.graph.graph
    survivors = [nid for nid in g.nodes if (g.nodes[nid].get("data") or {}).get("type") != "schema"]
    ages = [n_events - 1 - inserted_at[nid] for nid in survivors if nid in inserted_at]
    recent_cut = int(0.8 * (n_events - 1))
    recent_ids = {f"e{i}" for i in range(recent_cut, n_events)}
    recent_alive = sum(1 for nid in survivors if nid in recent_ids) / max(len(recent_ids), 1)

    # Survie de la moitié la plus récente de la capacité
    half_cap = list(range(n_events - max_nodes // 2, n_events))
    half_ids = {f"e{i}" for i in half_cap}
    half_alive = sum(1 for nid in survivors if nid in half_ids) / max(len(half_ids), 1)

    # Survie des importants (surtout les anciens)
    old_important = {eid for eid in important if inserted_at[eid] < n_events // 2}
    imp_alive = sum(1 for eid in old_important if eid in g.nodes) / max(len(old_important), 1)

    deg_surv = []
    for nid in survivors:
        idx = g.id_to_index[nid]
        deg_surv.append(float((g.edges[idx, : g.count] > 0.05).sum().item()))
    evicted = [f"e{i}" for i in range(n_events) if f"e{i}" not in g.nodes]

    hist = Counter()
    for age in ages:
        bucket = 10 * (age // 10)
        hist[bucket] += 1

    mean_age = float(np.mean(ages)) if ages else 0.0
    bias_old = recent_alive < 0.35 and mean_age > 0.4 * n_events

    print(f"=== {label} ===")
    print(f"max_nodes={max_nodes} n_events={n_events} survivors={len(survivors)}")
    print("histogramme ages (bucket -> count):")
    for bucket in sorted(hist):
        print(f"  [{bucket}, {bucket + 10}) : {hist[bucket]}")
    print(f"part des 20% les plus recents encore presents : {100 * recent_alive:.1f}%")
    print(f"survie moitie recente capacite ({max_nodes // 2}) : {100 * half_alive:.1f}%")
    print(f"survie importants anciens : {100 * imp_alive:.1f}%")
    print(f"degre moyen survivants : {np.mean(deg_surv) if deg_surv else 0:.2f}")
    print(f"n_evinces : {len(evicted)}")
    print(
        "Verdict : biais vers les anciens : "
        + ("oui" if bias_old else "non")
        + f" (âge moyen survivants={mean_age:.1f}, récents vivants={100 * recent_alive:.1f}%)."
    )
    return {
        "label": label,
        "survivors": len(survivors),
        "mean_age": mean_age,
        "recent_alive": recent_alive,
        "half_alive": half_alive,
        "imp_alive": imp_alive,
        "n_evicted": len(evicted),
    }


def main() -> int:
    _run("legacy", unified=False)
    print()
    _run("unifie", unified=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
