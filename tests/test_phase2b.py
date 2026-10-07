"""Phase 2b : oubli unifié, sync CA3/graphe, sémantisation."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ca3_hopfield import CA3ModernHopfield
from models.hcann import HCANN
from memory.dreaming import DreamingPhase
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
    cfg.schema_every = 0
    for key, value in overrides.items():
        setattr(cfg, key, value)
    # forget_tau suit max_nodes sauf override explicite
    if "forget_tau" not in overrides:
        cfg.forget_tau = int(cfg.max_nodes)
    return cfg


def _learner(cfg):
    device = torch.device("cpu")
    model = HCANN(cfg).to(device)
    return ContinualLearner(model, cfg, device), model


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


def test_ca3_remove_reindexes():
    cfg = _cfg(max_patterns=100)
    ca3 = CA3ModernHopfield(cfg)
    torch.manual_seed(0)
    keys = [f"k{i}" for i in range(20)]
    patterns = {}
    for k in keys:
        p = F.normalize(torch.randn(cfg.dg_dim), dim=0)
        patterns[k] = p
        ca3.store(p, key=k)
    gen = torch.Generator().manual_seed(1)
    remove = [keys[i] for i in torch.randperm(20, generator=gen)[:7].tolist()]
    for k in remove:
        assert ca3.remove(k) is True
    assert ca3.remove("unknown") is False
    remaining = [k for k in keys if k not in remove]
    assert set(ca3.keys()) == set(remaining)
    for k in remaining:
        idxs, _ = ca3.nearest(patterns[k], k=1)
        assert ca3.key_of(int(idxs[0].item())) == k
    # Indices cohérents : chaque clé pointe vers le bon pattern
    for k, idx in ca3._key_to_idx.items():
        assert 0 <= idx < ca3.num_patterns
        cos = float((ca3.patterns[idx] * patterns[k]).sum())
        assert cos > 0.99


def _overcap_stream(cfg, n_events=250, important_frac=0.1):
    learner, model = _learner(cfg)
    gen = torch.Generator().manual_seed(0)
    dim = cfg.semantic_dim
    mapping = {}
    important = set()
    snapshots: dict[str, torch.Tensor] = {}
    n_imp = int(important_frac * n_events)
    # Importants = famille rare (axe 0) mais empreintes distinctes pour CA3
    imp_indices = list(range(0, n_events // 2, max(1, (n_events // 2) // max(n_imp, 1))))[:n_imp]
    for i in range(n_events):
        eid = f"e{i}"
        if i in imp_indices:
            important.add(eid)
            vec = torch.zeros(dim)
            # Empreinte quasi-orthogonale + faible marqueur rare (nouveauté)
            vec[3 + (len(important) % max(1, dim - 4))] = 1.0
            vec[0] = 0.25
            mapping[eid] = F.normalize(vec, dim=0)
        else:
            mapping[eid] = F.normalize(torch.randn(dim, generator=gen), dim=0)
    bind_text_table(model, mapping)
    for i in range(n_events):
        eid = f"e{i}"
        learner.train_step(texts=[eid], episode_id=[eid])
        if eid in model.ca3._key_to_idx:
            snapshots[eid] = _stored(model, eid)
        if eid in important and eid in model.ca3._key_to_idx:
            pat = snapshots[eid]
            for _ in range(3):
                learner.recall(cue_dg=pat, touch=True)
            if eid in learner.graph.graph.nodes:
                d = learner.graph.graph.nodes[eid]["data"]
                d["surprise"] = max(float(d.get("surprise", 0.0)), 0.9)
                d["recall_count"] = max(int(d.get("recall_count", 0)), 3)
    return learner, model, important, n_events, snapshots


def test_overcapacity_unified_forgetting():
    max_n = 50
    cfg = _cfg(
        max_nodes=max_n,
        max_patterns=max_n,
        enable_dreaming=True,
        consolidation_freq=25,
        dream_cycles=1,
        schema_every=0,
        forget_frac=0.05,
    )
    learner, model, important, n_events, _ = _overcap_stream(cfg, n_events=250)
    g = learner.graph.graph
    survivors = [
        nid for nid in g.nodes
        if (g.nodes[nid].get("data") or {}).get("type") != "schema"
    ]
    half = {f"e{i}" for i in range(n_events - max_n // 2, n_events)}
    half_alive = sum(1 for nid in half if nid in g.nodes) / len(half)
    old_imp = {eid for eid in important if int(eid[1:]) < n_events // 2}
    old_imp_alive = sum(1 for eid in old_imp if eid in g.nodes) / max(len(old_imp), 1)
    ages = []
    for nid in survivors:
        born = int((g.nodes[nid].get("data") or {}).get("born", g.clock))
        # âge logique = clock - born ; insertion ≈ born pour événements uniques
        ages.append(g.clock - born)
    mean_age = sum(ages) / max(len(ages), 1)
    assert half_alive >= 0.8, f"half recent survival={half_alive:.3f}"
    assert old_imp_alive >= 0.6, f"old important survival={old_imp_alive:.3f}"
    assert mean_age < 220.5, f"mean survivor age={mean_age:.1f}"


def test_sync_ca3_graph_episodes():
    max_n = 50
    cfg = _cfg(
        max_nodes=max_n,
        max_patterns=max_n,
        enable_dreaming=True,
        consolidation_freq=20,
        dream_cycles=1,
        schema_every=0,
    )
    learner, model, _, _, _ = _overcap_stream(cfg, n_events=250)
    g = learner.graph.graph
    episode_ids = {
        nid for nid, node in g.nodes.items()
        if (node.get("data") or {}).get("type") != "schema"
    }
    assert set(model.ca3.keys()) == episode_ids
    stats = learner.memory_manager.reconcile(g, model.ca3, learner.episodes)
    assert stats["ca3_removed"] == 0
    assert stats["ca3_restored"] == 0
    assert stats["missing_trace"] == 0
    assert stats["episodes_removed"] == 0
    ep_ids = set(learner.episodes.all_ids())
    assert ep_ids <= set(g.nodes.keys())


def test_recall_under_capacity_pressure():
    max_n = 50
    cfg = _cfg(
        max_nodes=max_n,
        max_patterns=max_n,
        enable_dreaming=True,
        consolidation_freq=25,
        dream_cycles=1,
        schema_every=0,
        hopfield_beta=40.0,
        # Sous charge, le plancher 0.35 laisse trop de faux « clair » (mesuré ≈87 %).
        recall_fam_floor=0.65,
    )
    learner, model, important, n_events, snapshots = _overcap_stream(cfg, n_events=250)
    g = learner.graph.graph
    recent = [f"e{i}" for i in range(n_events - 20, n_events) if f"e{i}" in model.ca3._key_to_idx]
    gen = torch.Generator().manual_seed(4)
    hits = 0
    for eid in recent:
        cue = _partial_cue(_stored(model, eid), gen)
        result = learner.recall(cue_dg=cue, touch=False)
        if result.key == eid:
            hits += 1
    assert hits / max(len(recent), 1) >= 0.9, f"recent top1={hits}/{len(recent)}"

    alive_imp = [eid for eid in important if eid in model.ca3._key_to_idx]
    hits_imp = 0
    for eid in alive_imp:
        cue = _partial_cue(_stored(model, eid), gen)
        result = learner.recall(cue_dg=cue, touch=False)
        if result.key == eid:
            hits_imp += 1
    assert hits_imp / max(len(alive_imp), 1) >= 0.8, f"imp top1={hits_imp}/{len(alive_imp)}"

    evicted = [f"e{i}" for i in range(n_events) if f"e{i}" not in g.nodes and f"e{i}" in snapshots]
    clair = 0
    n_probe = min(40, len(evicted))
    gen2 = torch.Generator().manual_seed(5)
    for eid in evicted[:n_probe]:
        # Indice = empreinte CA3 d'origine (souvenir perdu), pas un ré-encodage ambigu
        cue = _partial_cue(snapshots[eid], gen2)
        result = learner.recall(cue_dg=cue, touch=False)
        if result.level == "clair":
            clair += 1
    assert clair / max(n_probe, 1) <= 0.1, f"clair on evicted={clair}/{n_probe}"


def test_recall_with_temporal_context():
    cfg = _cfg(ctx_dim=16, recall_ctx_mode="none", max_nodes=200, max_patterns=200)
    learner, model = _learner(cfg)
    ids = [f"t{i}" for i in range(40)]
    torch.manual_seed(0)
    mapping = {eid: F.normalize(torch.randn(cfg.semantic_dim), dim=0) for eid in ids}
    bind_text_table(model, mapping)
    for eid in ids:
        learner.train_step(texts=[eid], episode_id=[eid])
    # Indice = même texte (encodeur déterministe) ; bruit léger sur le DG stocké
    hits = 0
    for eid in ids:
        stored = _stored(model, eid)
        cue = F.normalize(stored + 0.05 * torch.randn_like(stored), dim=0)
        result = learner.recall(cue_dg=cue, touch=False)
        if result.key == eid:
            hits += 1
    assert hits / len(ids) >= 0.9, f"ctx top1={hits}/{len(ids)}"
    unknown = 0
    for _ in range(40):
        cue = F.normalize(torch.randn(cfg.dg_dim), dim=0)
        result = learner.recall(cue_dg=cue, touch=False)
        if result.level == "inconnu" and not result.events:
            unknown += 1
    assert unknown / 40 >= 0.95, f"ctx abstention={unknown}/40"


def test_schemas_formation_idempotence_and_recall():
    cfg = _cfg(
        max_nodes=200,
        max_patterns=200,
        enable_dreaming=False,
        schema_every=1,
        schema_tau=0.8,
        schema_min_support=5,
        schema_min_segments=2,
        schema_recall_floor=0.6,
        min_event_len=2,
        seg_delta=0.05,
    )
    learner, model = _learner(cfg)
    dim = cfg.semantic_dim
    themes = ["alpha", "beta", "gamma"]
    mapping = {}
    order = []
    # 6 tours : chaque thème 5 événements → >= 6 segments via bascules
    for round_i in range(6):
        for t_i, theme in enumerate(themes):
            for j in range(5):
                eid = f"{theme}_{round_i}_{j}"
                vec = torch.zeros(dim)
                vec[t_i * 2] = 1.0
                vec[t_i * 2 + 1] = 0.05 * (j + 1)  # faible jitter, thèmes bien séparés
                mapping[eid] = F.normalize(vec, dim=0)
                order.append((eid, theme))
    for k in range(10):
        eid = f"intruder_{k}"
        mapping[eid] = F.normalize(torch.randn(dim), dim=0)
        order.append((eid, None))
    bind_text_table(model, mapping)
    for eid, _ in order:
        learner.train_step(texts=[eid], episode_id=[eid])

    # Segments forcés : le test cible les schémas, pas le segmenter
    seg_i = 0
    for round_i in range(6):
        for theme in themes:
            sid = f"seg_forced_{seg_i}"
            seg_i += 1
            for j in range(5):
                eid = f"{theme}_{round_i}_{j}"
                if eid in learner.graph.graph.nodes:
                    learner.graph.graph.nodes[eid]["data"]["segment_id"] = sid
                    learner.graph.graph.nodes[eid]["data"]["event_idx"] = j

    stats = DreamingPhase.run_sleep_cycle(
        learner.buffer,
        model,
        learner.graph,
        cfg,
        memory_manager=learner.memory_manager,
        episodes=learner.episodes,
        schema_builder=learner.schema_builder,
    )
    g = learner.graph.graph
    schemas = [
        nid for nid, node in g.nodes.items()
        if (node.get("data") or {}).get("type") == "schema"
    ]
    assert abs(len(schemas) - 3) <= 1, f"n_schemas={len(schemas)} created={stats.get('schemas_created')}"

    # Pureté
    pure_ok = 0
    for sid in schemas:
        members = (g.nodes[sid]["data"]).get("members") or []
        themes_m = []
        for mid in members:
            th = mid.split("_")[0]
            if th in themes:
                themes_m.append(th)
        if not themes_m:
            continue
        majority = max(set(themes_m), key=themes_m.count)
        purity = themes_m.count(majority) / len(themes_m)
        if purity >= 0.9:
            pure_ok += 1
    assert pure_ok >= max(1, len(schemas) - 1)

    for k in range(10):
        eid = f"intruder_{k}"
        if eid in g.nodes:
            assert not (g.nodes[eid]["data"] or {}).get("schema_of")

    supports = {sid: int(g.nodes[sid]["data"]["support"]) for sid in schemas}
    DreamingPhase.run_sleep_cycle(
        learner.buffer,
        model,
        learner.graph,
        cfg,
        memory_manager=learner.memory_manager,
        episodes=learner.episodes,
        schema_builder=learner.schema_builder,
    )
    schemas2 = [
        nid for nid, node in g.nodes.items()
        if (node.get("data") or {}).get("type") == "schema"
    ]
    assert len(schemas2) == len(schemas)
    for sid in schemas:
        assert int(g.nodes[sid]["data"]["support"]) == supports[sid]

    # Pression : schémas non évincés
    cfg2 = _cfg(
        max_nodes=40,
        max_patterns=40,
        enable_dreaming=True,
        consolidation_freq=10,
        dream_cycles=1,
        schema_every=1,
        schema_min_support=5,
        schema_min_segments=2,
    )
    learner2, model2 = _learner(cfg2)
    bind_text_table(model2, mapping)
    for eid, _ in order:
        learner2.train_step(texts=[eid], episode_id=[eid])
    DreamingPhase.run_sleep_cycle(
        learner2.buffer,
        model2,
        learner2.graph,
        cfg2,
        memory_manager=learner2.memory_manager,
        episodes=learner2.episodes,
        schema_builder=learner2.schema_builder,
    )
    # Remplir encore
    gen = torch.Generator().manual_seed(9)
    for i in range(80):
        eid = f"fill_{i}"
        mapping[eid] = F.normalize(torch.randn(dim, generator=gen), dim=0)
    bind_text_table(model2, mapping)
    for i in range(80):
        learner2.train_step(texts=[f"fill_{i}"], episode_id=[f"fill_{i}"])
    g2 = learner2.graph.graph
    schemas_alive = [
        nid for nid, node in g2.nodes.items()
        if (node.get("data") or {}).get("type") == "schema"
    ]
    assert len(schemas_alive) >= 1

    # recall_general : cue = centroïde du thème (texte prototype)
    hits = 0
    for theme in themes:
        cue = f"{theme}_0_0"
        results = learner.recall_general(cue_texts=[cue], k=1)
        if not results:
            continue
        sid = results[0].schema_id
        proto = (g.nodes[sid]["data"]).get("prototype_id", "")
        members = results[0].members_sample or []
        if str(proto).startswith(theme) or any(str(m).startswith(theme) for m in members):
            hits += 1
        else:
            # majority theme among full members
            full = (g.nodes[sid]["data"]).get("members") or []
            labels = [m.split("_")[0] for m in full if m.split("_")[0] in themes]
            if labels and max(set(labels), key=labels.count) == theme:
                hits += 1
    assert hits / len(themes) >= 0.9, f"recall_general hits={hits}"
    # vecteur orthogonal aux thèmes
    mapping["zz_unknown_vec"] = F.normalize(torch.tensor([0.0] * (dim - 1) + [1.0]), dim=0)
    bind_text_table(model, mapping)
    unk = learner.recall_general(cue_texts=["zz_unknown_vec"], k=1)
    assert unk == []


def test_redundancy_bias_under_pressure():
    cfg = _cfg(
        max_nodes=80,
        max_patterns=80,
        enable_dreaming=False,
        schema_every=1,
        schema_min_support=5,
        schema_min_segments=2,
        forget_w_red=2.0,
        protect_frac=0.1,
    )
    learner, model = _learner(cfg)
    dim = cfg.semantic_dim
    mapping = {}
    order = []
    for round_i in range(6):
        for j in range(6):
            eid = f"theme_{round_i}_{j}"
            vec = torch.zeros(dim)
            vec[0] = 1.0
            vec[2 + j] = 0.12
            mapping[eid] = F.normalize(vec, dim=0)
            order.append(eid)
    gen = torch.Generator().manual_seed(0)
    for i in range(20):
        eid = f"noise_{i}"
        mapping[eid] = F.normalize(torch.randn(dim, generator=gen), dim=0)
        order.append(eid)
    bind_text_table(model, mapping)
    for eid in order:
        learner.train_step(texts=[eid], episode_id=[eid])
    # Segments forcés pour matérialiser le schéma thématique
    for round_i in range(6):
        sid = f"seg_red_{round_i}"
        for j in range(6):
            eid = f"theme_{round_i}_{j}"
            if eid in learner.graph.graph.nodes:
                learner.graph.graph.nodes[eid]["data"]["segment_id"] = sid
                learner.graph.graph.nodes[eid]["data"]["event_idx"] = j
    DreamingPhase.run_sleep_cycle(
        learner.buffer,
        model,
        learner.graph,
        cfg,
        memory_manager=learner.memory_manager,
        episodes=learner.episodes,
        schema_builder=learner.schema_builder,
    )
    g = learner.graph.graph
    low_sur_members = set()
    for nid, node in g.nodes.items():
        data = node.get("data") or {}
        if data.get("schema_of"):
            data["surprise"] = 0.05
            data["schema_cos"] = max(float(data.get("schema_cos", 0.0)), 0.9)
            low_sur_members.add(nid)
    assert low_sur_members, "aucun membre de schéma pour le test de redondance"
    pop = [
        nid for nid, node in g.nodes.items()
        if (node.get("data") or {}).get("type") != "schema"
    ]
    pop_frac = len(low_sur_members & set(pop)) / max(len(pop), 1)
    # Éviction contrôlée (pas de protection artificiellement longue)
    n_evict = max(5, len(low_sur_members) // 2)
    evicted = set(learner.memory_manager.evict(g, model.ca3, learner.episodes, n_evict))
    assert evicted, "aucune éviction"
    ev_frac = len(low_sur_members & evicted) / max(len(evicted), 1)
    assert ev_frac > pop_frac, f"evicted low-sur schema frac={ev_frac:.3f} pop={pop_frac:.3f}"


def test_episodic_mode_false_legacy():
    cfg = _cfg(episodic_mode=False, enable_dreaming=False)
    learner, model = _learner(cfg)
    assert learner.memory_manager is None
    assert learner.schema_builder is None
    assert learner.graph.graph.eviction_policy is None
    assert model.ca3._eviction_handler is None
    bind_text_table(model, {"a": F.normalize(torch.ones(cfg.semantic_dim), dim=0)})
    learner.train_step(texts=["a"], episode_id=["a"])
    assert "a" in learner.graph.graph.nodes
    assert learner.graph.graph.clock == 0  # pas de tick hors episodic
    result = learner.recall(cue_texts=["a"])
    assert result.events == []
