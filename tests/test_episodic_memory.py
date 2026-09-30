import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.episodic_memory import EpisodicMemory, apply_dg_tiebreak
from memory.consolidation import HebbianMemoryGraph
from models.ca3_hopfield import CA3ModernHopfield
from models.hcann import HCANN
from train.continual_learner import ContinualLearner
from utils.config import HCANNConfig


def _small_memory(tie_margin: float = 0.05) -> EpisodicMemory:
    cfg = HCANNConfig()
    cfg.use_mock_encoder = True
    cfg.enable_dreaming = False
    cfg.ec_dim = 128
    cfg.dg_dim = 64
    cfg.hpc_size = 64
    cfg.semantic_dim = 32
    cfg.wm_dim = 32
    cfg.max_patterns = 50
    cfg.max_nodes = 50
    device = torch.device("cpu")
    model = HCANN(cfg).to(device)
    learner = ContinualLearner(model, cfg, device)
    return EpisodicMemory(model, learner, device, tie_margin=tie_margin)


def test_encode_recall_deja_vu_smoke():
    mem = _small_memory()
    mem.encode("ep_a", "Cat named Momo at the clinic.", created_at="2026-04-02")
    mem.encode("ep_b", "Cat named Luna at the clinic.", created_at="2026-04-09")
    hits = mem.recall(query_text="Cat at the clinic", k=2)
    assert len(hits) >= 1
    assert hits[0]["id"] in {"ep_a", "ep_b"}
    assert hits[0]["path"] in {"semantic", "dg_tiebreak", "identity", "ambiguous_twins", "ca3"}
    probe = mem.deja_vu(text="Cat named Momo at the clinic.")
    assert "novelty" in probe
    assert "deja_vu" in probe
    assert "is_novel" in probe
    assert "nearest_id" in probe
    assert "nearest_cosine" in probe
    rag = mem.rag_recall(query_text="Cat at the clinic", k=2)
    assert len(rag) >= 1
    assert rag[0]["path"] == "rag_clip"

    from PIL import Image

    img = Image.new("RGB", (224, 224), (8, 8, 8))
    with_img = mem.recall(query_text="Cat at the clinic", image=img, k=2, tiebreak="auto")
    assert with_img[0]["path"] in {"semantic", "identity", "ambiguous_twins", "ca3"}


def test_apply_dg_tiebreak_swaps_when_margin_low():
    cfg = HCANNConfig()
    cfg.semantic_dim = 8
    cfg.dg_dim = 8
    cfg.hpc_size = 8
    g = HebbianMemoryGraph(cfg, torch.device("cpu"))

    sem_a = np.zeros(8, dtype=np.float32)
    sem_a[0] = 1.0
    sem_b = np.zeros(8, dtype=np.float32)
    sem_b[0] = 0.98
    sem_b[1] = 0.2
    dg_a = np.zeros(8, dtype=np.float32)
    dg_a[0] = 1.0
    dg_b = np.zeros(8, dtype=np.float32)
    dg_b[1] = 1.0

    g.add_node("a", {"text": "A"}, sem_a, sem_embedding=sem_a, dg_embedding=dg_a)
    g.add_node("b", {"text": "B"}, sem_b, sem_embedding=sem_b, dg_embedding=dg_b)

    q_sem = np.zeros(8, dtype=np.float32)
    q_sem[0] = 0.99
    q_sem[1] = 0.15
    hits = g.retrieve_semantic(q_sem, k=2)
    assert {h["id"] for h in hits} == {"a", "b"}

    kept, path_kept = apply_dg_tiebreak(hits, dg_a, g, margin=0.0)
    assert path_kept == "semantic"
    assert kept[0]["id"] == hits[0]["id"]

    swapped, _path = apply_dg_tiebreak(hits, dg_a, g, margin=1.0, dg_mix=1.0, twin_min_sim=0.5)
    assert swapped[0]["id"] == "a"

    ident, ipath = apply_dg_tiebreak(
        hits, dg_a, g, margin=1.0, query_text="remember Momo specifically"
    )
    assert ipath == "ambiguous_twins"
    assert ident[0]["id"] == hits[0]["id"]


def test_identity_cue_picks_matching_twin():
    cfg = HCANNConfig()
    cfg.semantic_dim = 8
    cfg.dg_dim = 8
    cfg.hpc_size = 8
    g = HebbianMemoryGraph(cfg, torch.device("cpu"))
    sem_luna = np.zeros(8, dtype=np.float32)
    sem_luna[0] = 0.98
    sem_luna[1] = 0.2
    sem_momo = np.zeros(8, dtype=np.float32)
    sem_momo[0] = 1.0
    dg_momo = np.zeros(8, dtype=np.float32)
    dg_momo[0] = 1.0
    dg_luna = np.zeros(8, dtype=np.float32)
    dg_luna[1] = 1.0
    g.add_node(
        "momo",
        {"text": "Cat named Momo at the clinic"},
        sem_momo,
        sem_embedding=sem_momo,
        dg_embedding=dg_momo,
    )
    g.add_node(
        "luna",
        {"text": "Cat named Luna at the clinic"},
        sem_luna,
        sem_embedding=sem_luna,
        dg_embedding=dg_luna,
    )
    q_sem = np.zeros(8, dtype=np.float32)
    q_sem[0] = 0.99
    q_sem[1] = 0.15
    hits = g.retrieve_semantic(q_sem, k=2)
    assert hits[0]["id"] == "luna"

    tied, tied_path = apply_dg_tiebreak(
        hits, dg_momo, g, margin=1.0, query_text="Cat at the clinic"
    )
    assert tied_path == "ambiguous_twins"
    assert {h["id"] for h in tied[:2]} == {"momo", "luna"}
    assert tied[0]["id"] == "luna"

    ident, ipath = apply_dg_tiebreak(
        hits, dg_momo, g, margin=1.0, query_text="Cat named Momo at the clinic"
    )
    assert ipath == "identity"
    assert ident[0]["id"] == "momo"


def test_generic_cue_pulls_pair_partner_and_stays_ambiguous():
    cfg = HCANNConfig()
    cfg.semantic_dim = 8
    cfg.dg_dim = 8
    cfg.hpc_size = 8
    g = HebbianMemoryGraph(cfg, torch.device("cpu"))
    sem_luna = np.zeros(8, dtype=np.float32)
    sem_luna[0] = 1.0
    sem_office = np.zeros(8, dtype=np.float32)
    sem_office[0] = 0.99
    sem_office[1] = 0.12
    sem_momo = np.zeros(8, dtype=np.float32)
    sem_momo[0] = 0.97
    sem_momo[1] = 0.22
    dg = np.zeros(8, dtype=np.float32)
    g.add_node(
        "luna",
        {"text": "Cat named Luna at the clinic"},
        sem_luna,
        sem_embedding=sem_luna,
        dg_embedding=dg,
    )
    g.add_node(
        "office",
        {"text": "Blue circle in the office"},
        sem_office,
        sem_embedding=sem_office,
        dg_embedding=dg,
    )
    g.add_node(
        "momo",
        {"text": "Cat named Momo at the clinic"},
        sem_momo,
        sem_embedding=sem_momo,
        dg_embedding=dg,
    )
    q_sem = np.zeros(8, dtype=np.float32)
    q_sem[0] = 0.995
    q_sem[1] = 0.05
    hits = g.retrieve_semantic(q_sem, k=2)
    assert hits[0]["id"] == "luna"
    assert hits[1]["id"] == "office"
    out, path = apply_dg_tiebreak(
        hits,
        dg,
        g,
        margin=1.0,
        query_text="Cat at the clinic",
        query_sem=q_sem,
    )
    assert path == "ambiguous_twins"
    assert {h["id"] for h in out[:2]} == {"luna", "momo"}


def test_ca3_completion_breaks_generic_tie():
    cfg = HCANNConfig()
    cfg.semantic_dim = 8
    cfg.dg_dim = 8
    cfg.hpc_size = 8
    cfg.hopfield_beta = 20.0
    cfg.hopfield_steps = 5
    g = HebbianMemoryGraph(cfg, torch.device("cpu"))
    sem_luna = np.zeros(8, dtype=np.float32)
    sem_luna[0] = 0.98
    sem_luna[1] = 0.2
    sem_momo = np.zeros(8, dtype=np.float32)
    sem_momo[0] = 1.0
    dg = np.zeros(8, dtype=np.float32)
    g.add_node(
        "momo",
        {"text": "Cat named Momo at the clinic"},
        sem_momo,
        sem_embedding=sem_momo,
        dg_embedding=dg,
    )
    g.add_node(
        "luna",
        {"text": "Cat named Luna at the clinic"},
        sem_luna,
        sem_embedding=sem_luna,
        dg_embedding=dg,
    )
    ca3 = CA3ModernHopfield(cfg)
    pat_momo = torch.zeros(8)
    pat_momo[0] = 1.0
    pat_luna = torch.zeros(8)
    pat_luna[1] = 1.0
    ca3.store(pat_momo, key="momo")
    ca3.store(pat_luna, key="luna")
    q_sem = np.zeros(8, dtype=np.float32)
    q_sem[0] = 0.99
    q_sem[1] = 0.15
    hits = g.retrieve_semantic(q_sem, k=2)
    assert hits[0]["id"] == "luna"
    q_dg = np.zeros(8, dtype=np.float32)
    q_dg[0] = 1.0
    q_dg[1] = 0.1
    out, path = apply_dg_tiebreak(
        hits,
        q_dg,
        g,
        margin=0.05,
        query_text="Cat at the clinic",
        query_sem=q_sem,
        ca3=ca3,
    )
    assert path == "ca3"
    assert out[0]["id"] == "momo"


def test_visual_only_twins_stay_ambiguous():
    cfg = HCANNConfig()
    cfg.semantic_dim = 8
    cfg.dg_dim = 8
    cfg.hpc_size = 8
    cfg.hopfield_beta = 20.0
    g = HebbianMemoryGraph(cfg, torch.device("cpu"))
    sem_luna = np.zeros(8, dtype=np.float32)
    sem_luna[0] = 0.98
    sem_luna[1] = 0.2
    sem_momo = np.zeros(8, dtype=np.float32)
    sem_momo[0] = 1.0
    dg = np.zeros(8, dtype=np.float32)
    g.add_node(
        "momo",
        {"text": "Cat named Momo at the clinic"},
        sem_momo,
        sem_embedding=sem_momo,
        dg_embedding=dg,
    )
    g.add_node(
        "luna",
        {"text": "Cat named Luna at the clinic"},
        sem_luna,
        sem_embedding=sem_luna,
        dg_embedding=dg,
    )
    ca3 = CA3ModernHopfield(cfg)
    pat_momo = torch.zeros(8)
    pat_momo[0] = 1.0
    pat_luna = torch.zeros(8)
    pat_luna[1] = 1.0
    ca3.store(pat_momo, key="momo")
    ca3.store(pat_luna, key="luna")
    q_sem = np.zeros(8, dtype=np.float32)
    q_sem[0] = 0.99
    q_sem[1] = 0.15
    hits = g.retrieve_semantic(q_sem, k=2)
    q_dg = np.zeros(8, dtype=np.float32)
    q_dg[0] = 1.0
    q_dg[1] = 0.1
    out, path = apply_dg_tiebreak(
        hits,
        q_dg,
        g,
        margin=0.05,
        query_sem=q_sem,
        ca3=ca3,
        visual_only=True,
    )
    assert path == "ambiguous_twins"
    assert out[0]["id"] == "luna"


def test_dg_skips_swap_when_top2_not_twins():
    cfg = HCANNConfig()
    cfg.semantic_dim = 8
    cfg.dg_dim = 8
    cfg.hpc_size = 8
    g = HebbianMemoryGraph(cfg, torch.device("cpu"))
    sem_a = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    sem_c = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    dg_a = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    dg_c = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    g.add_node("a", {"text": "A"}, sem_a, sem_embedding=sem_a, dg_embedding=dg_a)
    g.add_node("c", {"text": "C"}, sem_c, sem_embedding=sem_c, dg_embedding=dg_c)
    q = np.array([0.8, 0, 0.7, 0, 0, 0, 0, 0], dtype=np.float32)
    hits = g.retrieve_semantic(q, k=2)
    kept, path = apply_dg_tiebreak(hits, dg_c, g, margin=1.0, dg_mix=1.0, twin_min_sim=0.85)
    assert path == "semantic"
    assert kept[0]["id"] == hits[0]["id"]


def test_labeled_pair_id_blocks_cross_pair_identity():
    cfg = HCANNConfig()
    cfg.semantic_dim = 8
    cfg.dg_dim = 8
    cfg.hpc_size = 8
    g = HebbianMemoryGraph(cfg, torch.device("cpu"))
    sem = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    sem_b = np.array([0.99, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    dg = np.zeros(8, dtype=np.float32)
    g.add_node(
        "shop_a",
        {"text": "Visit to the white shop. Keys.", "pair_id": "shop_return"},
        sem,
        sem_embedding=sem,
        dg_embedding=dg,
    )
    g.add_node(
        "visit_b",
        {"text": "Visit to the white shop. Package.", "pair_id": "white_shop"},
        sem_b,
        sem_embedding=sem_b,
        dg_embedding=dg,
    )
    q = np.array([0.995, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    hits = g.retrieve_semantic(q, k=2)
    kept, path = apply_dg_tiebreak(
        hits, dg, g, margin=1.0, twin_min_sim=0.5, query_text="Visit keys"
    )
    assert path == "semantic"
    assert kept[0]["id"] == hits[0]["id"]


def test_close_clip_without_caption_overlap_is_not_a_twin():
    cfg = HCANNConfig()
    cfg.semantic_dim = 8
    cfg.dg_dim = 8
    cfg.hpc_size = 8
    g = HebbianMemoryGraph(cfg, torch.device("cpu"))
    sem = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    dg = np.zeros(8, dtype=np.float32)
    g.add_node(
        "visit",
        {
            "text": (
                "Visit to the white shop. Left an item at the counter "
                "for Mme X. Package on Tuesday."
            )
        },
        sem,
        sem_embedding=sem,
        dg_embedding=dg,
    )
    g.add_node(
        "retour",
        {
            "text": (
                "Back at the white shop. Picked something up from the counter. The keys."
            )
        },
        sem,
        sem_embedding=sem,
        dg_embedding=dg,
    )
    q = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    hits = g.retrieve_semantic(q, k=2)
    kept, path = apply_dg_tiebreak(
        hits, dg, g, margin=1.0, twin_min_sim=0.5, query_text="the white shop"
    )
    assert path == "semantic"
    assert kept[0]["id"] == hits[0]["id"]


def test_dg_tiebreak_does_not_promote_far_third():
    cfg = HCANNConfig()
    cfg.semantic_dim = 8
    cfg.dg_dim = 8
    cfg.hpc_size = 8
    g = HebbianMemoryGraph(cfg, torch.device("cpu"))
    sem_a = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    sem_b = np.array([0.98, 0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    sem_c = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    dg_a = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    dg_b = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    dg_c = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    g.add_node("a", {"text": "A"}, sem_a, sem_embedding=sem_a, dg_embedding=dg_a)
    g.add_node("b", {"text": "B"}, sem_b, sem_embedding=sem_b, dg_embedding=dg_b)
    g.add_node("c", {"text": "C"}, sem_c, sem_embedding=sem_c, dg_embedding=dg_c)
    q_sem = np.array([0.99, 0.15, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    hits = g.retrieve_semantic(q_sem, k=3)
    reranked, _path = apply_dg_tiebreak(hits, dg_c, g, margin=0.15, dg_mix=1.0)
    assert reranked[-1]["id"] == "c"


def test_recall_from_codes_uses_dg_tiebreak():
    mem = _small_memory(tie_margin=1.0)
    mem.dg_mix = 1.0
    g = mem.graph
    dim_s = mem.model.config.semantic_dim
    dim_d = mem.model.config.dg_dim
    dim_h = mem.model.config.hpc_size

    sem_a = np.zeros(dim_s, dtype=np.float32)
    sem_a[0] = 1.0
    sem_b = np.zeros(dim_s, dtype=np.float32)
    sem_b[0] = 0.97
    sem_b[1] = 0.24
    dg_a = np.zeros(dim_d, dtype=np.float32)
    dg_a[0] = 1.0
    dg_b = np.zeros(dim_d, dtype=np.float32)
    dg_b[1] = 1.0
    hpc_a = np.zeros(dim_h, dtype=np.float32)
    hpc_a[0] = 1.0
    hpc_b = np.zeros(dim_h, dtype=np.float32)
    hpc_b[1] = 1.0

    g.add_node("twin_a", {"text": "A", "created_at": "t0"}, hpc_a, sem_embedding=sem_a, dg_embedding=dg_a)
    g.add_node("twin_b", {"text": "B", "created_at": "t1"}, hpc_b, sem_embedding=sem_b, dg_embedding=dg_b)

    q_sem = np.zeros(dim_s, dtype=np.float32)
    q_sem[0] = 0.98
    q_sem[1] = 0.2
    hits = mem.recall_from_codes(q_sem, dg_a, k=2)
    assert hits[0]["id"] == "twin_a"
    assert hits[0]["path"] in {"semantic", "dg_tiebreak"}
    assert hits[0]["text"] == "A"
