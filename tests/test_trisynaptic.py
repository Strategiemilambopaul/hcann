import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import HCANNConfig
from models.entorhinal import EntorhinalCortex
from models.dentate_gyrus import DentateGyrus
from models.ca3_hopfield import CA3ModernHopfield
from models.ca1 import CA1Comparator
from models.subiculum import SubiculumGateway
from models.hcann import HCANN
from memory.replay_buffer import EpisodicBuffer
from memory.dreaming import DreamingPhase


@pytest.fixture
def config():
    cfg = HCANNConfig()
    cfg.use_mock_encoder = True
    cfg.ec_dim = 128
    cfg.dg_dim = 64
    cfg.semantic_dim = 32
    cfg.wm_dim = 32
    cfg.ec_sparsity = 0.05
    cfg.dg_sparsity = 0.1
    cfg.max_patterns = 100
    return cfg


def test_ec_sdr_sparsity(config):
    ec = EntorhinalCortex(config)
    sem = torch.randn(4, config.semantic_dim)
    sdr = ec(sem)
    assert sdr.shape == (4, config.ec_dim)
    assert torch.all((sdr == 0) | (sdr == 1))
    expected_k = max(1, int(config.ec_dim * config.ec_sparsity))
    assert int(sdr[0].sum()) == expected_k


def test_dg_separates_correlated_inputs(config):
    dg = DentateGyrus(config)
    grid = torch.randn(2, len(config.grid_periods) * 4)
    sdr_a = torch.zeros(2, config.ec_dim)
    sdr_b = torch.zeros(2, config.ec_dim)
    sdr_a[:, :10] = 1.0
    sdr_b[:, :10] = 1.0
    sdr_b[1, 10:12] = 1.0
    code_a = dg(sdr_a, grid)
    code_b = dg(sdr_b, grid)
    sim = torch.nn.functional.cosine_similarity(code_a[0], code_b[0], dim=0)
    sim_diff = torch.nn.functional.cosine_similarity(code_a[0], code_b[1], dim=0)
    assert sim > sim_diff or sim_diff < 0.9


def test_ca3_store_and_complete(config):
    ca3 = CA3ModernHopfield(config)
    pattern = torch.randn(config.dg_dim)
    pattern = torch.nn.functional.normalize(pattern, dim=0)
    ca3.store(pattern)
    assert ca3.num_patterns == 1
    noisy = pattern + 0.3 * torch.randn(config.dg_dim)
    recovered = ca3.complete(noisy).squeeze(0)
    sim = torch.nn.functional.cosine_similarity(recovered, pattern, dim=0)
    assert sim.item() > 0.5


def test_ca3_complete_among_picks_closer_attractor(config):
    config.hopfield_beta = 20.0
    ca3 = CA3ModernHopfield(config)
    a = torch.zeros(config.dg_dim)
    a[0] = 1.0
    b = torch.zeros(config.dg_dim)
    b[1] = 1.0
    ca3.store(a, key="momo")
    ca3.store(b, key="luna")
    q = torch.zeros(config.dg_dim)
    q[0] = 1.0
    q[1] = 0.15
    winner, scores = ca3.complete_among(q, ["momo", "luna"], margin=0.05)
    assert winner == "momo"
    assert scores["momo"] > scores["luna"]
    tied = 0.5 * a + 0.5 * b
    none, _ = ca3.complete_among(tied, ["momo", "luna"], margin=0.2)
    assert none is None


def test_ca3_completes_noisy_stored_pattern(config):
    """Amorce = motif stocké + bruit : la dynamique doit revenir à la clé."""
    config.hopfield_beta = 20.0
    ca3 = CA3ModernHopfield(config)
    a = torch.zeros(config.dg_dim)
    a[0] = 1.0
    b = torch.zeros(config.dg_dim)
    b[1] = 1.0
    ca3.store(a, key="momo")
    ca3.store(b, key="luna")
    torch.manual_seed(0)
    noisy = a + 0.45 * torch.randn(config.dg_dim)
    winner, scores = ca3.complete_among(noisy, ["momo", "luna"], margin=0.2)
    assert winner == "momo"
    assert scores["momo"] > scores["luna"]


def test_ca1_novelty_detection(config):
    ca1 = CA1Comparator(config)
    ca3_state = torch.randn(1, config.dg_dim)
    sem_known = ca1.decode(ca3_state)
    _, novelty_known, is_novel_known = ca1(ca3_state, sem_known)
    sem_unknown = torch.randn(1, config.semantic_dim)
    _, novelty_unknown, is_novel_unknown = ca1(ca3_state, sem_unknown)
    assert novelty_unknown.mean() >= novelty_known.mean()


def test_dreaming_hebb_without_backprop(config):
    model = HCANN(config)
    buffer = EpisodicBuffer(capacity=50)
    for i in range(20):
        sem = torch.randn(config.semantic_dim)
        hpc = torch.randn(config.dg_dim)
        buffer.push(sem, hpc, metadata={"episode_id": f"ep_{i}"})

    w_before = model.subiculum.W.clone()
    stats = DreamingPhase.run_sleep_cycle(
        buffer, model, _MockGraph(), config, n_cycles=2
    )
    assert stats["patterns_replayed"] > 0
    assert not torch.allclose(w_before, model.subiculum.W)
    assert stats["hebb_delta_norm"] >= 0


class _MockGraph:
    class _Inner:
        nodes = {}

        def update_hebbian_weights(self, a, b, w):
            pass

        def detect_hubs(self):
            return []

        def adaptive_forgetting(self):
            pass

        def apply_hebbian_decay(self):
            pass

    def __init__(self):
        self.graph = self._Inner()

    def consolidate(self, llm_callback=None):
        return []


def test_hcann_fast_encode_returns_trisynaptic_outputs(config):
    model = HCANN(config)
    texts = ["test episode"]
    ca3, sem, wm, novelty, dg_code, extras = model.fast_encode(texts=texts)
    assert ca3.shape == (1, config.dg_dim)
    assert sem.shape == (1, config.semantic_dim)
    assert wm.shape == (1, config.wm_dim)
    assert novelty.shape == (1,)
    assert dg_code.shape == (1, config.dg_dim)
    assert "is_novel" in extras
