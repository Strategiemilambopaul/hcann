import numpy as np
import torch

from memory.consolidation import HebbianMemoryGraph
from utils.config import HCANNConfig


def test_semantic_boosted_can_differ_from_semantic():
    cfg = HCANNConfig()
    cfg.spreading_strength = 0.5
    g = HebbianMemoryGraph(cfg, torch.device("cpu"))
    dim = cfg.semantic_dim

    for i, eid in enumerate(["a", "b", "c", "d"]):
        emb = np.zeros(dim, dtype=np.float32)
        emb[i] = 1.0
        g.add_node(eid, {"text": eid}, emb, sem_embedding=emb, dg_embedding=emb)

    g.update_hebbian_weights("a", "d", 0.8)

    q = np.zeros(dim, dtype=np.float32)
    q[0] = 1.0
    sem_ids = [n["id"] for n in g.retrieve_semantic(q, k=3)]
    boosted_ids = [n["id"] for n in g.retrieve_semantic_boosted(q, k=3)]
    assert sem_ids[0] == "a"
    assert boosted_ids != sem_ids or len(g.node_ids) <= 3
