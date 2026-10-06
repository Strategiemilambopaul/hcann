import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.episodic_bench import (
    LexicalStore,
    MergeStore,
    committed,
    ingest,
    perturb_dg,
    rank_ca3_patterns,
    score_completion,
    score_hits,
    summarize_system,
)
from integrations.episodic_scenes import crop_of, distractors, distinct_visits, twin_pairs


def test_score_hits_unique_winner():
    hits = [{"id": "twin", "score": 0.9, "path": "semantic"}]
    sc = score_hits(hits, "target", "twin")
    assert sc["committed"] is True
    assert sc["confused"] is True
    assert sc["hit"] is False


def test_score_hits_ambiguous_pair_counts_as_hit_not_confusion():
    hits = [
        {"id": "twin", "score": 0.5, "path": "ambiguous_twins", "ambiguous": True},
        {"id": "target", "score": 0.5, "path": "ambiguous_twins", "ambiguous": True},
    ]
    sc = score_hits(hits, "target", "twin")
    assert sc["committed"] is False
    assert sc["confused"] is False
    assert sc["hit"] is True
    assert sc["pair_at_2"] is True


def test_committed_false_on_ambiguous_path():
    assert committed([{"id": "a", "path": "ambiguous_twins"}]) is False
    assert committed([{"id": "a", "path": "rag_clip"}]) is True


def test_summarize_system_rates():
    rows = [
        {"query": "q", "confused": True, "hit": False, "pair_at_2": True, "committed": True},
        {"query": "q", "confused": False, "hit": True, "pair_at_2": True, "committed": False},
    ]
    s = summarize_system(rows)
    assert s["n_queries"] == 2
    assert s["confusion_at_1"] == 0.5
    assert s["hit"] == 0.5
    assert s["n_uncommitted"] == 1


def test_ingest_keeps_caption_and_stores_trace():
    store = LexicalStore()
    ingest(store, twin_pairs(), distractors())
    momo = next(ep for ep in store._eps if ep["id"] == "vet_momo")
    assert momo["text"].endswith("This is Momo.")
    assert "bring Momo" not in momo["text"]
    trace = momo["trace"]
    assert trace["trace_id"] == "vet_momo"
    assert trace["entity_id"] == "paul"
    assert trace["place"] == "vet clinic"
    assert trace["participants"] == ["paul", "Momo"]
    assert trace["intent"] == "bring Momo to the vet"
    assert trace["outcome"] == "Momo checked"
    assert trace["timestamp"] == "2026-04-02T09:30:00"
    hits = store.recall(
        query_text="Cat at the vet clinic sitting on the table. This is Momo.", k=1
    )
    assert hits[0]["id"] == "vet_momo"
    assert hits[0]["trace"]["outcome"] == "Momo checked"


def test_lexical_identity_picks_named_twin():
    store = LexicalStore()
    ingest(store, twin_pairs(), distractors())
    hits = store.recall(
        query_text="Cat at the vet clinic sitting on the table. This is Momo.", k=5
    )
    assert hits[0]["id"] == "vet_momo"
    assert hits[0]["path"] == "lexical"


def test_lexical_shared_caption_commits_the_more_recent_twin():
    store = LexicalStore()
    ingest(store, twin_pairs(), distractors())
    hits = store.recall(query_text="Cat at the vet clinic sitting on the table.", k=5)
    assert hits[0]["id"] == "vet_luna"
    assert {hits[0]["id"], hits[1]["id"]} == {"vet_momo", "vet_luna"}
    sc = score_hits(hits, "vet_momo", "vet_luna")
    assert sc["committed"] is True
    assert sc["confused"] is True


def test_lexical_empty_query_falls_back_to_recency():
    store = LexicalStore()
    extras = distractors()
    ingest(store, twin_pairs(), extras)
    hits = store.recall(query_text=None, image=None, k=3)
    assert hits
    assert hits[0]["id"] == extras[-1]["id"]


def test_merge_store_collapses_each_twin_pair():
    store = MergeStore()
    ingest(store, twin_pairs(), distractors())
    assert len(store._eps) == 8
    assert "vet_luna" not in {e["id"] for e in store._eps}
    assert "vet_momo" in {e["id"] for e in store._eps}
    hits = store.recall(
        query_text="Cat at the vet clinic sitting on the table. This is Luna.", k=5
    )
    assert hits[0]["id"] == "vet_momo"
    sc = score_hits(hits, "vet_luna", "vet_momo")
    assert sc["committed"] is True
    assert sc["confused"] is True


def test_score_completion_hit_and_miss():
    hit = score_completion([{"id": "a", "score": 0.9, "path": "ca3_complete"}], "a")
    assert hit["hit"] is True
    assert hit["confused"] is False
    miss = score_completion([{"id": "b", "score": 0.9, "path": "ca3_complete"}], "a")
    assert miss["hit"] is False
    assert miss["confused"] is True
    assert miss["pair_at_2"] is False


def test_distinct_visits_are_unique_and_crop_keeps_size():
    visits = distinct_visits()
    ids = [v["id"] for v in visits]
    assert len(ids) == len(set(ids)) == 4
    cropped = crop_of(visits[0]["image"])
    assert cropped.size == (224, 224)


def test_perturb_dg_mask_and_noise_change_the_vector():
    import numpy as np

    v = np.zeros(32, dtype=np.float32)
    v[0] = 1.0
    noisy = perturb_dg(v, "noisy_dg", 0)
    masked = perturb_dg(v, "masked_dg", 0)
    assert noisy.shape == (32,)
    assert not np.allclose(noisy, v)
    assert float(np.linalg.norm(noisy) - 1.0) < 1e-5
    assert masked.shape == (32,)
    assert int((masked == 0).sum()) >= 1


def test_ca3_complete_recovers_noisy_stored_pattern_better_than_chance():
    import torch

    from models.ca3_hopfield import CA3ModernHopfield
    from utils.config import HCANNConfig

    cfg = HCANNConfig()
    cfg.dg_dim = 64
    cfg.max_patterns = 20
    ca3 = CA3ModernHopfield(cfg)
    a = torch.zeros(cfg.dg_dim)
    a[0] = 1.0
    b = torch.zeros(cfg.dg_dim)
    b[1] = 1.0
    ca3.store(a, key="momo")
    ca3.store(b, key="luna")
    cue = perturb_dg(a.numpy(), "noisy_dg", 0)
    hits = rank_ca3_patterns(ca3, cue, k=2, complete=True)
    knn = rank_ca3_patterns(ca3, cue, k=2, complete=False)
    assert hits[0]["id"] == "momo"
    assert knn[0]["id"] == "momo"


def test_mixed_dg_cue_follows_the_majority_attractor():
    import torch

    from models.ca3_hopfield import CA3ModernHopfield
    from utils.config import HCANNConfig

    cfg = HCANNConfig()
    cfg.dg_dim = 64
    cfg.max_patterns = 20
    ca3 = CA3ModernHopfield(cfg)
    a = torch.zeros(cfg.dg_dim)
    a[0] = 1.0
    b = torch.zeros(cfg.dg_dim)
    b[1] = 1.0
    ca3.store(a, key="momo")
    ca3.store(b, key="luna")
    cue = (0.6 * a + 0.4 * b).numpy()
    hits = rank_ca3_patterns(ca3, cue, k=2, complete=True)
    knn = rank_ca3_patterns(ca3, cue, k=2, complete=False)
    assert hits[0]["id"] == "momo"
    assert knn[0]["id"] == "momo"
