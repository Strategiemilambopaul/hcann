"""Façade mémoire épisodique : journal d'épisodes vécus.

Unité = image + texte + temps. Geste distinctif = ne pas fusionner deux
visites trop proches. Jumeaux inférés (CLIP + légendes quasi doubles).
Identité dans la requête → path identity ; sinon CA3 seulement
si la requête a du texte et que les attracteurs sont séparés ;
image seule → path ambiguous_twins. Voir document/HCANN_concept.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.hcann import HCANN
from train.baselines.rag_baseline import RagBaseline
from train.continual_learner import ContinualLearner
from utils.config import HCANNConfig

DEFAULT_TIE_MARGIN = 0.05
DEFAULT_DG_MIX = 0.4
DEFAULT_TWIN_MIN_SIM = 0.85
DEFAULT_TWIN_MIN_JACCARD = 0.5
DEFAULT_CA3_MARGIN = 0.2

_IMAGE_TF = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def image_to_batch(image: Image.Image | torch.Tensor, device: torch.device) -> torch.Tensor:
    """PIL ou tenseur -> batch [1, 3, 224, 224] sur device."""
    if isinstance(image, torch.Tensor):
        t = image.detach()
        if t.dim() == 4:
            return t.to(device)
        if t.dim() == 3:
            if t.shape[0] not in (1, 3):
                t = t.permute(2, 0, 1)
            return t.unsqueeze(0).to(device)
        raise ValueError(f"Tenseur image de rang non supporté: {t.dim()}")
    t = _IMAGE_TF(image.convert("RGB"))
    return t.unsqueeze(0).to(device)


def _node_sem_vec(graph, episode_id: str) -> np.ndarray | None:
    idx = graph.id_to_index.get(str(episode_id))
    if idx is None:
        return None
    vec = graph.sem_embeddings[idx].detach().cpu().numpy().astype(np.float32).reshape(-1)
    n = float(np.linalg.norm(vec))
    return vec / max(n, 1e-8)


def _pair_sem_cosine(graph, id_a: str, id_b: str) -> float:
    va, vb = _node_sem_vec(graph, id_a), _node_sem_vec(graph, id_b)
    if va is None or vb is None:
        return 0.0
    return float(va @ vb)


def _tokens(s: str) -> set[str]:
    buf = "".join(c.lower() if c.isalnum() else " " for c in (s or ""))
    return {t for t in buf.split() if len(t) > 2}


def _caption(graph, episode_id: str) -> str:
    node = graph.nodes.get(str(episode_id)) or {}
    data = node.get("data") or node
    return str(data.get("text") or "")


def _text_jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _same_pair_id(graph, id_a: str, id_b: str) -> bool:
    na = graph.nodes.get(str(id_a)) or {}
    nb = graph.nodes.get(str(id_b)) or {}
    da = na.get("data") or na
    db = nb.get("data") or nb
    pa, pb = da.get("pair_id"), db.get("pair_id")
    return bool(pa) and pa == pb


def _are_twins(
    graph,
    id_a: str,
    id_b: str,
    min_sim: float,
    min_jaccard: float = DEFAULT_TWIN_MIN_JACCARD,
) -> bool:
    """Jumeaux : même pair_id si les deux sont étiquetés, sinon cosine CLIP + légendes quasi doubles."""
    na = graph.nodes.get(str(id_a)) or {}
    nb = graph.nodes.get(str(id_b)) or {}
    da = na.get("data") or na
    db = nb.get("data") or nb
    pa, pb = da.get("pair_id"), db.get("pair_id")
    if pa and pb:
        return pa == pb
    if pa or pb:
        return False
    if _pair_sem_cosine(graph, id_a, id_b) < min_sim:
        return False
    ta, tb = _tokens(_caption(graph, id_a)), _tokens(_caption(graph, id_b))
    if len(ta) < 2 or len(tb) < 2:
        return True
    return _text_jaccard(_caption(graph, id_a), _caption(graph, id_b)) >= min_jaccard


def _pair_partner(graph, episode_id: str, min_sim: float) -> str | None:
    eid = str(episode_id)
    node = graph.nodes.get(eid) or {}
    data = node.get("data") or node
    pair = data.get("pair_id")
    if pair:
        for other_id, other in graph.nodes.items():
            if str(other_id) == eid:
                continue
            od = other.get("data") or other
            if od.get("pair_id") == pair:
                return str(other_id)
        return None
    best_id, best_cos = None, -1.0
    for other_id in graph.nodes:
        if str(other_id) == eid:
            continue
        if not _are_twins(graph, eid, str(other_id), min_sim):
            continue
        cos = _pair_sem_cosine(graph, eid, str(other_id))
        if cos > best_cos:
            best_cos, best_id = cos, str(other_id)
    return best_id


def _clip_score(graph, episode_id: str, query_sem: np.ndarray | None) -> float | None:
    if query_sem is None:
        return None
    vec = _node_sem_vec(graph, episode_id)
    if vec is None:
        return None
    q = np.asarray(query_sem, dtype=np.float32).reshape(-1)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    return float(vec @ q)


def _as_hit(graph, episode_id: str, score: float) -> dict:
    node = graph.nodes.get(str(episode_id)) or {}
    data = node.get("data") or {}
    return {
        "id": str(episode_id),
        "score": float(score),
        "data": data,
        "text": data.get("text", ""),
    }


def _without(hits: list[dict], ids: set[str]) -> list[dict]:
    skip = {str(i) for i in ids}
    return [item for item in hits if str(item.get("id")) not in skip]


def _item_text(graph, item: dict) -> str:
    eid = str(item.get("id", ""))
    node = graph.nodes.get(eid) or {}
    data = node.get("data") or node
    parts = [data.get("text"), data.get("identity"), item.get("text")]
    return " ".join(str(p) for p in parts if p)


def _lex_overlap(query: str, stored: str) -> float:
    """Part des tokens de la requête (len>2) présents dans le souvenir."""
    q, s = _tokens(query), _tokens(stored)
    if not q:
        return 0.0
    return len(q & s) / len(q)


def _resolve_twin(semantic_hits: list[dict], graph, query_sem, twin_min_sim: float):
    first = semantic_hits[0]
    second = semantic_hits[1] if len(semantic_hits) > 1 else None
    partner_id = _pair_partner(graph, str(first.get("id")), twin_min_sim)
    if partner_id:
        for item in semantic_hits:
            if str(item.get("id")) == partner_id:
                return first, item
        score = _clip_score(graph, partner_id, query_sem)
        if score is None and second is not None and str(second.get("id")) == partner_id:
            score = float(second.get("score", 0.0))
        if score is None:
            return first, None
        return first, _as_hit(graph, partner_id, score)
    if second is not None and _are_twins(
        graph, str(first.get("id")), str(second.get("id")), twin_min_sim
    ):
        return first, second
    return first, None


def _ca3_decide(ca3, query_dg, id_a: str, id_b: str, margin: float):
    if ca3 is None:
        return None, {}
    decide = getattr(ca3, "complete_among", None)
    if decide is None:
        return None, {}
    q = torch.as_tensor(np.asarray(query_dg, dtype=np.float32).reshape(-1))
    return decide(q, [id_a, id_b], margin=margin)


def apply_dg_tiebreak(
    semantic_hits: list[dict],
    query_dg: np.ndarray,
    graph,
    margin: float = DEFAULT_TIE_MARGIN,
    dg_mix: float = DEFAULT_DG_MIX,
    twin_min_sim: float = DEFAULT_TWIN_MIN_SIM,
    query_text: str | None = None,
    query_sem: np.ndarray | None = None,
    ca3=None,
    ca3_margin: float = DEFAULT_CA3_MARGIN,
    visual_only: bool = False,
) -> tuple[list[dict], str]:
    """Jumeaux collés : identité, sinon CA3 si la requête a du texte, sinon paire ambiguë."""
    if len(semantic_hits) < 1:
        return semantic_hits, "semantic"
    first, twin = _resolve_twin(semantic_hits, graph, query_sem, twin_min_sim)
    if twin is None:
        return semantic_hits, "semantic"

    s0 = float(first.get("score", 0.0))
    st = float(twin.get("score", 0.0))
    if query_sem is not None:
        c0 = _clip_score(graph, str(first.get("id")), query_sem)
        ct = _clip_score(graph, str(twin.get("id")), query_sem)
        if c0 is not None:
            s0 = c0
            first = {**first, "score": s0}
        if ct is not None:
            st = ct
            twin = {**twin, "score": st}

    rest = [
        {**item, "semantic_score": float(item.get("score", 0.0))}
        for item in _without(semantic_hits, {str(first.get("id")), str(twin.get("id"))})
    ]
    close = (s0 - st) < margin

    if query_text:
        lex0 = _lex_overlap(query_text, _item_text(graph, first))
        lex1 = _lex_overlap(query_text, _item_text(graph, twin))
        tagged_first = {
            **first,
            "semantic_score": s0,
            "lex_overlap": lex0,
            "score": s0 + 2.0 * lex0,
            "twin_id": str(twin.get("id")),
        }
        tagged_twin = {
            **twin,
            "semantic_score": st,
            "lex_overlap": lex1,
            "score": st + 2.0 * lex1,
            "twin_id": str(first.get("id")),
        }
        if abs(lex1 - lex0) > 1e-9:
            if lex1 > lex0:
                return [tagged_twin, tagged_first, *rest], "identity"
            return [tagged_first, tagged_twin, *rest], "identity"
        if close:
            winner, ca3_scores = _ca3_decide(
                ca3, query_dg, str(first.get("id")), str(twin.get("id")), ca3_margin
            )
            tagged_first["ca3_score"] = ca3_scores.get(str(first.get("id")))
            tagged_twin["ca3_score"] = ca3_scores.get(str(twin.get("id")))
            if winner == str(twin.get("id")):
                return [tagged_twin, tagged_first, *rest], "ca3"
            if winner == str(first.get("id")):
                return [tagged_first, tagged_twin, *rest], "ca3"
            return [tagged_first, tagged_twin, *rest], "ambiguous_twins"
        return semantic_hits, "semantic"

    if not close:
        return semantic_hits, "semantic"

    if visual_only:
        tagged_first = {
            **first,
            "semantic_score": s0,
            "score": s0,
            "twin_id": str(twin.get("id")),
        }
        tagged_twin = {
            **twin,
            "semantic_score": st,
            "score": st,
            "twin_id": str(first.get("id")),
        }
        return [tagged_first, tagged_twin, *rest], "ambiguous_twins"

    winner, ca3_scores = _ca3_decide(
        ca3, query_dg, str(first.get("id")), str(twin.get("id")), ca3_margin
    )
    if winner is not None:
        tagged_first = {
            **first,
            "semantic_score": s0,
            "ca3_score": ca3_scores.get(str(first.get("id"))),
            "score": s0,
            "twin_id": str(twin.get("id")),
        }
        tagged_twin = {
            **twin,
            "semantic_score": st,
            "ca3_score": ca3_scores.get(str(twin.get("id"))),
            "score": st,
            "twin_id": str(first.get("id")),
        }
        if winner == str(twin.get("id")):
            return [tagged_twin, tagged_first, *rest], "ca3"
        return [tagged_first, tagged_twin, *rest], "ca3"

    q = np.asarray(query_dg, dtype=np.float32).reshape(-1)
    q = q / max(float(np.linalg.norm(q)), 1e-8)

    def _dg_score(item: dict) -> float:
        eid = str(item.get("id"))
        idx = graph.id_to_index.get(eid)
        if idx is None:
            return -1e9
        dg_vec = graph.dg_embeddings[idx].detach().cpu().numpy().astype(np.float32).reshape(-1)
        dg_vec = dg_vec / max(float(np.linalg.norm(dg_vec)), 1e-8)
        return float(dg_vec @ q)

    d0, d1 = _dg_score(first), _dg_score(twin)
    mix = min(max(float(dg_mix), 0.0), 1.0)
    h0 = (1.0 - mix) * s0 + mix * d0
    h1 = (1.0 - mix) * st + mix * d1
    tagged_first = {
        **first,
        "semantic_score": s0,
        "dg_score": d0,
        "score": h0,
        "twin_id": str(twin.get("id")),
    }
    tagged_twin = {
        **twin,
        "semantic_score": st,
        "dg_score": d1,
        "score": h1,
        "twin_id": str(first.get("id")),
    }
    dg_lead = d1 - d0
    sem_lead = s0 - st
    if h1 > h0 and dg_lead > sem_lead:
        return [tagged_twin, tagged_first, *rest], "dg_tiebreak"
    return [tagged_first, tagged_twin, *rest], "semantic"


def _node_payload(item: dict, path: str) -> dict:
    data = item.get("data") or {}
    text = data.get("text") or item.get("text") or ""
    created_at = data.get("created_at") or item.get("created_at")
    return {
        "id": str(item.get("id")),
        "score": float(item.get("score", 0.0)),
        "path": path,
        "text": text,
        "created_at": created_at,
        "semantic_score": item.get("semantic_score"),
        "dg_score": item.get("dg_score"),
        "lex_overlap": item.get("lex_overlap"),
        "ca3_score": item.get("ca3_score"),
        "twin_id": item.get("twin_id"),
        "ambiguous": path == "ambiguous_twins",
    }


class EpisodicMemory:
    """Journal d'épisodes au-dessus de HCANN + graphe hébbien."""

    def __init__(
        self,
        model: HCANN,
        learner: ContinualLearner,
        device: torch.device,
        tie_margin: float = DEFAULT_TIE_MARGIN,
        dg_mix: float = DEFAULT_DG_MIX,
        twin_min_sim: float = DEFAULT_TWIN_MIN_SIM,
        ca3_margin: float = DEFAULT_CA3_MARGIN,
    ):
        self.model = model
        self.learner = learner
        self.device = device
        self.tie_margin = tie_margin
        self.dg_mix = dg_mix
        self.twin_min_sim = twin_min_sim
        self.ca3_margin = ca3_margin
        self._rag = RagBaseline()

    @classmethod
    def create(
        cls,
        *,
        use_clip: bool = False,
        device: torch.device | str | None = None,
        enable_dreaming: bool = False,
        tie_margin: float = DEFAULT_TIE_MARGIN,
        dg_mix: float = DEFAULT_DG_MIX,
        twin_min_sim: float = DEFAULT_TWIN_MIN_SIM,
        ca3_margin: float = DEFAULT_CA3_MARGIN,
    ) -> "EpisodicMemory":
        from utils.clip_checkpoint import resolve_clip_model

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        cfg = HCANNConfig(use_mock_encoder=not use_clip, enable_dreaming=enable_dreaming)
        clip_path = resolve_clip_model(use_clip)
        if clip_path:
            cfg.clip_model = clip_path
            cfg.clip_local_files_only = True

        model = HCANN(cfg).to(device)
        learner = ContinualLearner(model, cfg, device)
        return cls(
            model,
            learner,
            device,
            tie_margin=tie_margin,
            dg_mix=dg_mix,
            twin_min_sim=twin_min_sim,
            ca3_margin=ca3_margin,
        )

    @property
    def uses_mock_encoder(self) -> bool:
        return bool(getattr(self.model.encoder, "use_mock", True))

    @property
    def graph(self):
        return self.learner.graph.graph

    def _encode_query(
        self, text: str | None, image: Image.Image | torch.Tensor | None
    ):
        imgs = image_to_batch(image, self.device) if image is not None else None
        texts = [text] if text is not None else None
        if imgs is None and texts is None:
            raise ValueError("recall / deja_vu : fournir text et/ou image")
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            ca3, sem, _, novelty, dg, extras = self.model.fast_encode(
                images=imgs, texts=texts, store=False
            )
        if was_training:
            self.model.train()
        return ca3, sem, novelty, dg, extras

    def encode(
        self,
        episode_id: str,
        text: str,
        image: Image.Image | torch.Tensor | None = None,
        created_at: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Enregistre un épisode (image + texte + date) dans le graphe."""
        eid = str(episode_id)
        imgs = image_to_batch(image, self.device) if image is not None else None
        meta = {"text": text, "created_at": created_at}
        if extra:
            meta.update(extra)
        self.learner.train_step(
            images=imgs,
            texts=[text],
            episode_id=[eid],
            metadata=[meta],
        )
        self._sync_rag()
        return eid

    def deja_vu(
        self,
        text: str | None = None,
        image: Image.Image | torch.Tensor | None = None,
    ) -> dict:
        """Déjà-vu journal : plus proche souvenir stocké (cosine) + novelty CA1."""
        _, sem, novelty, _, extras = self._encode_query(text, image)
        nov = float(novelty[0].item()) if novelty.numel() else 0.0
        is_novel_ca1 = extras.get("is_novel")
        thresh = float(getattr(self.model.config, "ca1_novelty_threshold", 0.35))
        if is_novel_ca1 is None:
            ca1_novel = nov > thresh
        else:
            ca1_novel = bool(is_novel_ca1.reshape(-1)[0].item())

        sem_np = sem[0].detach().cpu().numpy()
        nearest = self.learner.graph.retrieve_semantic(sem_np, k=1)
        nearest_id = str(nearest[0]["id"]) if nearest else None
        nearest_cos = float(nearest[0]["score"]) if nearest else 0.0
        seen = bool(nearest) and nearest_cos >= (1.0 - thresh)
        return {
            "novelty": round(1.0 - nearest_cos, 4),
            "is_novel": not seen,
            "deja_vu": seen,
            "nearest_id": nearest_id,
            "nearest_cosine": round(nearest_cos, 4),
            "ca1_novelty": round(nov, 4),
            "ca1_deja_vu": not ca1_novel,
        }

    def recall_from_codes(
        self,
        sem: np.ndarray | torch.Tensor,
        dg: np.ndarray | torch.Tensor,
        k: int = 5,
        apply_tiebreak: bool = True,
        query_text: str | None = None,
        visual_only: bool = False,
    ) -> list[dict]:
        """Rappel à partir de codes déjà calculés (tests / ablation)."""
        if isinstance(sem, torch.Tensor):
            sem_np = sem.detach().cpu().numpy()
        else:
            sem_np = np.asarray(sem)
        if sem_np.ndim > 1:
            sem_np = sem_np[0]
        if isinstance(dg, torch.Tensor):
            dg_np = dg.detach().cpu().numpy()
        else:
            dg_np = np.asarray(dg)
        if dg_np.ndim > 1:
            dg_np = dg_np[0]

        hits = self.learner.graph.retrieve_semantic(sem_np, k=k)
        if not apply_tiebreak:
            return [_node_payload(item, "semantic") for item in hits]
        hits, path = apply_dg_tiebreak(
            hits,
            dg_np,
            self.graph,
            margin=self.tie_margin,
            dg_mix=self.dg_mix,
            twin_min_sim=self.twin_min_sim,
            query_text=query_text,
            query_sem=sem_np,
            ca3=self.model.ca3,
            ca3_margin=self.ca3_margin,
            visual_only=visual_only,
        )
        return [_node_payload(item, path) for item in hits]

    def recall(
        self,
        query_text: str | None = None,
        image: Image.Image | torch.Tensor | None = None,
        k: int = 5,
        tiebreak: str = "auto",
    ) -> list[dict]:
        """Rappel : identité, sinon complétion CA3 sur les jumeaux, sinon paire ambiguë."""
        mode = (tiebreak or "auto").lower()
        if mode not in {"auto", "always", "never"}:
            raise ValueError("tiebreak doit être auto, always ou never")
        _, sem, _, dg, _ = self._encode_query(query_text, image)
        apply = mode != "never"
        visual_only = image is not None and not query_text
        return self.recall_from_codes(
            sem[0],
            dg[0],
            k=k,
            apply_tiebreak=apply,
            query_text=query_text,
            visual_only=visual_only,
        )

    def rag_recall(
        self,
        query_text: str | None = None,
        image: Image.Image | torch.Tensor | None = None,
        k: int = 5,
    ) -> list[dict]:
        """Baseline cosine CLIP (même encodeur, sans tie-break DG)."""
        _, sem, _, _, _ = self._encode_query(query_text, image)
        if not self._rag.ids:
            return []
        ids = self._rag.retrieve_ids(sem[0], k=k)
        q = sem[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        mat = self._rag._matrix
        id_to_idx = {eid: i for i, eid in enumerate(self._rag.ids)}
        out = []
        for eid in ids:
            score = float(mat[id_to_idx[eid]] @ q) if eid in id_to_idx else 0.0
            node = self.graph.nodes.get(eid, {})
            data = node.get("data") or node
            out.append(
                {
                    "id": eid,
                    "score": score,
                    "path": "rag_clip",
                    "text": data.get("text", ""),
                    "created_at": data.get("created_at"),
                    "semantic_score": score,
                }
            )
        return out

    def _sync_rag(self) -> None:
        rag = RagBaseline()
        g = self.graph
        for i in range(g.count):
            eid = g.node_ids[i]
            if not eid:
                continue
            rag.add_episode(eid, g.sem_embeddings[i])
        self._rag = rag
