"""Unité de mesure : journal d'épisodes, pas un système.

N'importe quelle mémoire qui encode un épisode (id, texte, image, temps)
et rappelle une liste classée peut passer ce banc. HCANN est une
implémentation de référence, pas le mètre.
"""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image

from integrations.episodic_scenes import trace_fields
from integrations.episodic_trace import build_episode_meta


class EpisodicStore(Protocol):
    """Contrat minimal pour mesurer une IA sur des épisodes vécus."""

    name: str

    def encode(
        self,
        episode_id: str,
        text: str,
        image: Image.Image | None = None,
        created_at: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str: ...

    def recall(
        self,
        query_text: str | None = None,
        image: Image.Image | None = None,
        k: int = 5,
    ) -> list[dict]: ...


def committed(hits: list[dict]) -> bool:
    """Vainqueur unique : pas une paire déclarée ambiguë."""
    if not hits:
        return False
    h0 = hits[0]
    if h0.get("ambiguous") or h0.get("path") == "ambiguous_twins":
        return False
    return True


def score_hits(hits: list[dict], target_id: str, twin_id: str) -> dict:
    """Métriques du mètre : confusion exclusive, hit, paire@2."""
    ids = [str(h.get("id")) for h in hits[:2] if h.get("id") is not None]
    top1 = ids[0] if ids else None
    pair = {target_id, twin_id}
    top2 = set(ids)
    did_commit = committed(hits)
    return {
        "top1": top1,
        "committed": did_commit,
        "confused": did_commit and top1 == twin_id,
        "hit": (target_id in top2) if not did_commit else (top1 == target_id),
        "pair_at_2": pair <= top2,
        "path": hits[0].get("path") if hits else None,
        "top": [
            {"id": str(h.get("id")), "score": round(float(h.get("score", 0.0)), 4)}
            for h in hits[:3]
        ],
    }


def summarize_system(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {
            "n_queries": 0,
            "confusion_at_1": 0.0,
            "hit": 0.0,
            "pair_at_2": 0.0,
            "n_uncommitted": 0,
        }
    return {
        "n_queries": n,
        "confusion_at_1": sum(1 for r in rows if r["confused"]) / n,
        "hit": sum(1 for r in rows if r["hit"]) / n,
        "pair_at_2": sum(1 for r in rows if r["pair_at_2"]) / n,
        "n_uncommitted": sum(1 for r in rows if not r["committed"]),
    }


def by_query(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for qname in sorted({r["query"] for r in rows}):
        sub = [r for r in rows if r["query"] == qname]
        block = summarize_system(sub)
        block["n"] = len(sub)
        out[qname] = block
    return out


class HcannStore:
    name = "hcann"

    def __init__(self, memory, tiebreak: str = "auto"):
        self.memory = memory
        self.tiebreak = tiebreak

    def encode(self, episode_id, text, image=None, created_at=None, extra=None):
        return self.memory.encode(
            episode_id, text, image=image, created_at=created_at, extra=extra
        )

    def recall(self, query_text=None, image=None, k=5):
        return self.memory.recall(
            query_text=query_text, image=image, k=k, tiebreak=self.tiebreak
        )


class RagStore:
    """Cosine CLIP plat sur le même journal (baseline, pas un second encode)."""

    name = "rag"

    def __init__(self, memory):
        self.memory = memory

    def encode(self, episode_id, text, image=None, created_at=None, extra=None):
        return episode_id

    def recall(self, query_text=None, image=None, k=5):
        return self.memory.rag_recall(query_text=query_text, image=image, k=k)


def _tokens(s: str) -> set[str]:
    buf = "".join(c.lower() if c.isalnum() else " " for c in (s or ""))
    return {t for t in buf.split() if len(t) > 2}


def _lex_overlap(query: str, stored: str) -> float:
    q, s = _tokens(query), _tokens(stored)
    if not q:
        return 0.0
    return len(q & s) / len(q)


def _text_jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def score_completion(hits: list[dict], target_id: str) -> dict:
    """Hit@1 / hit@2 sur une visite unique (pas une paire jumeau)."""
    ids = [str(h.get("id")) for h in hits if h.get("id") is not None]
    top1 = ids[0] if ids else None
    did_commit = committed(hits)
    hit = top1 == target_id
    return {
        "top1": top1,
        "committed": did_commit,
        "confused": did_commit and not hit,
        "hit": hit,
        "pair_at_2": target_id in ids[:2],
        "path": hits[0].get("path") if hits else None,
        "top": [
            {"id": str(h.get("id")), "score": round(float(h.get("score", 0.0)), 4)}
            for h in hits[:3]
        ],
    }


class LexicalStore:
    """Rappel par recouvrement de mots. Pas de vision, vainqueur unique (égalité → plus récent)."""

    name = "lexical"

    def __init__(self):
        self._eps: list[dict] = []

    def encode(self, episode_id, text, image=None, created_at=None, extra=None):
        meta = build_episode_meta(episode_id, text or "", created_at, extra)
        self._eps.append(
            {
                "id": str(episode_id),
                "text": text or "",
                "created_at": created_at or "",
                "trace": meta["trace"],
                "order": len(self._eps),
            }
        )
        return str(episode_id)

    def recall(self, query_text=None, image=None, k=5):
        q = query_text or ""
        ranked = []
        for ep in self._eps:
            ov = _lex_overlap(q, ep["text"])
            ranked.append((ov, ep["order"], ep))
        ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
        out = []
        for ov, _order, ep in ranked[:k]:
            out.append(
                {
                    "id": ep["id"],
                    "score": float(ov),
                    "path": "lexical",
                    "text": ep["text"],
                    "created_at": ep["created_at"],
                    "trace": ep.get("trace"),
                }
            )
        return out


class MergeStore:
    """Fusionne deux visites trop proches (geste Mem0 / Zep). Vainqueur unique."""

    name = "merge"

    def __init__(self, min_jaccard: float = 0.5):
        self.min_jaccard = min_jaccard
        self._eps: list[dict] = []
        self._alias: dict[str, str] = {}

    def encode(self, episode_id, text, image=None, created_at=None, extra=None):
        eid = str(episode_id)
        caption = text or ""
        meta = build_episode_meta(eid, caption, created_at, extra)
        for ep in self._eps:
            if _text_jaccard(caption, ep["text"]) >= self.min_jaccard:
                ep["text"] = (ep["text"] + " " + caption).strip()
                ep["merged_ids"].append(eid)
                self._alias[eid] = ep["id"]
                return ep["id"]
        self._eps.append(
            {
                "id": eid,
                "text": caption,
                "created_at": created_at or "",
                "trace": meta["trace"],
                "order": len(self._eps),
                "merged_ids": [eid],
            }
        )
        self._alias[eid] = eid
        return eid

    def recall(self, query_text=None, image=None, k=5):
        q = query_text or ""
        ranked = []
        for ep in self._eps:
            ov = _lex_overlap(q, ep["text"])
            ranked.append((ov, ep["order"], ep))
        ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
        out = []
        for ov, _order, ep in ranked[:k]:
            out.append(
                {
                    "id": ep["id"],
                    "score": float(ov),
                    "path": "merge",
                    "text": ep["text"],
                    "created_at": ep["created_at"],
                    "trace": ep.get("trace"),
                    "merged_ids": list(ep["merged_ids"]),
                }
            )
        return out


class Ca3Store:
    """Complétion Hopfield sur les motifs DG indexés par episode_id. Pas la politique jumeaux."""

    name = "ca3"

    def __init__(self, memory):
        self.memory = memory

    def encode(self, episode_id, text, image=None, created_at=None, extra=None):
        return episode_id

    def recall(self, query_text=None, image=None, k=5):
        _, _, _, dg, _ = self.memory._encode_query(query_text, image)
        return rank_ca3_patterns(self.memory.model.ca3, dg, k=k, complete=True, graph=self.memory.graph)

    def recall_dg(self, query_dg, k=5):
        return rank_ca3_patterns(
            self.memory.model.ca3, query_dg, k=k, complete=True, graph=self.memory.graph
        )


class DgKnnStore:
    """Cosine DG plat : même espace que CA3, sans dynamique Hopfield."""

    name = "dg_knn"

    def __init__(self, memory):
        self.memory = memory

    def encode(self, episode_id, text, image=None, created_at=None, extra=None):
        return episode_id

    def recall(self, query_text=None, image=None, k=5):
        _, _, _, dg, _ = self.memory._encode_query(query_text, image)
        return rank_ca3_patterns(self.memory.model.ca3, dg, k=k, complete=False, graph=self.memory.graph)

    def recall_dg(self, query_dg, k=5):
        return rank_ca3_patterns(
            self.memory.model.ca3, query_dg, k=k, complete=False, graph=self.memory.graph
        )


def rank_ca3_patterns(ca3, query_dg, k: int = 5, complete: bool = False, graph=None) -> list[dict]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    if ca3.num_patterns == 0 or not ca3._key_to_idx:
        return []
    q = torch.as_tensor(
        np.asarray(query_dg, dtype=np.float32).reshape(-1),
        device=ca3.patterns.device,
    )
    if complete:
        q = ca3.complete(q)
    q = F.normalize(q.reshape(-1), dim=0)
    ranked: list[tuple[float, str]] = []
    for key, idx in ca3._key_to_idx.items():
        if idx >= ca3.patterns.size(0):
            continue
        pat = F.normalize(ca3.patterns[idx].reshape(-1), dim=0)
        ranked.append((float((q * pat).sum().item()), str(key)))
    ranked.sort(key=lambda t: t[0], reverse=True)
    path = "ca3_complete" if complete else "dg_knn"
    out = []
    for score, eid in ranked[:k]:
        text = ""
        created_at = None
        if graph is not None:
            node = graph.nodes.get(eid) or {}
            data = node.get("data") or node
            text = data.get("text") or ""
            created_at = data.get("created_at")
        out.append(
            {
                "id": eid,
                "score": score,
                "path": path,
                "text": text,
                "created_at": created_at,
            }
        )
    return out


def stored_ca3_pattern(memory, episode_id: str):
    import numpy as np

    ca3 = memory.model.ca3
    idx = ca3._key_to_idx.get(str(episode_id))
    if idx is None or idx >= ca3.patterns.size(0):
        return None
    return ca3.patterns[idx].detach().cpu().numpy().astype(np.float32)


def perturb_dg(vec, kind: str, seed: int):
    """Amorce = motif stocké altéré, pas un re-encodage CLIP."""
    import numpy as np

    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(seed)
    if kind == "noisy_dg":
        out = v + 0.45 * rng.standard_normal(v.shape).astype(np.float32)
    elif kind == "masked_dg":
        out = v.copy()
        out[rng.random(v.shape) < 0.5] = 0.0
    else:
        raise ValueError(kind)
    nrm = float(np.linalg.norm(out))
    return out / max(nrm, 1e-8)


def evaluate_pattern_cues(memory, visits: list[dict], k: int = 5) -> list[dict]:
    """Complétion vraie : amorce = DG stocké bruité, masqué, ou mélange 60/40."""
    import numpy as np

    ca3 = Ca3Store(memory)
    knn = DgKnnStore(memory)
    rows = []
    for i, ep in enumerate(visits):
        stored = stored_ca3_pattern(memory, ep["id"])
        if stored is None:
            continue
        eid = ep["id"]
        seed = sum((j + 1) * ord(c) for j, c in enumerate(eid))
        for qname in ("noisy_dg", "masked_dg"):
            cue = perturb_dg(stored, qname, seed + (0 if qname == "noisy_dg" else 17))
            scored = {
                "dg_knn": score_completion(knn.recall_dg(cue, k=k), eid),
                "ca3": score_completion(ca3.recall_dg(cue, k=k), eid),
            }
            rows.append({"target_id": eid, "query": qname, "systems": scored})
        foil_ep = visits[(i + 1) % len(visits)]
        foil = stored_ca3_pattern(memory, foil_ep["id"])
        if foil is None:
            continue
        mix = 0.6 * stored + 0.4 * foil
        nrm = float(np.linalg.norm(mix))
        cue = mix / max(nrm, 1e-8)
        scored = {
            "dg_knn": score_completion(knn.recall_dg(cue, k=k), eid),
            "ca3": score_completion(ca3.recall_dg(cue, k=k), eid),
        }
        rows.append({"target_id": eid, "query": "mixed_dg", "systems": scored})
    return rows


def ingest(store: EpisodicStore, pairs: list[dict], distractors: list[dict]) -> int:
    n = 0
    for pair in pairs:
        for side in ("a", "b"):
            ep = pair[side]
            extra = trace_fields(ep)
            store.encode(
                ep["id"],
                ep["text"],
                image=ep.get("image"),
                created_at=ep.get("created_at"),
                extra=extra,
            )
            n += 1
    for d in distractors:
        store.encode(
            d["id"],
            d["text"],
            image=d.get("image"),
            created_at=d.get("created_at"),
            extra=trace_fields(d),
        )
        n += 1
    return n


def queries_for_target(pair: dict, target: dict, degraded) -> list[tuple[str, str | None, Any]]:
    return [
        ("shared_caption", pair["cue_text"], None),
        ("identity_cue", f"{pair['cue_text']} {target['identity']}", None),
        ("image+shared_caption", pair["cue_text"], target["image"]),
        ("degraded_image", None, degraded),
    ]


def evaluate_pair(stores: dict[str, EpisodicStore], pair: dict, k: int, degraded_fn) -> list[dict]:
    rows = []
    for target_key, twin_key in (("a", "b"), ("b", "a")):
        target = pair[target_key]
        twin = pair[twin_key]
        degraded = degraded_fn(target)
        for qname, qtext, qimg in queries_for_target(pair, target, degraded):
            scored = {}
            for name, store in stores.items():
                hits = store.recall(query_text=qtext, image=qimg, k=k)
                scored[name] = score_hits(hits, target["id"], twin["id"])
            row = {
                "pair_id": pair["pair_id"],
                "target_id": target["id"],
                "twin_id": twin["id"],
                "identity": target["identity"],
                "query": qname,
                "systems": scored,
            }
            if "hcann" in scored and "rag" in scored:
                h, r = scored["hcann"], scored["rag"]
                row.update(
                    {
                        "hcann_top1": h["top1"],
                        "rag_top1": r["top1"],
                        "hcann_path": h["path"],
                        "hcann_ambiguous": not h["committed"],
                        "hcann_hit": h["hit"],
                        "rag_hit": r["hit"],
                        "hcann_confused": h["confused"],
                        "rag_confused": r["confused"],
                        "hcann_pair_at_2": h["pair_at_2"],
                        "rag_pair_at_2": r["pair_at_2"],
                        "hcann_top": h["top"],
                        "rag_top": r["top"],
                    }
                )
            if "lexical" in scored:
                lx = scored["lexical"]
                row.update(
                    {
                        "lexical_top1": lx["top1"],
                        "lexical_hit": lx["hit"],
                        "lexical_confused": lx["confused"],
                        "lexical_pair_at_2": lx["pair_at_2"],
                        "lexical_top": lx["top"],
                    }
                )
            if "merge" in scored:
                mg = scored["merge"]
                row.update(
                    {
                        "merge_top1": mg["top1"],
                        "merge_hit": mg["hit"],
                        "merge_confused": mg["confused"],
                        "merge_pair_at_2": mg["pair_at_2"],
                        "merge_top": mg["top"],
                    }
                )
            rows.append(row)
    return rows


def ingest_visits(store: EpisodicStore, visits: list[dict]) -> int:
    n = 0
    for ep in visits:
        extra = trace_fields(ep)
        store.encode(
            ep["id"],
            ep["text"],
            image=ep.get("image"),
            created_at=ep.get("created_at"),
            extra=extra,
        )
        n += 1
    return n


def evaluate_visits(
    stores: dict[str, EpisodicStore],
    visits: list[dict],
    k: int,
    queries_fn,
) -> list[dict]:
    rows = []
    for ep in visits:
        for qname, qtext, qimg in queries_fn(ep):
            scored = {}
            for name, store in stores.items():
                hits = store.recall(query_text=qtext, image=qimg, k=k)
                scored[name] = score_completion(hits, ep["id"])
            rows.append(
                {
                    "target_id": ep["id"],
                    "query": qname,
                    "systems": scored,
                }
            )
    return rows


def compare_summaries(rows: list[dict]) -> dict:
    """Vue par système + pivot HCANN/RAG pour l'export actuel."""
    by_sys: dict[str, list[dict]] = {}
    for row in rows:
        for name, sc in row.get("systems", {}).items():
            by_sys.setdefault(name, []).append({"query": row["query"], **sc})
    systems = {name: {**summarize_system(rs), "by_query": by_query(rs)} for name, rs in by_sys.items()}
    h = systems.get("hcann") or {}
    r = systems.get("rag") or {}
    lx = systems.get("lexical") or {}
    mg = systems.get("merge") or {}
    c3 = systems.get("ca3") or {}
    n = (
        h.get("n_queries")
        or r.get("n_queries")
        or lx.get("n_queries")
        or mg.get("n_queries")
        or c3.get("n_queries")
        or 0
    )
    n_ca3 = sum(1 for row in rows if (row.get("systems") or {}).get("hcann", {}).get("path") == "ca3")
    n_amb = h.get("n_uncommitted", 0)
    n_id = sum(
        1
        for row in rows
        if (row.get("systems") or {}).get("hcann", {}).get("path") in {"identity", "ca3", "ambiguous_twins", "dg_tiebreak"}
    )
    by_q: dict[str, dict] = {}
    for qname in sorted({row["query"] for row in rows}):
        hq = (h.get("by_query") or {}).get(qname) or {}
        rq = (r.get("by_query") or {}).get(qname) or {}
        lq = (lx.get("by_query") or {}).get(qname) or {}
        mq = (mg.get("by_query") or {}).get(qname) or {}
        cq = (c3.get("by_query") or {}).get(qname) or {}
        by_q[qname] = {
            "n": hq.get("n") or rq.get("n") or lq.get("n") or mq.get("n") or cq.get("n") or 0,
            "rag_confusion_at_1": rq.get("confusion_at_1", 0.0),
            "lexical_confusion_at_1": lq.get("confusion_at_1", 0.0),
            "merge_confusion_at_1": mq.get("confusion_at_1", 0.0),
            "ca3_confusion_at_1": cq.get("confusion_at_1", 0.0),
            "hcann_confusion_at_1": hq.get("confusion_at_1", 0.0),
            "rag_hit_at_1": rq.get("hit", 0.0),
            "lexical_hit_at_1": lq.get("hit", 0.0),
            "merge_hit_at_1": mq.get("hit", 0.0),
            "ca3_hit_at_1": cq.get("hit", 0.0),
            "hcann_hit_at_1": hq.get("hit", 0.0),
            "rag_pair_at_2": rq.get("pair_at_2", 0.0),
            "lexical_pair_at_2": lq.get("pair_at_2", 0.0),
            "merge_pair_at_2": mq.get("pair_at_2", 0.0),
            "ca3_pair_at_2": cq.get("pair_at_2", 0.0),
            "hcann_pair_at_2": hq.get("pair_at_2", 0.0),
            "hcann_ambiguous": hq.get("n_uncommitted", 0),
            "hcann_ca3": sum(
                1
                for row in rows
                if row["query"] == qname
                and (row.get("systems") or {}).get("hcann", {}).get("path") == "ca3"
            ),
        }
    return {
        "n_queries": n,
        "systems": systems,
        "rag_confusion_at_1": r.get("confusion_at_1", 0.0),
        "hcann_confusion_at_1": h.get("confusion_at_1", 0.0),
        "rag_hit_at_1": r.get("hit", 0.0),
        "hcann_hit_at_1": h.get("hit", 0.0),
        "rag_pair_at_2": r.get("pair_at_2", 0.0),
        "lexical_confusion_at_1": lx.get("confusion_at_1", 0.0),
        "lexical_hit_at_1": lx.get("hit", 0.0),
        "lexical_pair_at_2": lx.get("pair_at_2", 0.0),
        "merge_confusion_at_1": mg.get("confusion_at_1", 0.0),
        "merge_hit_at_1": mg.get("hit", 0.0),
        "merge_pair_at_2": mg.get("pair_at_2", 0.0),
        "ca3_confusion_at_1": c3.get("confusion_at_1", 0.0),
        "ca3_hit_at_1": c3.get("hit", 0.0),
        "ca3_pair_at_2": c3.get("pair_at_2", 0.0),
        "hcann_pair_at_2": h.get("pair_at_2", 0.0),
        "hcann_ambiguous": n_amb,
        "hcann_ca3": n_ca3,
        "dg_tiebreak_used": n_id,
        "by_query": by_q,
    }
