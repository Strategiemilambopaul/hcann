"""Rappel ordonné avec score de confiance (mode épisodique)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from memory.schemas import SchemaResult


@dataclass
class RecalledEvent:
    node_id: str
    segment_id: str | None
    event_idx: int | None
    data: dict
    confidence: float


@dataclass
class RecallResult:
    events: list[RecalledEvent]
    confidence: float
    level: str
    features: dict[str, float]
    key: str | None = None
    schema_id: str | None = None


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class EpisodicRecall:
    """Complétion CA3 + reconstruction ordonnée + confiance calibrable."""

    def __init__(self, model, graph, episodes, config):
        self.model = model
        self.graph = graph
        self.episodes = episodes
        self.config = config
        weights = list(getattr(config, "recall_weights", [2.0, 1.5, 3.0, 0.5, -1.5]))
        self.weights = [float(w) for w in weights]

    def _feature_vector(self, features: dict[str, float]) -> list[float]:
        return [
            features["familiarity"],
            features["completion_cos"],
            features["margin"],
            features["energy_norm"],
            1.0,
        ]

    def _confidence(self, features: dict[str, float]) -> float:
        vec = self._feature_vector(features)
        score = sum(w * v for w, v in zip(self.weights, vec))
        return float(_sigmoid(score))

    def _level(self, conf: float) -> str:
        hi = float(getattr(self.config, "recall_hi", 0.7))
        lo = float(getattr(self.config, "recall_lo", 0.4))
        if conf >= hi:
            return "clair"
        if conf >= lo:
            return "flou"
        return "inconnu"

    def _node_payload(self, node_id: str) -> dict:
        g = self.graph.graph if hasattr(self.graph, "graph") else self.graph
        node = g.nodes.get(str(node_id), {})
        return node.get("data") or {}

    def _segment_from(self, node_id: str, follow: int | None) -> list[str]:
        data = self._node_payload(node_id)
        seg = data.get("segment_id")
        start = int(data.get("event_idx", 0))
        if self.episodes is None or seg is None:
            return [node_id]
        members = self.episodes.members(seg)
        ordered = []
        for mid in members:
            mdata = self._node_payload(mid)
            if int(mdata.get("event_idx", -1)) >= start:
                ordered.append(mid)
        ordered.sort(key=lambda mid: int(self._node_payload(mid).get("event_idx", 0)))
        if follow is not None:
            ordered = ordered[: max(1, int(follow))]
        return ordered or [node_id]

    def _chain_from(self, node_id: str, follow: int | None) -> tuple[list[str], list[float]]:
        max_len = int(getattr(self.config, "recall_max_chain", 8))
        floor = float(getattr(self.config, "recall_succ_floor", 0.02))
        if follow is not None:
            max_len = min(max_len, max(1, int(follow)))
        path = [node_id]
        margins = []
        seen = {node_id}
        current = node_id
        while len(path) < max_len:
            succ = self.graph.successors(current, k=3)
            if not succ:
                break
            nxt = succ[0]
            if float(nxt["weight"]) < floor or nxt["id"] in seen:
                break
            top_w = float(nxt["weight"])
            second = float(succ[1]["weight"]) if len(succ) > 1 else 0.0
            margins.append(max(0.0, min(1.0, top_w / (top_w + second + 1e-6))) if top_w > 0 else 0.0)
            path.append(nxt["id"])
            seen.add(nxt["id"])
            current = nxt["id"]
        return path, margins

    def _encode_cue(self, cue_texts, cue_images):
        mode = str(getattr(self.config, "recall_ctx_mode", "none"))
        ctx_mode = mode if mode in {"none", "current"} else "none"
        return self.model.fast_encode(
            images=cue_images,
            texts=cue_texts,
            store=False,
            ctx_mode=ctx_mode,
        )

    def _features_from_cue(self, dg_code: torch.Tensor) -> tuple[dict[str, float], str | None, torch.Tensor]:
        ca3 = self.model.ca3
        if dg_code.dim() == 1:
            dg_code = dg_code.unsqueeze(0)
        familiarity = float(ca3.max_pattern_similarity(dg_code)[0].item())
        completed = ca3.complete(dg_code)
        q = F.normalize(dg_code, dim=-1)
        c = F.normalize(completed, dim=-1)
        completion_cos = float((q * c).sum(dim=-1)[0].item())
        idxs, cosines = ca3.nearest(completed, k=2)
        if cosines.numel() == 0:
            features = {
                "familiarity": 0.0,
                "completion_cos": 0.0,
                "margin": 0.0,
                "energy_norm": 0.0,
            }
            return features, None, completed
        top1 = float(cosines[0].item())
        top2 = float(cosines[1].item()) if cosines.numel() > 1 else 0.0
        margin = top1 - top2
        energy = float(ca3.energy(completed)[0].item())
        beta = float(getattr(self.config, "hopfield_beta", 8.0))
        denom = beta * math.log(max(ca3.num_patterns, 2))
        energy_norm = float(-energy / max(denom, 1e-6))
        key = ca3.key_of(int(idxs[0].item()))
        if key is None:
            margin *= 0.5
            familiarity *= 0.5
        elif key not in self.graph.graph.nodes:
            margin *= 0.25
            familiarity *= 0.25
            key = None
        features = {
            "familiarity": familiarity,
            "completion_cos": completion_cos,
            "margin": margin,
            "energy_norm": energy_norm,
        }
        return features, key, completed

    def recall(
        self,
        cue_texts=None,
        cue_images=None,
        scope: str = "segment",
        follow: int | None = None,
        touch: bool = True,
        cue_dg: torch.Tensor | None = None,
    ) -> RecallResult:
        if not bool(getattr(self.config, "episodic_mode", False)):
            return RecallResult([], 0.0, "inconnu", {}, None, None)
        if cue_dg is not None:
            dg_code = cue_dg
            if not isinstance(dg_code, torch.Tensor):
                dg_code = torch.as_tensor(dg_code, dtype=torch.float32)
            if dg_code.dim() == 1:
                dg_code = dg_code.unsqueeze(0)
            dg_code = dg_code.to(next(self.model.parameters()).device)
        else:
            _, _, _, _, dg_code, _ = self._encode_cue(cue_texts, cue_images)
        features, key, completed = self._features_from_cue(dg_code)
        conf = self._confidence(features)
        fam_floor = float(getattr(self.config, "recall_fam_floor", 0.35))
        if features.get("familiarity", 0.0) < fam_floor:
            conf = min(conf, float(getattr(self.config, "recall_lo", 0.4)) - 1e-3)
        level = self._level(conf)
        if level == "inconnu" or key is None:
            return RecallResult([], conf, "inconnu", features, key, None)

        if scope == "chain":
            ids, step_margins = self._chain_from(key, follow)
        else:
            ids = self._segment_from(key, follow)
            step_margins = []

        events: list[RecalledEvent] = []
        running = 1.0
        schema_id = None
        for i, nid in enumerate(ids):
            if i > 0 and step_margins:
                running *= max(0.05, step_margins[i - 1] if i - 1 < len(step_margins) else 0.5)
            data = self._node_payload(nid)
            if schema_id is None:
                schema_id = data.get("schema_of")
            events.append(
                RecalledEvent(
                    node_id=nid,
                    segment_id=data.get("segment_id"),
                    event_idx=data.get("event_idx"),
                    data=data,
                    confidence=float(conf * running),
                )
            )

        if touch and conf >= float(getattr(self.config, "recon_min_conf", 0.6)):
            self._reconsolidate(key, dg_code, completed, ids)

        return RecallResult(events, conf, level, features, key, schema_id)

    def _reconsolidate(self, key: str, cue_dg: torch.Tensor, attractor: torch.Tensor, ids: list[str]) -> None:
        target_mode = str(getattr(self.config, "recon_target", "attractor"))
        target = attractor if target_mode != "cue" else cue_dg
        lr = float(getattr(self.config, "recon_lr", 0.05))
        self.model.ca3.reconsolidate(key, target, lr)
        g = self.graph.graph
        now = time.time()
        clock = int(getattr(g, "clock", 0))
        for nid in ids:
            if nid not in g.nodes:
                continue
            node = g.nodes[nid]
            node["access_count"] = node.get("access_count", 0) + 1
            data = node.setdefault("data", {})
            data["recall_count"] = int(data.get("recall_count", 0)) + 1
            data["last_recall"] = clock
            node["timestamp"] = now
            idx = g.id_to_index.get(nid)
            if idx is not None:
                g.access_counts[idx] += 1
                g.timestamps[idx] = now
        for a, b in zip(ids, ids[1:]):
            self.graph.link_next(a, b, 0.05)
            self.graph.update_connections(a, [b], strength=0.05)

    def fit_calibration(self, features: list[dict], correct: list[int], steps: int = 500) -> float:
        """Régression logistique torch sur les caractéristiques -> confiance."""
        x = torch.tensor([self._feature_vector(f) for f in features], dtype=torch.float32)
        y = torch.tensor(correct, dtype=torch.float32).reshape(-1, 1)
        w = torch.nn.Parameter(torch.tensor(self.weights, dtype=torch.float32))
        opt = torch.optim.Adam([w], lr=0.05)
        for _ in range(int(steps)):
            opt.zero_grad()
            logits = x @ w.reshape(-1, 1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
        self.weights = [float(v) for v in w.detach().tolist()]
        self.config.recall_weights = list(self.weights)
        with torch.no_grad():
            probs = torch.sigmoid(x @ w.reshape(-1, 1)).reshape(-1)
        return float(probs.mean().item())

    def recall_general(
        self,
        cue_texts=None,
        cue_images=None,
        k: int = 1,
    ) -> list[SchemaResult]:
        """Rappel de schémas par cosinus sémantique ; s'abstient sous le seuil."""
        if not bool(getattr(self.config, "episodic_mode", False)):
            return []
        _, sem, _, _, _, _ = self.model.fast_encode(
            images=cue_images, texts=cue_texts, store=False, ctx_mode="none"
        )
        q = F.normalize(sem[0].detach().float(), dim=0)
        g = self.graph.graph
        floor = float(getattr(self.config, "schema_recall_floor", 0.6))
        scored = []
        for nid, node in g.nodes.items():
            data = node.get("data") or {}
            if data.get("type") != "schema":
                continue
            idx = g.id_to_index[nid]
            s = F.normalize(g.sem_embeddings[idx].detach().float(), dim=0)
            cos = float(torch.dot(q, s).item())
            if cos >= floor:
                scored.append((cos, nid, data))
        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for cos, nid, data in scored[: max(1, int(k))]:
            proto_id = data.get("prototype_id")
            proto = None
            if proto_id and proto_id in g.nodes:
                proto = g.nodes[proto_id].get("data")
            out.append(
                SchemaResult(
                    schema_id=nid,
                    support=int(data.get("support", 0)),
                    prototype=proto,
                    confidence=cos,
                    members_sample=list(data.get("members", []))[:8],
                )
            )
        return out

    def narrate(self, result: RecallResult, llm_callback: Callable[[list[dict]], str] | None = None) -> str:
        payloads = [e.data for e in result.events]
        if llm_callback is not None:
            return llm_callback(payloads)
        if not result.events:
            return "aucun souvenir assez sûr"
        parts = []
        for event in result.events:
            text = event.data.get("text") or event.node_id
            parts.append(f"{event.node_id}:{text}")
        return " -> ".join(parts)
