"""Sémantisation : schémas récurrents extraits pendant le sommeil."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SchemaResult:
    schema_id: str | None
    support: int
    prototype: dict | None
    confidence: float
    members_sample: list[str]


class SchemaBuilder:
    """Leader clustering sur les embeddings sémantiques des épisodes."""

    def __init__(self, config):
        self.config = config
        self._calls = 0
        self._next_id = 0

    def maybe_build(self, graph, llm_callback=None) -> list[str]:
        every = int(getattr(self.config, "schema_every", 1))
        self._calls += 1
        if every <= 0 or (self._calls % every) != 0:
            return []
        return self.build(graph, llm_callback=llm_callback)

    def build(self, graph, llm_callback=None) -> list[str]:
        tau = float(getattr(self.config, "schema_tau", 0.8))
        min_support = int(getattr(self.config, "schema_min_support", 5))
        min_segments = int(getattr(self.config, "schema_min_segments", 2))

        # 1) Rattacher les orphelins aux schémas existants (mise à jour incrémentale)
        self._attach_to_existing(graph, tau)

        episode_ids = []
        for nid, node in graph.nodes.items():
            data = node.get("data") or {}
            if data.get("type") == "schema":
                continue
            if data.get("schema_of"):
                continue
            episode_ids.append(nid)
        if not episode_ids:
            return []

        groups: list[dict] = []
        for nid in episode_ids:
            idx = graph.id_to_index[nid]
            sem = F.normalize(graph.sem_embeddings[idx].detach().float(), dim=0)
            best_g, best_cos = None, -1.0
            for g in groups:
                cos = float(torch.dot(sem, g["centroid"]).item())
                if cos > best_cos:
                    best_cos, best_g = cos, g
            if best_g is not None and best_cos >= tau:
                support = best_g["support"]
                best_g["centroid"] = F.normalize(
                    (support * best_g["centroid"] + sem) / (support + 1), dim=0
                )
                best_g["support"] = support + 1
                best_g["members"].append(nid)
                best_g["cos"][nid] = best_cos
            else:
                groups.append(
                    {
                        "centroid": sem.clone(),
                        "support": 1,
                        "members": [nid],
                        "cos": {nid: 1.0},
                    }
                )

        created = []
        for g in groups:
            g["members"] = [m for m in g["members"] if m in graph.nodes]
            g["support"] = len(g["members"])
            if g["support"] < min_support:
                continue
            segments = set()
            for mid in g["members"]:
                seg = (graph.nodes[mid].get("data") or {}).get("segment_id")
                if seg is not None:
                    segments.add(seg)
            if len(segments) < min_segments:
                continue
            sid = self._materialize(graph, g, segments, llm_callback)
            if sid:
                created.append(sid)
        return created

    def _attach_to_existing(self, graph, tau: float) -> None:
        schemas = []
        for nid, node in graph.nodes.items():
            data = node.get("data") or {}
            if data.get("type") != "schema":
                continue
            idx = graph.id_to_index[nid]
            cent = F.normalize(graph.sem_embeddings[idx].detach().float(), dim=0)
            schemas.append((nid, data, cent, int(data.get("support", 1))))
        if not schemas:
            return
        for nid, node in list(graph.nodes.items()):
            data = node.get("data") or {}
            if data.get("type") == "schema" or data.get("schema_of"):
                continue
            idx = graph.id_to_index[nid]
            sem = F.normalize(graph.sem_embeddings[idx].detach().float(), dim=0)
            best_sid, best_cos, best_i = None, -1.0, -1
            for i, (sid, sdata, cent, support) in enumerate(schemas):
                cos = float(torch.dot(sem, cent).item())
                if cos > best_cos:
                    best_cos, best_sid, best_i = cos, sid, i
            if best_sid is None or best_cos < tau:
                continue
            sid, sdata, cent, support = schemas[best_i]
            new_cent = F.normalize((support * cent + sem) / (support + 1), dim=0)
            sdata["support"] = support + 1
            members = list(sdata.get("members") or [])
            if nid not in members:
                members.append(nid)
            sdata["members"] = members[:50]
            segs = set(sdata.get("segments") or [])
            seg = data.get("segment_id")
            if seg is not None:
                segs.add(str(seg))
            sdata["segments"] = sorted(segs)
            sidx = graph.id_to_index[sid]
            graph.sem_embeddings[sidx] = new_cent.to(graph.sem_embeddings.dtype)
            # moyenne DG incrémentale
            dg = graph.dg_embeddings[idx].detach().float()
            old_dg = graph.dg_embeddings[sidx].detach().float()
            graph.dg_embeddings[sidx] = F.normalize(
                (support * old_dg + dg) / (support + 1), dim=0
            ).to(graph.dg_embeddings.dtype)
            data["schema_of"] = sid
            data["schema_cos"] = float(best_cos)
            graph.update_hebbian_weights(nid, sid, 0.1)
            schemas[best_i] = (sid, sdata, new_cent, support + 1)

    def _materialize(self, graph, group: dict, segments: set, llm_callback) -> str | None:
        members = group["members"][:50]
        best_id, best_cos = None, -1.0
        for mid in members:
            idx = graph.id_to_index[mid]
            sem = F.normalize(graph.sem_embeddings[idx].detach().float(), dim=0)
            cos = float(torch.dot(sem, group["centroid"]).item())
            if cos > best_cos:
                best_cos, best_id = cos, mid
        if best_id is None:
            return None

        hpc_stack = []
        dg_stack = []
        for mid in members:
            idx = graph.id_to_index[mid]
            hpc_stack.append(graph.node_embeddings[idx].detach())
            dg_stack.append(graph.dg_embeddings[idx].detach())
        hpc_mean = F.normalize(torch.stack(hpc_stack, dim=0).mean(dim=0).float(), dim=0)
        dg_mean = F.normalize(torch.stack(dg_stack, dim=0).mean(dim=0).float(), dim=0)
        sem_np = group["centroid"].detach().cpu().numpy().astype(np.float32)
        hpc_np = hpc_mean.detach().cpu().numpy().astype(np.float32)
        dg_np = dg_mean.detach().cpu().numpy().astype(np.float32)

        payloads = [(graph.nodes[m].get("data") or {}) for m in members]
        if llm_callback is not None:
            summary = llm_callback(payloads)
        else:
            summary = (graph.nodes[best_id].get("data") or {}).copy()

        sid = f"schema:{self._next_id}"
        self._next_id += 1
        data = {
            "type": "schema",
            "support": int(group["support"]),
            "members": list(members),
            "segments": sorted(str(s) for s in segments),
            "prototype_id": best_id,
            "summary": summary,
            "born": int(getattr(graph, "clock", 0)),
            "surprise": 0.0,
            "recall_count": 0,
            "last_recall": -1,
        }
        graph.add_node(sid, data, hpc_np, sem_embedding=sem_np, dg_embedding=dg_np)
        for mid in members:
            node = graph.nodes.get(mid)
            if node is None:
                continue
            d = node.setdefault("data", {})
            d["schema_of"] = sid
            d["schema_cos"] = float(group["cos"].get(mid, best_cos))
            graph.update_hebbian_weights(mid, sid, 0.1)
        return sid
