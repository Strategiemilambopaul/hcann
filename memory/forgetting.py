"""Oubli unifié graphe + CA3 + EpisodeIndex (episodic_mode)."""

from __future__ import annotations

import math
import warnings

import torch


class MemoryManager:
    """Valeur multi-critères et éviction synchronisée des trois mémoires."""

    def __init__(self, config):
        self.config = config
        self.forced_evictions = 0
        self.last_evicted: list[str] = []

    def values(self, graph) -> torch.Tensor:
        """Une valeur par slot sous le high-water mark ; -inf si invalide."""
        n = graph.count
        out = torch.full((n,), float("-inf"), device=graph.device)
        if n == 0:
            return out
        w_rec = float(getattr(self.config, "forget_w_rec", 1.0))
        w_sur = float(getattr(self.config, "forget_w_sur", 1.0))
        w_rcl = float(getattr(self.config, "forget_w_rcl", 1.5))
        w_deg = float(getattr(self.config, "forget_w_deg", 0.5))
        w_red = float(getattr(self.config, "forget_w_red", 1.0))
        # Si forget_tau est resté au défaut d'un max_nodes plus grand, caler sur le graphe.
        tau_cfg = getattr(self.config, "forget_tau", None)
        tau = float(tau_cfg if tau_cfg is not None else graph.max_nodes)
        if tau <= 0:
            tau = float(graph.max_nodes)
        # Heuristique : tau >> capacité courante ⇒ récence plate ; recentrer.
        if tau > 4.0 * float(graph.max_nodes):
            tau = float(graph.max_nodes)
        clock = int(getattr(graph, "clock", 0))

        degrees = []
        recalls = []
        for i in range(n):
            if not bool(graph.valid[i].item()):
                continue
            degrees.append(float((graph.edges[i, :n] > 0.05).sum().item()))
            nid = graph.node_ids[i]
            data = (graph.nodes.get(nid) or {}).get("data") or {}
            recalls.append(int(data.get("recall_count", 0)))
        deg_max = max(degrees) if degrees else 1.0
        rc_max = max(recalls) if recalls else 1.0
        deg_max = max(deg_max, 1.0)
        rc_max = max(rc_max, 1.0)

        for i in range(n):
            if not bool(graph.valid[i].item()):
                continue
            nid = graph.node_ids[i]
            node = graph.nodes.get(nid) or {}
            data = node.get("data") or {}
            born = int(data.get("born", clock))
            age = max(0, clock - born)
            recency = math.exp(-age / max(tau, 1e-6))
            surprise = float(max(0.0, min(1.0, float(data.get("surprise", 0.0)))))
            rcl = math.log1p(int(data.get("recall_count", 0))) / math.log1p(rc_max)
            deg = math.log1p(float((graph.edges[i, :n] > 0.05).sum().item())) / math.log1p(deg_max)
            red = float(data.get("schema_cos", 0.0) or 0.0)
            red = max(0.0, min(1.0, red))
            out[i] = (
                w_rec * recency
                + w_sur * surprise
                + w_rcl * rcl
                + w_deg * deg
                - w_red * red
            )
        return out

    def _candidates(self, graph, protect: bool) -> list[int]:
        n = graph.count
        protect_age = float(getattr(self.config, "protect_frac", 0.2)) * float(graph.max_nodes)
        clock = int(getattr(graph, "clock", 0))
        cands = []
        for i in range(n):
            if not bool(graph.valid[i].item()):
                continue
            nid = graph.node_ids[i]
            data = (graph.nodes.get(nid) or {}).get("data") or {}
            if data.get("type") == "schema":
                continue
            if protect:
                born = int(data.get("born", clock))
                if (clock - born) < protect_age:
                    continue
            cands.append(i)
        return cands

    def evict(self, graph, ca3, episodes, n: int) -> list[str]:
        """Évince les n nœuds de plus faible valeur (sync CA3 + EpisodeIndex)."""
        if n <= 0:
            return []
        vals = self.values(graph)
        cands = self._candidates(graph, protect=True)
        if len(cands) < n:
            self.forced_evictions += 1
            cands = self._candidates(graph, protect=False)
        if not cands:
            return []
        scored = sorted(cands, key=lambda i: float(vals[i].item()))
        chosen = scored[:n]
        removed = []
        for idx in chosen:
            nid = graph.node_ids[idx]
            if not nid:
                continue
            graph.remove_node(nid)
            if ca3 is not None:
                ca3.remove(nid)
            if episodes is not None:
                episodes.remove(nid)
            removed.append(nid)
        self.last_evicted = removed
        return removed

    def enforce(self, graph, ca3, episodes) -> list[str]:
        """Hystérésis : évince jusqu'à max_nodes - ceil(forget_frac * max_nodes)."""
        frac = float(getattr(self.config, "forget_frac", 0.05))
        target = graph.max_nodes - int(math.ceil(frac * graph.max_nodes))
        target = max(1, target)
        if graph.num_valid < graph.max_nodes:
            return []
        need = graph.num_valid - target
        return self.evict(graph, ca3, episodes, need)

    def reconcile(self, graph, ca3, episodes) -> dict:
        """Aligne CA3, graphe et EpisodeIndex. Retourne des compteurs."""
        stats = {"ca3_removed": 0, "ca3_restored": 0, "missing_trace": 0, "episodes_removed": 0}
        graph_ids = {
            nid for nid, node in graph.nodes.items()
            if (node.get("data") or {}).get("type") != "schema"
        }
        for key in list(ca3.keys()):
            if key not in graph_ids:
                if ca3.remove(key):
                    stats["ca3_removed"] += 1
        ca3_keys = set(ca3.keys())
        for nid in list(graph_ids):
            if nid in ca3_keys:
                continue
            idx = graph.id_to_index.get(nid)
            if idx is None:
                continue
            if ca3.num_patterns >= ca3.config.max_patterns:
                stats["missing_trace"] += 1
                continue
            dg = graph.dg_embeddings[idx].detach()
            ca3.store(dg, key=nid)
            stats["ca3_restored"] += 1
        if episodes is not None:
            for mid in list(episodes.all_ids()):
                if mid not in graph.nodes:
                    episodes.remove(mid)
                    stats["episodes_removed"] += 1
        return stats


def warn_capacity(config) -> None:
    max_p = int(getattr(config, "max_patterns", 0))
    max_n = int(getattr(config, "max_nodes", 0))
    if max_p < max_n:
        warnings.warn(
            f"episodic_mode: max_patterns ({max_p}) < max_nodes ({max_n}) ; "
            "CA3 et graphe divergeront sous charge.",
            stacklevel=2,
        )
