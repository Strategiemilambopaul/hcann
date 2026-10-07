import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple, Optional
import numpy as np

class HebbianMemoryGraph(nn.Module):
    """HeLa-Mem + HCANN: Graphe hebbien dynamique + consolidation + spreading activation"""
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        # Supporte config dict ou objet
        if hasattr(config, 'get'):
            self.max_nodes = config.get('max_nodes', 2000)
        else:
            self.max_nodes = getattr(config, 'max_nodes', 2000)
        self.hpc_size = config.hpc_size
        self.sem_dim = getattr(config, "semantic_dim", 256)
        self.dg_dim = getattr(config, "dg_dim", config.hpc_size)
        
        # Structure de données du graphe
        self.nodes: Dict[str, dict] = {}  # node_id -> {embedding, data, timestamp, access_count, consolidated}
        self.node_embeddings = torch.zeros(self.max_nodes, self.hpc_size, device=device)
        self.sem_embeddings = torch.zeros(self.max_nodes, self.sem_dim, device=device)
        self.dg_embeddings = torch.zeros(self.max_nodes, self.dg_dim, device=device)
        self.edges = torch.zeros(self.max_nodes, self.max_nodes, device=device)
        self.timestamps = torch.zeros(self.max_nodes, device=device)
        self.access_counts = torch.zeros(self.max_nodes, device=device)
        self.succ = torch.zeros(self.max_nodes, self.max_nodes, device=device)
        self.valid = torch.zeros(self.max_nodes, dtype=torch.bool, device=device)
        self.node_ids: List[Optional[str]] = []  # index -> node_id (None = slot libre)
        self.id_to_index: Dict[str, int] = {}  # Mapping node_id -> index
        self._free: List[int] = []
        self.count = 0  # high-water mark
        self.clock = 0
        self.eviction_policy = None

    def tick(self) -> int:
        """Avance l'horloge logique d'un cran. Renvoie la nouvelle valeur."""
        self.clock += 1
        return self.clock

    @property
    def num_valid(self) -> int:
        """Nombre de nœuds vivants (les slots libres sous count ne comptent pas)."""
        if self.count == 0:
            return 0
        return int(self.valid[: self.count].sum().item())
        
    def add_node(
        self,
        node_id: str,
        data: dict,
        embedding: np.ndarray,
        sem_embedding: np.ndarray | None = None,
        dg_embedding: np.ndarray | None = None,
    ) -> str:
        """Ajoute ou met a jour un noeud (upsert par episode_id)."""
        node_id = str(node_id)

        if node_id in self.id_to_index:
            idx = self.id_to_index[node_id]
            self.node_embeddings[idx] = torch.from_numpy(embedding).to(self.device)
            if sem_embedding is not None:
                self.sem_embeddings[idx] = torch.from_numpy(sem_embedding).to(self.device)
            if dg_embedding is not None:
                self.dg_embeddings[idx] = torch.from_numpy(dg_embedding).to(self.device)
            self.timestamps[idx] = time.time()
            prev_access = self.nodes[node_id].get("access_count", 0)
            prev_data = self.nodes[node_id].get("data") or {}
            merged = dict(data)
            if "born" in prev_data and "born" not in merged:
                merged["born"] = prev_data["born"]
            self.nodes[node_id] = {
                "index": idx,
                "data": merged,
                "timestamp": time.time(),
                "access_count": prev_access,
                "consolidated": self.nodes[node_id].get("consolidated", False),
            }
            return node_id

        if self.num_valid >= self.max_nodes:
            if self.eviction_policy is not None:
                self.eviction_policy()
            else:
                self.adaptive_forgetting()
        idx = self._take_slot()
        if idx is None:
            if self.eviction_policy is not None:
                self.eviction_policy()
                idx = self._take_slot()
            if idx is None:
                self._evict_lowest()
                idx = self._take_slot()
        if idx is None:
            raise RuntimeError("graphe plein : impossible d'évincer un nœud")

        self._write_slot(idx, node_id, data, embedding, sem_embedding, dg_embedding, access=0, consolidated=False)
        return node_id

    def _write_slot(
        self,
        idx: int,
        node_id: str,
        data: dict,
        embedding: np.ndarray,
        sem_embedding: np.ndarray | None,
        dg_embedding: np.ndarray | None,
        access: float,
        consolidated: bool,
    ) -> None:
        self.node_embeddings[idx] = torch.from_numpy(np.asarray(embedding)).to(self.device)
        if sem_embedding is not None:
            self.sem_embeddings[idx] = torch.from_numpy(np.asarray(sem_embedding)).to(self.device)
        if dg_embedding is not None:
            self.dg_embeddings[idx] = torch.from_numpy(np.asarray(dg_embedding)).to(self.device)
        now = time.time()
        self.timestamps[idx] = now
        self.access_counts[idx] = access
        self.valid[idx] = True
        self.nodes[node_id] = {
            "index": idx,
            "data": data,
            "timestamp": now,
            "access_count": access,
            "consolidated": consolidated,
        }
        self.id_to_index[node_id] = idx
        if idx == len(self.node_ids):
            self.node_ids.append(node_id)
        else:
            self.node_ids[idx] = node_id

    def _take_slot(self) -> Optional[int]:
        """Réutilise un slot libre, sinon étend le high-water mark."""
        if self._free:
            return self._free.pop()
        if self.count < self.max_nodes:
            idx = self.count
            self.count += 1
            if len(self.node_ids) < self.count:
                self.node_ids.append(None)
            return idx
        return None

    def _evict_lowest(self) -> None:
        """Évince le nœud de moindre valeur (poids + accès), puis le plus ancien."""
        best_idx = None
        best_key = None
        n = self.count
        has_non_schema = False
        for i in range(n):
            if not bool(self.valid[i].item()):
                continue
            nid = self.node_ids[i]
            data = (self.nodes.get(nid) or {}).get("data") or {}
            if data.get("type") != "schema":
                has_non_schema = True
                break
        for i in range(n):
            if not bool(self.valid[i].item()):
                continue
            nid = self.node_ids[i]
            data = (self.nodes.get(nid) or {}).get("data") or {}
            if has_non_schema and data.get("type") == "schema":
                continue
            value = float(self.edges[i, :n].sum().item()) + float(self.access_counts[i].item())
            key = (value, float(self.timestamps[i].item()))
            if best_key is None or key < best_key:
                best_key = key
                best_idx = i
        if best_idx is None:
            return
        self._remove_node_by_index(best_idx)
        
    def update_hebbian_weights(self, node_id_1: str, node_id_2: str, weight: float):
        """Mise a jour hebbienne locale (sans decay global — voir apply_hebbian_decay)."""
        if node_id_1 not in self.id_to_index or node_id_2 not in self.id_to_index:
            return

        idx1, idx2 = self.id_to_index[node_id_1], self.id_to_index[node_id_2]
        if idx1 == idx2:
            return
        self.edges[idx1, idx2] += weight
        self.edges[idx2, idx1] += weight

    def link_next(self, prev_id: str, next_id: str, weight: float) -> None:
        """Lien temporel dirigé prev -> next. Ne symétrise pas."""
        if prev_id not in self.id_to_index or next_id not in self.id_to_index:
            return
        i, j = self.id_to_index[prev_id], self.id_to_index[next_id]
        if i == j or not bool(self.valid[i].item()) or not bool(self.valid[j].item()):
            return
        self.succ[i, j] += weight

    def successors(self, node_id: str, k: int = 3) -> List[dict]:
        """Successeurs temporels, poids décroissant."""
        if node_id not in self.id_to_index or self.count == 0:
            return []
        idx = self.id_to_index[node_id]
        n = self.count
        weights = self.succ[idx, :n].clone()
        weights[~self.valid[:n]] = 0
        weights[idx] = 0
        positive = int((weights > 0).sum().item())
        if positive == 0:
            return []
        vals, inds = torch.topk(weights, min(k, positive))
        out: List[dict] = []
        for value, j in zip(vals.tolist(), inds.tolist()):
            if value <= 0:
                continue
            nid = self.node_ids[j]
            if not nid or nid not in self.nodes:
                continue
            node = self.nodes[nid]
            if (node.get("data") or {}).get("type") == "schema":
                continue
            out.append({
                "id": nid,
                "weight": float(value),
                "data": node.get("data"),
                "index": node.get("index"),
                "timestamp": node.get("timestamp"),
                "access_count": node.get("access_count"),
                "consolidated": node.get("consolidated"),
            })
        return out

    def apply_hebbian_decay(self):
        """Decay periodique des poids (appele en dreaming, pas a chaque arete)."""
        n = self.count
        if n > 0:
            decay = self.config.hebbian_decay
            self.edges[:n, :n] *= decay
            self.succ[:n, :n] *= decay

    def _mask_invalid(self, scores: torch.Tensor, allowed_ids: set | None) -> torch.Tensor:
        n = self.count
        scores = scores.clone()
        scores[~self.valid[:n]] = float("-inf")
        if allowed_ids is not None:
            for i in range(n):
                nid = self.node_ids[i] if i < len(self.node_ids) else None
                if nid not in allowed_ids:
                    scores[i] = float("-inf")
        return scores

    def _pack_hits(self, scores: torch.Tensor, indices: List[int], touch: bool) -> List[dict]:
        results = []
        for idx in indices:
            if not torch.isfinite(scores[idx]):
                continue
            node_id = self.node_ids[idx] if idx < len(self.node_ids) else None
            if not node_id or node_id not in self.nodes:
                continue
            node_data = self.nodes[node_id].copy()
            node_data["id"] = node_id
            node_data["score"] = scores[idx].item()
            results.append(node_data)
            if touch:
                self.nodes[node_id]["access_count"] = self.nodes[node_id].get("access_count", 0) + 1
                self.access_counts[idx] += 1
                data = self.nodes[node_id].setdefault("data", {})
                data["last_recall"] = int(self.clock)
        return results

    def spreading_activation(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        allowed_ids: set | None = None,
        touch: bool = True,
    ) -> List[dict]:
        """Récupération par spreading activation en 2 phases."""
        if self.num_valid == 0:
            return []

        n = self.count
        query_tensor = torch.from_numpy(query_embedding).to(self.device)
        active = self.node_embeddings[:n]
        query_norm = F.normalize(query_tensor.unsqueeze(0), dim=-1)
        active_norm = F.normalize(active, dim=-1)
        base_scores = torch.matmul(query_norm, active_norm.T).squeeze(0)
        base_scores = self._mask_invalid(base_scores, allowed_ids)

        threshold = 0.2
        edge = self.edges[:n, :n]
        strong = edge * (edge > 0.05).to(edge.dtype)
        fired = (base_scores > threshold).float()
        propagated = strong.transpose(0, 1) @ fired
        if allowed_ids is not None:
            for i in range(n):
                nid = self.node_ids[i] if i < len(self.node_ids) else None
                if nid not in allowed_ids:
                    propagated[i] = 0
        boosted = base_scores + self.config.spreading_strength * propagated
        boosted = self._mask_invalid(boosted, allowed_ids)

        k_eff = min(k, self.num_valid)
        if k_eff <= 0:
            return []
        _values, top_k_indices = torch.topk(boosted, k_eff)
        return self._pack_hits(boosted, top_k_indices.tolist(), touch)

    def retrieve_semantic_boosted(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        allowed_ids: set | None = None,
        touch: bool = True,
    ) -> List[dict]:
        """
        HCANN vs RAG : top-(k-g) cosine CLIP + g voisins hebbiens injectes.
        Garantit un top-k different du RAG plat des que le graphe a des aretes.
        """
        if self.num_valid == 0:
            return []

        graph_slots = min(2, max(1, k // 3))
        sem_k = max(1, k - graph_slots)
        sem_results = self.retrieve_semantic(
            query_embedding,
            k=min(sem_k + graph_slots, self.num_valid),
            allowed_ids=allowed_ids,
            touch=touch,
        )
        if not sem_results:
            return []

        picked_ids: list[str] = []
        picked: list[dict] = []
        for node in sem_results[:sem_k]:
            picked.append(node)
            picked_ids.append(str(node["id"]))

        # Voisins 1-hop des meilleures graines semantiques
        neighbor_pool: list[tuple[float, dict]] = []
        for seed in sem_results[: min(2, len(sem_results))]:
            sid = str(seed["id"])
            if sid not in self.id_to_index:
                continue
            idx = self.id_to_index[sid]
            nbrs = torch.where(self.edges[idx, : self.count] > 0.02)[0]
            for n in nbrs.tolist():
                if not bool(self.valid[n].item()):
                    continue
                nid = self.node_ids[n]
                if not nid or nid in picked_ids:
                    continue
                if allowed_ids is not None and nid not in allowed_ids:
                    continue
                w = float(self.edges[idx, n].item())
                sem_score = next((s["score"] for s in sem_results if str(s["id"]) == nid), 0.0)
                neighbor_pool.append((w + 0.15 * sem_score, {
                    **self.nodes[nid],
                    "id": nid,
                    "score": w,
                    "via_hebbian": True,
                }))

        neighbor_pool.sort(key=lambda x: x[0], reverse=True)
        for _, node in neighbor_pool[:graph_slots]:
            picked.append(node)
            picked_ids.append(str(node["id"]))

        # Completer avec semantic si pas assez de voisins
        for node in sem_results:
            if len(picked) >= k:
                break
            if str(node["id"]) not in picked_ids:
                picked.append(node)
                picked_ids.append(str(node["id"]))

        return picked[:k]

    def retrieve_semantic(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        allowed_ids: set | None = None,
        touch: bool = True,
    ) -> List[dict]:
        """Retrieval cosinus dans l'espace semantique CLIP (requetes texte)."""
        return self._cosine_retrieve(self.sem_embeddings, query_embedding, k, allowed_ids, touch)

    def retrieve_dg(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        allowed_ids: set | None = None,
        touch: bool = True,
    ) -> List[dict]:
        """Retrieval cosinus DG (requete texte -> code sparse hippocampe)."""
        return self._cosine_retrieve(self.dg_embeddings, query_embedding, k, allowed_ids, touch)

    def _cosine_retrieve(
        self,
        matrix: torch.Tensor,
        query_embedding: np.ndarray,
        k: int,
        allowed_ids: set | None,
        touch: bool,
    ) -> List[dict]:
        if self.num_valid == 0:
            return []
        n = self.count
        query_tensor = torch.from_numpy(query_embedding).to(self.device).float()
        active = matrix[:n]
        query_norm = F.normalize(query_tensor.unsqueeze(0), dim=-1)
        active_norm = F.normalize(active, dim=-1)
        scores = self._mask_invalid(torch.matmul(query_norm, active_norm.T).squeeze(0), allowed_ids)
        k_eff = min(k, self.num_valid)
        if k_eff <= 0:
            return []
        _values, top_k_indices = torch.topk(scores, k_eff)
        return self._pack_hits(scores, top_k_indices.tolist(), touch)

    def detect_hubs(self, threshold: int = None) -> List[str]:
        """Détecte les hubs (nœuds fortement connectés)"""
        if self.num_valid == 0:
            return []
        threshold = threshold or self.config.hub_threshold
        n = self.count
        degrees = torch.sum(self.edges[:n, :n] > 0.05, dim=0)
        degrees = degrees * self.valid[:n].float()
        hub_indices = torch.where(degrees > threshold)[0].tolist()
        hubs = []
        for i in hub_indices:
            nid = self.node_ids[i] if i < len(self.node_ids) else None
            if nid:
                hubs.append(nid)
        return hubs

    def adaptive_forgetting(self):
        """Oubli adaptatif multi-critères"""
        now = time.time()
        to_remove = []

        for i in range(self.count):
            if not bool(self.valid[i].item()):
                continue
            node_id = self.node_ids[i]
            if not node_id:
                continue
            w_total = torch.sum(self.edges[i, :self.count]).item()
            age = now - self.timestamps[i].item()
            access_count = self.access_counts[i].item()

            if (w_total < self.config.prune_weight_thresh and
                age > self.config.prune_age_thresh and
                access_count == 0):
                to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            self._remove_node_by_index(i)

    def _remove_node_by_index(self, idx: int):
        """Libère un slot sans baisser le high-water mark."""
        if idx >= self.count or not bool(self.valid[idx].item()):
            return

        node_id = self.node_ids[idx]
        self.node_embeddings[idx].zero_()
        self.sem_embeddings[idx].zero_()
        self.dg_embeddings[idx].zero_()
        self.edges[idx, :].zero_()
        self.edges[:, idx].zero_()
        self.succ[idx, :].zero_()
        self.succ[:, idx].zero_()
        self.timestamps[idx] = 0
        self.access_counts[idx] = 0
        self.valid[idx] = False
        self.node_ids[idx] = None
        if node_id and node_id in self.nodes:
            del self.nodes[node_id]
        if node_id and node_id in self.id_to_index:
            del self.id_to_index[node_id]
        self._free.append(idx)

    def remove_node(self, node_id: str) -> bool:
        """Supprime un nœud nommé. Renvoie False s'il est absent."""
        node_id = str(node_id)
        if node_id not in self.id_to_index:
            return False
        self._remove_node_by_index(self.id_to_index[node_id])
        return True


class HebbianConsolidation:
    """Wrapper pour la consolidation hebbienne avec support LLM"""
    def __init__(self, config, device):
        self.graph = HebbianMemoryGraph(config, device)
        self.config = config
        self.device = device
        self.consolidation_counter = 0
        # Exposer nodes et graph pour compatibilité avec l'agent
        self.nodes = self.graph.nodes
        self.graph_dict = self.graph  # Alias pour agent.memory_graph.graph
        
    def add_node(
        self,
        node_id: str,
        data: dict,
        embedding: np.ndarray,
        sem_embedding: np.ndarray = None,
        dg_embedding: np.ndarray = None,
    ):
        """Ajoute un nœud au graphe (alias pour add_episode)"""
        self.graph.add_node(
            node_id, data, embedding, sem_embedding=sem_embedding, dg_embedding=dg_embedding
        )

    def add_episode(
        self,
        episode_id: str,
        data: dict,
        embedding: np.ndarray,
        sem_embedding: np.ndarray = None,
        dg_embedding: np.ndarray = None,
    ):
        """Ajoute un épisode et met à jour les connexions hebbiennes"""
        self.graph.add_node(
            episode_id, data, embedding, sem_embedding=sem_embedding, dg_embedding=dg_embedding
        )
        
    def update_connections(self, current_id: str, related_ids: List[str], strength: float = 0.1):
        """Renforce les connexions hebbiennes entre épisodes liés"""
        for related_id in related_ids:
            self.graph.update_hebbian_weights(current_id, related_id, strength)
            
    @property
    def num_valid(self) -> int:
        return self.graph.num_valid

    def link_next(self, prev_id: str, next_id: str, weight: float) -> None:
        self.graph.link_next(prev_id, next_id, weight)

    def successors(self, node_id: str, k: int = 3) -> List[dict]:
        return self.graph.successors(node_id, k=k)

    def retrieve(
        self, query_embedding: np.ndarray, k: int = 5, allowed_ids: set = None, touch: bool = True
    ) -> List[dict]:
        """Retrieval par spreading activation"""
        return self.graph.spreading_activation(query_embedding, k, allowed_ids=allowed_ids, touch=touch)

    def retrieve_semantic(
        self, query_embedding: np.ndarray, k: int = 5, allowed_ids: set = None, touch: bool = True
    ) -> List[dict]:
        """Retrieval cosinus sémantique (requêtes texte -> épisodes mémorisés)."""
        return self.graph.retrieve_semantic(query_embedding, k, allowed_ids=allowed_ids, touch=touch)

    def retrieve_semantic_boosted(
        self, query_embedding: np.ndarray, k: int = 5, allowed_ids: set = None, touch: bool = True
    ) -> List[dict]:
        """CLIP + spreading hebbien (HCANN vs RAG plat)."""
        return self.graph.retrieve_semantic_boosted(query_embedding, k, allowed_ids=allowed_ids, touch=touch)

    def retrieve_dg(
        self, query_embedding: np.ndarray, k: int = 5, allowed_ids: set = None, touch: bool = True
    ) -> List[dict]:
        """Retrieval cosinus DG (voie hippocampique, requete texte)."""
        return self.graph.retrieve_dg(query_embedding, k, allowed_ids=allowed_ids, touch=touch)
        
    def remove_node(self, node_id: str) -> bool:
        return self.graph.remove_node(node_id)

    def tick(self) -> int:
        return self.graph.tick()

    def consolidate(self, llm_callback=None):
        """Déclenche la consolidation des hubs"""
        hubs = self.graph.detect_hubs()
        consolidated = []
        episodic = bool(getattr(self.config, "episodic_mode", False))

        for hub_id in hubs:
            if hub_id in self.graph.nodes and not self.graph.nodes[hub_id]['consolidated']:
                hub_idx = self.graph.id_to_index[hub_id]
                neighbors = torch.where(self.graph.edges[hub_idx, :self.graph.count] > 0.05)[0].tolist()
                related_episodes = []
                for n in neighbors:
                    if n >= len(self.graph.node_ids) or not bool(self.graph.valid[n].item()):
                        continue
                    nid = self.graph.node_ids[n]
                    if nid and nid in self.graph.nodes:
                        related_episodes.append(self.graph.nodes[nid]["data"])

                summary = None
                if llm_callback and len(related_episodes) > 0:
                    summary = llm_callback(related_episodes)
                    self.graph.nodes[hub_id]['summary'] = summary

                if episodic and summary is None:
                    continue
                self.graph.nodes[hub_id]['consolidated'] = True
                consolidated.append(hub_id)

        self.graph.adaptive_forgetting()
        return consolidated
    
    def get_stats(self) -> dict:
        """Retourne des statistiques sur le graphe"""
        n = self.graph.count
        if n == 0:
            edge_count = 0
        else:
            valid = self.graph.valid[:n]
            pair = valid.unsqueeze(1) & valid.unsqueeze(0)
            edge_count = torch.sum((self.graph.edges[:n, :n] > 0.05) & pair).item()
        return {
            'total_nodes': self.graph.num_valid,
            'total_edges': edge_count,
            'hubs': len(self.graph.detect_hubs()),
            'consolidated': sum(1 for n in self.graph.nodes.values() if n['consolidated'])
        }