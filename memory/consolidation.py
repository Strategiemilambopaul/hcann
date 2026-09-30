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
        self.node_ids: List[str] = []  # Mapping index -> node_id
        self.id_to_index: Dict[str, int] = {}  # Mapping node_id -> index
        self.count = 0
        
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
            self.nodes[node_id] = {
                "index": idx,
                "data": data,
                "timestamp": time.time(),
                "access_count": prev_access,
                "consolidated": self.nodes[node_id].get("consolidated", False),
            }
            return node_id

        if self.count >= self.max_nodes:
            self.adaptive_forgetting()

        if self.count < self.max_nodes:
            idx = self.count
            self.node_embeddings[idx] = torch.from_numpy(embedding).to(self.device)
            if sem_embedding is not None:
                self.sem_embeddings[idx] = torch.from_numpy(sem_embedding).to(self.device)
            if dg_embedding is not None:
                self.dg_embeddings[idx] = torch.from_numpy(dg_embedding).to(self.device)
            self.timestamps[idx] = time.time()
            self.access_counts[idx] = 0

            self.nodes[node_id] = {
                "index": idx,
                "data": data,
                "timestamp": time.time(),
                "access_count": 0,
                "consolidated": False,
            }
            self.node_ids.append(node_id)
            self.id_to_index[node_id] = idx
            self.count += 1
            return node_id
        return None
        
    def update_hebbian_weights(self, node_id_1: str, node_id_2: str, weight: float):
        """Mise a jour hebbienne locale (sans decay global — voir apply_hebbian_decay)."""
        if node_id_1 not in self.id_to_index or node_id_2 not in self.id_to_index:
            return

        idx1, idx2 = self.id_to_index[node_id_1], self.id_to_index[node_id_2]
        self.edges[idx1, idx2] += weight
        self.edges[idx2, idx1] += weight

    def apply_hebbian_decay(self):
        """Decay periodique des poids (appele en dreaming, pas a chaque arete)."""
        n = self.count
        if n > 0:
            self.edges[:n, :n] *= self.config.hebbian_decay
        
    def spreading_activation(
        self, query_embedding: np.ndarray, k: int = 5, allowed_ids: set | None = None
    ) -> List[dict]:
        """Récupération par spreading activation en 2 phases"""
        if self.count == 0:
            return []
        
        query_tensor = torch.from_numpy(query_embedding).to(self.device)
        active = self.node_embeddings[:self.count]
        
        # Phase 1: Similarité cosinus directe
        query_norm = F.normalize(query_tensor.unsqueeze(0), dim=-1)
        active_norm = F.normalize(active, dim=-1)
        base_scores = torch.matmul(query_norm, active_norm.T).squeeze(0)
        if allowed_ids is not None:
            for i in range(self.count):
                if self.node_ids[i] not in allowed_ids:
                    base_scores[i] = float("-inf")
        
        # Phase 2: Propagation hebbienne
        boosted = base_scores.clone()
        threshold = 0.2
        
        for i in range(self.count):
            if base_scores[i] > threshold:
                # Trouver les voisins connectés
                neighbors = torch.where(self.edges[i, :self.count] > 0.05)[0]
                for n in neighbors:
                    n_idx = int(n.item())
                    if allowed_ids is not None and self.node_ids[n_idx] not in allowed_ids:
                        continue
                    boosted[n] += self.config.spreading_strength * self.edges[i, n]
        
        # Sélectionner top-k
        top_k_values, top_k_indices = torch.topk(boosted, min(k, self.count))
        
        results = []
        for idx in top_k_indices.tolist():
            if boosted[idx].item() == float("-inf"):
                continue
            node_id = self.node_ids[idx]
            node_data = self.nodes[node_id].copy()
            node_data['id'] = node_id
            node_data['score'] = boosted[idx].item()
            results.append(node_data)
            self.nodes[node_id]['access_count'] += 1
            self.access_counts[idx] += 1
        
        return results

    def retrieve_semantic_boosted(
        self, query_embedding: np.ndarray, k: int = 5, allowed_ids: set | None = None
    ) -> List[dict]:
        """
        HCANN vs RAG : top-(k-g) cosine CLIP + g voisins hebbiens injectes.
        Garantit un top-k different du RAG plat des que le graphe a des aretes.
        """
        if self.count == 0:
            return []

        graph_slots = min(2, max(1, k // 3))
        sem_k = max(1, k - graph_slots)
        sem_results = self.retrieve_semantic(query_embedding, k=min(sem_k + graph_slots, self.count), allowed_ids=allowed_ids)
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
                nid = self.node_ids[n]
                if nid in picked_ids:
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
        self, query_embedding: np.ndarray, k: int = 5, allowed_ids: set | None = None
    ) -> List[dict]:
        """Retrieval cosinus dans l'espace semantique CLIP (requetes texte)."""
        if self.count == 0:
            return []

        query_tensor = torch.from_numpy(query_embedding).to(self.device).float()
        active = self.sem_embeddings[: self.count]
        query_norm = F.normalize(query_tensor.unsqueeze(0), dim=-1)
        active_norm = F.normalize(active, dim=-1)
        scores = torch.matmul(query_norm, active_norm.T).squeeze(0)

        if allowed_ids is not None:
            for i in range(self.count):
                if self.node_ids[i] not in allowed_ids:
                    scores[i] = float("-inf")

        top_k_values, top_k_indices = torch.topk(scores, min(k, self.count))
        results = []
        for idx in top_k_indices.tolist():
            if scores[idx].item() == float("-inf"):
                continue
            node_id = self.node_ids[idx]
            node_data = self.nodes[node_id].copy()
            node_data["id"] = node_id
            node_data["score"] = scores[idx].item()
            results.append(node_data)
        return results

    def retrieve_dg(self, query_embedding: np.ndarray, k: int = 5, allowed_ids: set | None = None) -> List[dict]:
        """Retrieval cosinus DG (requete texte -> code sparse hippocampe)."""
        if self.count == 0:
            return []

        query_tensor = torch.from_numpy(query_embedding).to(self.device).float()
        active = self.dg_embeddings[: self.count]
        query_norm = F.normalize(query_tensor.unsqueeze(0), dim=-1)
        active_norm = F.normalize(active, dim=-1)
        scores = torch.matmul(query_norm, active_norm.T).squeeze(0)

        if allowed_ids is not None:
            for i in range(self.count):
                if self.node_ids[i] not in allowed_ids:
                    scores[i] = float("-inf")

        top_k_values, top_k_indices = torch.topk(scores, min(k, self.count))
        results = []
        for idx in top_k_indices.tolist():
            if scores[idx].item() == float("-inf"):
                continue
            node_id = self.node_ids[idx]
            node_data = self.nodes[node_id].copy()
            node_data["id"] = node_id
            node_data["score"] = scores[idx].item()
            results.append(node_data)
        return results
        
    def detect_hubs(self, threshold: int = None) -> List[str]:
        """Détecte les hubs (nœuds fortement connectés)"""
        threshold = threshold or self.config.hub_threshold
        degrees = torch.sum(self.edges[:self.count, :self.count] > 0.05, dim=0)
        hub_indices = torch.where(degrees > threshold)[0].tolist()
        return [self.node_ids[i] for i in hub_indices if i < len(self.node_ids)]
        
    def adaptive_forgetting(self):
        """Oubli adaptatif multi-critères"""
        now = time.time()
        to_remove = []
        
        for i in range(self.count):
            node_id = self.node_ids[i]
            w_total = torch.sum(self.edges[i, :self.count]).item()
            age = now - self.timestamps[i].item()
            access_count = self.access_counts[i].item()
            
            # Critères d'oubli
            if (w_total < self.config.prune_weight_thresh and 
                age > self.config.prune_age_thresh and 
                access_count == 0):
                to_remove.append(i)
        
        # Suppression en ordre inverse pour préserver les indices
        for i in sorted(to_remove, reverse=True):
            self._remove_node_by_index(i)
    
    def _remove_node_by_index(self, idx: int):
        """Supprime un nœud par son indice"""
        if idx >= self.count:
            return
        
        node_id = self.node_ids[idx]
        
        # Effacer les connexions
        self.node_embeddings[idx].zero_()
        self.edges[idx, :].zero_()
        self.edges[:, idx].zero_()
        
        # Marquer comme supprimé
        del self.nodes[node_id]
        self.node_ids[idx] = None
        del self.id_to_index[node_id]


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
            
    def retrieve(
        self, query_embedding: np.ndarray, k: int = 5, allowed_ids: set = None
    ) -> List[dict]:
        """Retrieval par spreading activation"""
        return self.graph.spreading_activation(query_embedding, k, allowed_ids=allowed_ids)

    def retrieve_semantic(self, query_embedding: np.ndarray, k: int = 5, allowed_ids: set = None) -> List[dict]:
        """Retrieval cosinus sémantique (requêtes texte -> épisodes mémorisés)."""
        return self.graph.retrieve_semantic(query_embedding, k, allowed_ids=allowed_ids)

    def retrieve_semantic_boosted(
        self, query_embedding: np.ndarray, k: int = 5, allowed_ids: set = None
    ) -> List[dict]:
        """CLIP + spreading hebbien (HCANN vs RAG plat)."""
        return self.graph.retrieve_semantic_boosted(query_embedding, k, allowed_ids=allowed_ids)

    def retrieve_dg(self, query_embedding: np.ndarray, k: int = 5, allowed_ids: set = None) -> List[dict]:
        """Retrieval cosinus DG (voie hippocampique, requete texte)."""
        return self.graph.retrieve_dg(query_embedding, k, allowed_ids=allowed_ids)
        
    def consolidate(self, llm_callback=None):
        """Déclenche la consolidation des hubs"""
        hubs = self.graph.detect_hubs()
        consolidated = []
        
        for hub_id in hubs:
            if hub_id in self.graph.nodes and not self.graph.nodes[hub_id]['consolidated']:
                # Récupérer les épisodes connectés
                hub_idx = self.graph.id_to_index[hub_id]
                neighbors = torch.where(self.graph.edges[hub_idx, :self.graph.count] > 0.05)[0].tolist()
                related_episodes = [
                    self.graph.nodes[self.graph.node_ids[n]]['data']
                    for n in neighbors if n < len(self.graph.node_ids)
                ]
                
                # Distillation via LLM si callback fourni
                if llm_callback and len(related_episodes) > 0:
                    summary = llm_callback(related_episodes)
                    self.graph.nodes[hub_id]['summary'] = summary
                
                self.graph.nodes[hub_id]['consolidated'] = True
                consolidated.append(hub_id)
        
        # Oubli adaptatif
        self.graph.adaptive_forgetting()
        
        return consolidated
    
    def get_stats(self) -> dict:
        """Retourne des statistiques sur le graphe"""
        return {
            'total_nodes': self.graph.count,
            'total_edges': torch.sum(self.graph.edges[:self.graph.count, :self.graph.count] > 0.05).item(),
            'hubs': len(self.graph.detect_hubs()),
            'consolidated': sum(1 for n in self.graph.nodes.values() if n['consolidated'])
        }