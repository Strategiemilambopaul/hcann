"""
utils/entity_utils.py
Module dédié pour la gestion robuste des entités
Clustering avancé, fusion, résolution d'identité cross-modale
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import time

@dataclass
class EntityInfo:
    """Information complète sur une entité"""
    entity_id: str
    name: Optional[str] = None
    faces: List[Dict] = field(default_factory=list)  # {embedding, timestamp, confidence}
    voices: List[Dict] = field(default_factory=list)  # {embedding, timestamp, confidence}
    attributes: Dict[str, any] = field(default_factory=dict)  # name, role, traits...
    relationships: Dict[str, str] = field(default_factory=dict)  # entity_id -> relation type
    first_seen: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    confidence: float = 1.0
    
    def add_face(self, embedding: torch.Tensor, confidence: float, metadata: Dict = None):
        """Ajoute une observation faciale"""
        self.faces.append({
            'embedding': embedding.detach().cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding,
            'confidence': confidence,
            'timestamp': time.time(),
            'metadata': metadata or {}
        })
        self.last_updated = time.time()
        
    def add_voice(self, embedding: torch.Tensor, confidence: float, metadata: Dict = None):
        """Ajoute une observation vocale"""
        self.voices.append({
            'embedding': embedding.detach().cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding,
            'confidence': confidence,
            'timestamp': time.time(),
            'metadata': metadata or {}
        })
        self.last_updated = time.time()
    
    def get_avg_face_embedding(self) -> Optional[np.ndarray]:
        """Calcule l'embedding facial moyen pondéré par la confiance"""
        if not self.faces:
            return None
        embeddings = np.stack([f['embedding'] for f in self.faces])
        weights = np.array([f['confidence'] for f in self.faces])
        weights /= weights.sum()
        return np.average(embeddings, axis=0, weights=weights)
    
    def get_avg_voice_embedding(self) -> Optional[np.ndarray]:
        """Calcule l'embedding vocal moyen pondéré par la confiance"""
        if not self.voices:
            return None
        embeddings = np.stack([v['embedding'] for v in self.voices])
        weights = np.array([v['confidence'] for v in self.voices])
        weights /= weights.sum()
        return np.average(embeddings, axis=0, weights=weights)

class EntityRegistry:
    """Registre centralisé pour la gestion des entités"""
    
    def __init__(self, 
                 face_sim_threshold: float = 0.85,
                 voice_sim_threshold: float = 0.75,
                 min_observations: int = 2,
                 clustering_method: str = "dbscan"):
        """
        Args:
            face_sim_threshold: Seuil de similarité cosinus pour les visages
            voice_sim_threshold: Seuil de similarité cosinus pour les voix
            min_observations: Nombre minimum d'observations pour valider une entité
            clustering_method: "dbscan" ou "agglomerative"
        """
        self.entities: Dict[str, EntityInfo] = {}
        self.face_sim_threshold = face_sim_threshold
        self.voice_sim_threshold = voice_sim_threshold
        self.min_observations = min_observations
        self.clustering_method = clustering_method
        self.next_id = 0
        
        # Cache pour accélération des recherches
        self._face_index: List[Tuple[str, np.ndarray]] = []
        self._voice_index: List[Tuple[str, np.ndarray]] = []
        
    def _generate_entity_id(self) -> str:
        """Génère un ID d'entité unique"""
        eid = f"entity_{self.next_id:04d}"
        self.next_id += 1
        return eid
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcule la similarité cosinus entre deux embeddings"""
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a_norm, b_norm))
    
    def _find_matching_entity_face(self, embedding: np.ndarray) -> Optional[str]:
        """Trouve une entité existante correspondant à un embedding facial"""
        if not self._face_index:
            return None
        
        best_sim = -1
        best_eid = None
        
        for eid, cached_emb in self._face_index:
            sim = self._cosine_sim(embedding, cached_emb)
            if sim > best_sim and sim >= self.face_sim_threshold:
                best_sim = sim
                best_eid = eid
                
        return best_eid
    
    def _find_matching_entity_voice(self, embedding: np.ndarray) -> Optional[str]:
        """Trouve une entité existante correspondant à un embedding vocal"""
        if not self._voice_index:
            return None
        
        best_sim = -1
        best_eid = None
        
        for eid, cached_emb in self._voice_index:
            sim = self._cosine_sim(embedding, cached_emb)
            if sim > best_sim and sim >= self.voice_sim_threshold:
                best_sim = sim
                best_eid = eid
                
        return best_eid
    
    def register_face(self, 
                     embedding: torch.Tensor, 
                     confidence: float,
                     metadata: Dict = None) -> str:
        """
        Enregistre une observation faciale et retourne l'entity_id
        """
        emb_np = embedding.detach().cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
        
        # Chercher une entité existante
        existing_eid = self._find_matching_entity_face(emb_np)
        
        if existing_eid and existing_eid in self.entities:
            # Mise à jour de l'entité existante
            entity = self.entities[existing_eid]
            entity.add_face(embedding, confidence, metadata)
            # Mise à jour du cache
            self._update_face_cache(existing_eid, entity.get_avg_face_embedding())
            return existing_eid
        else:
            # Création d'une nouvelle entité
            eid = self._generate_entity_id()
            entity = EntityInfo(entity_id=eid)
            entity.add_face(embedding, confidence, metadata)
            self.entities[eid] = entity
            self._update_face_cache(eid, emb_np)
            return eid
    
    def register_voice(self,
                      embedding: torch.Tensor,
                      confidence: float,
                      metadata: Dict = None) -> str:
        """
        Enregistre une observation vocale et retourne l'entity_id
        """
        emb_np = embedding.detach().cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
        
        existing_eid = self._find_matching_entity_voice(emb_np)
        
        if existing_eid and existing_eid in self.entities:
            entity = self.entities[existing_eid]
            entity.add_voice(embedding, confidence, metadata)
            self._update_voice_cache(existing_eid, entity.get_avg_voice_embedding())
            return existing_eid
        else:
            eid = self._generate_entity_id()
            entity = EntityInfo(entity_id=eid)
            entity.add_voice(embedding, confidence, metadata)
            self.entities[eid] = entity
            self._update_voice_cache(eid, emb_np)
            return eid
    
    def _update_face_cache(self, eid: str, avg_embedding: Optional[np.ndarray]):
        """Met à jour le cache facial"""
        # Supprimer les anciennes entrées pour cet eid
        self._face_index = [(e, emb) for e, emb in self._face_index if e != eid]
        if avg_embedding is not None:
            self._face_index.append((eid, avg_embedding))
    
    def _update_voice_cache(self, eid: str, avg_embedding: Optional[np.ndarray]):
        """Met à jour le cache vocal"""
        self._voice_index = [(e, emb) for e, emb in self._voice_index if e != eid]
        if avg_embedding is not None:
            self._voice_index.append((eid, avg_embedding))
    
    def link_face_voice(self, face_eid: str, voice_eid: str, confidence: float):
        """Lie deux entités (face et voice) en une seule"""
        if face_eid == voice_eid:
            return face_eid
            
        # Fusionner voice_eid dans face_eid
        target = self.entities.get(face_eid)
        source = self.entities.get(voice_eid)
        
        if not target or not source:
            return None
            
        # Transférer les observations vocales
        for voice_obs in source.voices:
            target.add_voice(
                torch.tensor(voice_obs['embedding']),
                voice_obs['confidence'],
                voice_obs.get('metadata')
            )
        
        # Transférer les attributs
        target.attributes.update(source.attributes)
        target.relationships.update(source.relationships)
        
        # Mettre à jour la confiance
        target.confidence = max(target.confidence, source.confidence) * confidence
        
        # Supprimer l'entité source
        del self.entities[voice_eid]
        self._voice_index = [(e, emb) for e, emb in self._voice_index if e != voice_eid]
        
        # Mettre à jour les caches
        self._update_face_cache(face_eid, target.get_avg_face_embedding())
        self._update_voice_cache(face_eid, target.get_avg_voice_embedding())
        
        return face_eid
    
    def cluster_pending_entities(self, modality: str = "face"):
        """
        Clustering batch des entités non-validées
        Utile pour la consolidation périodique
        """
        if modality == "face":
            embeddings = []
            eids = []
            for eid, entity in self.entities.items():
                avg_emb = entity.get_avg_face_embedding()
                if avg_emb is not None and len(entity.faces) < self.min_observations:
                    embeddings.append(avg_emb)
                    eids.append(eid)
                    
            if len(embeddings) < 2:
                return
                
            # Clustering
            if self.clustering_method == "dbscan":
                clustering = DBSCAN(eps=1-self.face_sim_threshold, min_samples=2, metric='cosine')
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=None, 
                    distance_threshold=1-self.face_sim_threshold,
                    affinity='cosine',
                    linkage='average'
                )
                
            labels = clustering.fit_predict(embeddings)
            
            # Fusion des clusters
            clusters = defaultdict(list)
            for eid, label in zip(eids, labels):
                if label >= 0:  # Ignorer le noise (-1)
                    clusters[label].append(eid)
                    
            for cluster_eids in clusters.values():
                if len(cluster_eids) >= 2:
                    # Fusionner tous dans le premier
                    primary = cluster_eids[0]
                    for secondary in cluster_eids[1:]:
                        self.link_face_voice(primary, secondary, confidence=0.9)
                        
        # TODO: Implémenter pour "voice" de manière similaire
    
    def list_entities(self, min_confidence: float = 0.0) -> Dict[str, Dict]:
        """Liste toutes les entités avec leurs métadonnées"""
        result = {}
        for eid, entity in self.entities.items():
            if entity.confidence >= min_confidence:
                result[eid] = {
                    'name': entity.name,
                    'face_count': len(entity.faces),
                    'voice_count': len(entity.voices),
                    'attributes': entity.attributes,
                    'first_seen': entity.first_seen,
                    'confidence': entity.confidence
                }
        return result
    
    def get_entity(self, entity_id: str) -> Optional[EntityInfo]:
        """Récupère une entité par son ID"""
        return self.entities.get(entity_id)
    
    def cleanup_old_entities(self, max_age_hours: float = 24, min_observations: int = None):
        """Supprime les entités anciennes ou peu observées"""
        min_obs = min_observations or self.min_observations
        current_time = time.time()
        to_remove = []
        
        for eid, entity in self.entities.items():
            age_hours = (current_time - entity.first_seen) / 3600
            total_obs = len(entity.faces) + len(entity.voices)
            
            if age_hours > max_age_hours and total_obs < min_obs:
                to_remove.append(eid)
                
        for eid in to_remove:
            del self.entities[eid]
            self._face_index = [(e, emb) for e, emb in self._face_index if e != eid]
            self._voice_index = [(e, emb) for e, emb in self._voice_index if e != eid]
            
        return len(to_remove)

# ─────────────────────────────────────────────────────────────
# Fonctions utilitaires avancées
# ─────────────────────────────────────────────────────────────

def merge_entities(entities: List[EntityInfo], method: str = "weighted_avg") -> EntityInfo:
    """
    Fusionne plusieurs entités en une seule
    """
    if not entities:
        return None
        
    # Entité cible: la plus récente ou la plus confiante
    target = max(entities, key=lambda e: (e.confidence, e.last_updated))
    
    for source in entities:
        if source.entity_id == target.entity_id:
            continue
            
        # Fusion des observations
        for face_obs in source.faces:
            target.add_face(
                torch.tensor(face_obs['embedding']),
                face_obs['confidence'] * 0.9,  # Pénalité pour les merges
                face_obs.get('metadata')
            )
        for voice_obs in source.voices:
            target.add_voice(
                torch.tensor(voice_obs['embedding']),
                voice_obs['confidence'] * 0.9,
                voice_obs.get('metadata')
            )
            
        # Fusion des attributs (priorité aux plus récents)
        for k, v in source.attributes.items():
            if k not in target.attributes or source.last_updated > target.attributes.get(f"{k}_updated", 0):
                target.attributes[k] = v
                target.attributes[f"{k}_updated"] = source.last_updated
                
    # Recalcul de la confiance
    target.confidence = min(1.0, sum(e.confidence for e in entities) / len(entities))
    
    return target

def resolve_identity_conflicts(registry: EntityRegistry, 
                              face_id: str, 
                              voice_id: str,
                              context: Dict = None) -> Tuple[str, float]:
    """
    Résout les conflits d'identité entre face et voice
    Retourne: (entity_id_resolved, confidence_score)
    """
    face_entity = registry.get_entity(face_id)
    voice_entity = registry.get_entity(voice_id)
    
    if not face_entity or not voice_entity:
        return None, 0.0
        
    # Si déjà liés, retour direct
    if face_id == voice_id:
        return face_id, face_entity.confidence
        
    # Score de similarité cross-modale (si embeddings disponibles)
    face_emb = face_entity.get_avg_face_embedding()
    voice_emb = voice_entity.get_avg_voice_embedding()
    
    # Heuristiques de résolution
    score = 0.0
    
    # 1. Co-occurrence temporelle
    if context and 'timestamp' in context:
        ts = context['timestamp']
        face_times = [f['timestamp'] for f in face_entity.faces]
        voice_times = [v['timestamp'] for v in voice_entity.voices]
        
        # Vérifier si observés dans la même fenêtre temporelle (±5s)
        for ft in face_times:
            for vt in voice_times:
                if abs(ft - vt) < 5.0:
                    score += 0.3
                    break
                    
    # 2. Similarité des attributs (si noms disponibles)
    if face_entity.attributes.get('name') and voice_entity.attributes.get('name'):
        if face_entity.attributes['name'].lower() == voice_entity.attributes['name'].lower():
            score += 0.4
            
    # 3. Confiance intrinsèque
    score += (face_entity.confidence + voice_entity.confidence) * 0.15
    
    # Décision
    if score >= 0.6:
        # Fusionner
        merged_id = registry.link_face_voice(face_id, voice_id, confidence=score)
        return merged_id, score
    else:
        # Garder séparé, retourner le plus confiant
        best = face_entity if face_entity.confidence >= voice_entity.confidence else voice_entity
        return best.entity_id, best.confidence