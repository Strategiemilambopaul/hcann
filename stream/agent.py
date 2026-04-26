import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time
import threading
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity

# Importation des modules HCANN existants
from models.hcann import HCANN
from memory.consolidation import HebbianConsolidation
from utils.config import HCANNConfig

# Importations pour la vision et l'audio
try:
    import whisper
    from insightface.app import FaceAnalysis
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("⚠️ Libraries manquantes. Certaines fonctionnalités seront simulées.")

class HCANN_Agent:
    """
    L'Agent Central HCANN - Version Complète
    Connecte perception → encodage → mémoire → retrieval
    """
    
    def __init__(self, config: HCANNConfig):
        print("🧠 Initialisation de l'Agent HCANN...")
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ========== 1. MODÈLE HCANN (Le Cerveau) ==========
        self.model = HCANN(config).to(self.device)
        self.model.eval()
        
        # ========== 2. MÉMOIRE HEBBIENNE (Graphe Dynamique) ==========
        self.memory_graph = HebbianConsolidation(config, self.device)
        
        # ========== 3. OUTILS DE PERCEPTION ==========
        self._init_perception_tools()
        
        # ========== 4. GESTION DES ENTITÉS ==========
        self.entities = {
            'faces': {},           # face_id -> {'embedding': tensor, 'count': int}
            'voices': {},          # voice_id -> {'embedding': tensor, 'count': int}
            'associations': defaultdict(lambda: defaultdict(float)),  # face_id -> {voice_id: weight}
            'entity_map': {}       # {face_id or voice_id} -> unified_entity_id
        }
        self.next_face_id = 0
        self.next_voice_id = 0
        self.next_entity_id = 0
        
        # ========== 5. BUFFER TEMPORIEL ==========
        self.temporal_buffer = deque(maxlen=30)  # 30 secondes de contexte
        
        print("✅ Agent HCANN prêt. Mode Stream actif.")

    def _init_perception_tools(self):
        """Initialise les outils de perception (Vision + Audio + Texte)"""
        # --- Vision : InsightFace ---
        try:
            self.face_app = FaceAnalysis(
                name='buffalo_l', 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_app.prepare(ctx_id=0 if self.device.type == 'cuda' else -1, det_size=(640, 640))
            print("👤 InsightFace chargé.")
        except Exception as e:
            print(f"❌ InsightFace: {e}")
            self.face_app = None
        
        # --- Audio : Whisper ---
        try:
            self.whisper_model = whisper.load_model("base", device=self.device)
            print("🎙️ Whisper chargé.")
        except Exception as e:
            print(f"❌ Whisper: {e}")
            self.whisper_model = None
        
        # --- Texte : Sentence Transformer pour embeddings ---
        try:
            self.text_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.text_encoder.to(self.device)
            print("📝 Text encoder chargé.")
        except Exception as e:
            print(f"❌ Text encoder: {e}")
            self.text_encoder = None

    # ========== MÉTHODE PRINCIPALE DE TRAITEMENT ==========
    
    def process_step(self, video_frame: np.ndarray, audio_data: np.ndarray = None, timestamp: float = None):
        """
        Traite un pas de temps : perception → encodage → mémoire
        """
        timestamp = timestamp or time.time()
        
        # --- 1. PERCEPTION VISUELLE ---
        face_results = self._detect_and_encode_faces(video_frame)
        
        # --- 2. PERCEPTION AUDIO ---
        text_content, voice_embedding = "", None
        if audio_data is not None and self.whisper_model is not None:
            text_content, voice_embedding = self._transcribe_and_encode_audio(audio_data)
        
        # --- 3. GESTION DES ENTITÉS (Face ↔ Voice Association) ---
        unified_entities = self._resolve_entities(face_results, voice_embedding, timestamp)
        
        # --- 4. ENCODAGE MULTIMODAL HCANN ---
        hpc_embedding = self._encode_multimodal(
            frame=video_frame,
            text=text_content,
            faces=face_results,
            voice=voice_embedding,
            entities=unified_entities
        )
        
        # --- 5. STOCKAGE DANS LA MÉMOIRE HEBBIENNE ---
        episode_node = {
            'timestamp': timestamp,
            'entities': list(unified_entities.values()),
            'faces': [f['id'] for f in face_results],
            'voice': voice_embedding is not None,
            'text': text_content,
            'raw_frame_mean': video_frame.mean(axis=(0,1)).tolist()  # Pour debug
        }
        
        self._store_episode(hpc_embedding, episode_node)
        
        # --- 6. MISE À JOUR TEMPORELLE ---
        self.temporal_buffer.append({
            'timestamp': timestamp,
            'entities': unified_entities,
            'hpc_state': hpc_embedding.detach().cpu()
        })

    # ========== PERCEPTION VISUELLE ==========
    
    def _detect_and_encode_faces(self, frame: np.ndarray) -> list:
        """Détecte les visages et retourne leurs IDs + embeddings"""
        if self.face_app is None:
            return []
        
        faces = self.face_app.get(frame)
        results = []
        
        for face in faces:
            if face.det_score < 0.7:  # Filtre de qualité
                continue
            
            embedding = torch.from_numpy(face.embedding).float().to(self.device)
            embedding = F.normalize(embedding, dim=0)  # Normalisation cosinus
            
            # Trouver ou créer un face_id
            face_id = self._get_or_create_face_id(embedding)
            
            results.append({
                'id': face_id,
                'embedding': embedding,
                'bbox': face.bbox.tolist(),
                'landmarks': face.landmark_2d_106.tolist() if hasattr(face, 'landmark_2d_106') else None
            })
        
        return results
    
    def _get_or_create_face_id(self, embedding: torch.Tensor, threshold: float = 0.85) -> str:
        """Retourne un face_id existant ou en crée un nouveau"""
        if not self.entities['faces']:
            return self._create_new_face_id(embedding)
        
        # Comparaison cosinus avec les faces connues
        best_sim = -1
        best_id = None
        
        for face_id, data in self.entities['faces'].items():
            sim = F.cosine_similarity(embedding, data['embedding'], dim=0).item()
            if sim > best_sim:
                best_sim = sim
                best_id = face_id
        
        if best_sim >= threshold:
            # Mise à jour de l'embedding moyen (moving average)
            self.entities['faces'][best_id]['embedding'] = (
                0.9 * self.entities['faces'][best_id]['embedding'] + 
                0.1 * embedding
            )
            self.entities['faces'][best_id]['count'] += 1
            return best_id
        else:
            return self._create_new_face_id(embedding)
    
    def _create_new_face_id(self, embedding: torch.Tensor) -> str:
        """Crée un nouveau face_id"""
        face_id = f"face_{self.next_face_id}"
        self.entities['faces'][face_id] = {
            'embedding': embedding.clone(),
            'count': 1,
            'first_seen': time.time()
        }
        self.next_face_id += 1
        return face_id

    # ========== PERCEPTION AUDIO ==========
    
    def _transcribe_and_encode_audio(self, audio_chunk: np.ndarray) -> tuple[str, torch.Tensor]:
        """Transcrit l'audio et encode l'embedding vocal"""
        if self.whisper_model is None:
            return "", None
        
        try:
            # Transcription
            result = self.whisper_model.transcribe(
                audio_chunk, 
                language='fr',  # ou 'en' selon besoin
                fp16=(self.device.type == 'cuda')
            )
            text = result['text'].strip()
            
            # Encodage vocal (placeholder - à remplacer par ERes2NetV2 en prod)
            # Ici on utilise un hash simple du texte comme proxy
            voice_embedding = self._text_to_embedding(text) if text else None
            
            return text, voice_embedding
            
        except Exception as e:
            print(f"❌ Transcription: {e}")
            return "", None

    # ========== GESTION DES ENTITÉS ==========
    
    def _resolve_entities(self, faces: list, voice_emb: torch.Tensor, timestamp: float) -> dict:
        """
        Résout les identités : lie faces ↔ voices ↔ unified entities
        Retourne: {entity_id: {'type': 'face'|'voice'|'unified', 'id': str, ...}}
        """
        unified = {}
        
        # --- Cas 1: Visages détectés ---
        for face in faces:
            face_id = face['id']
            
            # Vérifier si ce visage a déjà une entité unifiée
            if face_id in self.entities['entity_map']:
                entity_id = self.entities['entity_map'][face_id]
                unified[entity_id] = {'type': 'unified', 'face_id': face_id, 'voice_id': None}
            else:
                # Nouvelle entité temporaire (juste visage)
                unified[face_id] = {'type': 'face', 'id': face_id}
        
        # --- Cas 2: Voix détectée ---
        if voice_emb is not None:
            voice_id = self._get_or_create_voice_id(voice_emb)
            
            # Chercher une association avec un visage présent
            associated_face = None
            if faces:
                # Heuristique simple: prendre le premier visage non-associé
                for face in faces:
                    if face['id'] not in self.entities['entity_map']:
                        associated_face = face['id']
                        break
            
            if associated_face:
                # Créer une entité unifiée
                entity_id = f"entity_{self.next_entity_id}"
                self.next_entity_id += 1
                
                # Mapper face et voice vers cette entité
                self.entities['entity_map'][associated_face] = entity_id
                self.entities['entity_map'][voice_id] = entity_id
                
                # Renforcer l'association hebbienne
                self.entities['associations'][associated_face][voice_id] += 0.1
                
                unified[entity_id] = {
                    'type': 'unified',
                    'face_id': associated_face,
                    'voice_id': voice_id
                }
            else:
                # Entité juste voix
                if voice_id not in self.entities['entity_map']:
                    unified[voice_id] = {'type': 'voice', 'id': voice_id}
                else:
                    entity_id = self.entities['entity_map'][voice_id]
                    unified[entity_id] = {'type': 'unified', 'face_id': None, 'voice_id': voice_id}
        
        return unified

    def _get_or_create_voice_id(self, embedding: torch.Tensor, threshold: float = 0.8) -> str:
        """Similaire à _get_or_create_face_id mais pour les voix"""
        if not self.entities['voices']:
            return self._create_new_voice_id(embedding)
        
        best_sim = -1
        best_id = None
        
        for voice_id, data in self.entities['voices'].items():
            sim = F.cosine_similarity(embedding, data['embedding'], dim=0).item()
            if sim > best_sim:
                best_sim = sim
                best_id = voice_id
        
        if best_sim >= threshold:
            self.entities['voices'][best_id]['embedding'] = (
                0.9 * self.entities['voices'][best_id]['embedding'] + 
                0.1 * embedding
            )
            self.entities['voices'][best_id]['count'] += 1
            return best_id
        else:
            return self._create_new_voice_id(embedding)
    
    def _create_new_voice_id(self, embedding: torch.Tensor) -> str:
        voice_id = f"voice_{self.next_voice_id}"
        self.entities['voices'][voice_id] = {
            'embedding': embedding.clone(),
            'count': 1,
            'first_seen': time.time()
        }
        self.next_voice_id += 1
        return voice_id

    # ========== ENCODAGE MULTIMODAL HCANN ==========
    
    def _encode_multimodal(self, frame: np.ndarray, text: str, faces: list, 
                          voice: torch.Tensor, entities: dict) -> torch.Tensor:
        """
        Encode les inputs multimodaux en état hippocampique via HCANN
        """
        # --- 1. Encodage visuel (via le cortex du modèle) ---
        # Convertir frame en tensor et normaliser
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        if frame_tensor.shape[0] == 3:  # RGB
            frame_tensor = frame_tensor.to(self.device)
            
            # Passage par le cortex pour features visuelles
            with torch.no_grad():
                # Note: adapter selon l'interface réelle de ton CortexModule
                if hasattr(self.model, 'cortex'):
                    visual_features = self.model.cortex(frame_tensor.unsqueeze(0))
                    if isinstance(visual_features, tuple):
                        visual_features = visual_features[0]  # Prendre logits ou features
                else:
                    # Fallback: pooling simple
                    visual_features = F.adaptive_avg_pool2d(frame_tensor, (1, 1)).flatten()
        else:
            visual_features = torch.zeros(self.config.hpc_size, device=self.device)
        
        # --- 2. Encodage textuel ---
        text_embedding = self._text_to_embedding(text) if text else torch.zeros(self.config.hpc_size, device=self.device)
        
        # --- 3. Fusion multimodale (simple averaging pour l'exemple) ---
        # Dans une version avancée: utiliser l'EntorhinalGateway de VLEM
        fused_embedding = (visual_features + text_embedding) / 2
        fused_embedding = F.normalize(fused_embedding, dim=0)
        
        # --- 4. Passage par l'hippocampe (Vector-HaSH) ---
        with torch.no_grad():
            # Velocity = 0 pour l'instant (pas de mouvement temporel explicite)
            velocity = torch.zeros(1, 2, device=self.device)
            hpc_state, _ = self.model.hippocampus(fused_embedding.unsqueeze(0), velocity=velocity)
        
        return hpc_state.squeeze(0)  # Retourne [hpc_size]

    def _text_to_embedding(self, text: str) -> torch.Tensor:
        """Encode du texte en embedding via SentenceTransformer"""
        if self.text_encoder is None or not text:
            return torch.zeros(self.config.hpc_size, device=self.device)
        
        with torch.no_grad():
            emb = self.text_encoder.encode(text, convert_to_tensor=True, device=self.device)
            # Project to hpc_size if needed
            if emb.shape[-1] != self.config.hpc_size:
                # Simple projection linéaire (à remplacer par une vraie couche)
                proj = torch.nn.Linear(emb.shape[-1], self.config.hpc_size).to(self.device)
                emb = proj(emb)
            return F.normalize(emb, dim=0)

    # ========== STOCKAGE MÉMOIRE ==========
    
    def _store_episode(self, hpc_embedding: torch.Tensor, episode_data: dict):
        """Ajoute un épisode au graphe hebbien avec mises à jour associatives"""
        # --- 1. Ajouter le nœud ---
        node_id = f"ep_{int(episode_data['timestamp'] * 1000)}"
        self.memory_graph.add_node(
            node_id=node_id,
            data=episode_data,
            embedding=hpc_embedding.detach().cpu().numpy()
        )
        
        # --- 2. Mise à jour hebbienne avec épisodes récents ---
        # Renforcer les connexions avec les épisodes contenant les mêmes entités
        recent_nodes = list(self.memory_graph.nodes.keys())[-10:]  # 10 derniers
        
        for recent_id in recent_nodes:
            if recent_id == node_id:
                continue
            
            recent_data = self.memory_graph.nodes[recent_id]['data']
            
            # Calculer similarité d'entités
            entity_overlap = len(set(episode_data['entities']) & set(recent_data['entities']))
            if entity_overlap > 0:
                # Renforcement hebbien proportionnel au chevauchement
                weight = 0.1 * entity_overlap
                self.memory_graph.update_hebbian_weights(node_id, recent_id, weight)
        
        # --- 3. Consolidation périodique (tous les N épisodes) ---
        if len(self.memory_graph.nodes) % self.config.consolidation_freq == 0:
            self._trigger_consolidation()

    def _trigger_consolidation(self):
        """Déclenche la consolidation hebbienne (distillation des hubs)"""
        print("🔄 Consolidation hebbienne...")
        
        # Détecter les hubs (nœuds fortement connectés)
        hubs = self.memory_graph.detect_hubs(threshold=5)
        
        for hub_id in hubs:
            # Ici: appeler un LLM pour distiller le hub en connaissance sémantique
            # Pour l'instant, on marque juste le nœud comme "consolidé"
            self.memory_graph.nodes[hub_id]['consolidated'] = True
            print(f"  ✓ Hub consolidé: {hub_id}")
        
        # Oubli adaptatif
        self.memory_graph.adaptive_forgetting()

    # ========== RETRIEVAL & RÉPONSE ==========
    
    def query_memory(self, query_text: str, top_k: int = 5) -> str:
        """
        Pose une question et récupère la réponse via la mémoire hebbienne
        """
        print(f"\n🔍 Question: {query_text}")
        
        # --- 1. Encoder la requête ---
        query_embedding = self._text_to_embedding(query_text)
        
        # --- 2. Retrieval via Spreading Activation ---
        retrieved = self.memory_graph.spreading_activation(
            query_embedding.detach().cpu().numpy(), 
            top_k=top_k
        )
        
        if not retrieved:
            return "🤖 Je n'ai pas encore de souvenirs pertinents."
        
        # --- 3. Synthèse de la réponse ---
        response = f"🧠 {len(retrieved)} souvenir(s) activé(s) :\n\n"
        
        for i, node in enumerate(retrieved, 1):
            data = node['data']
            ts = time.strftime("%H:%M:%S", time.localtime(data['timestamp']))
            entities = ", ".join(data.get('entities', []))
            text_preview = data.get('text', "—")[:80] + ("..." if len(data.get('text', "")) > 80 else "")
            
            response += f"{i}. [{ts}] Entités: {{{entities}}}\n"
            response += f"   🗣️ \"{text_preview}\"\n"
            if data.get('faces'):
                response += f"   👤 Visages: {', '.join(data['faces'])}\n"
            response += "\n"
        
        return response

    # ========== UTILITAIRES ==========
    
    def get_entity_info(self, entity_id: str) -> dict:
        """Retourne les informations sur une entité"""
        info = {'entity_id': entity_id, 'faces': [], 'voices': [], 'episodes': []}
        
        # Chercher dans les épisodes
        for node_id, node in self.memory_graph.nodes.items():
            if entity_id in node['data'].get('entities', []):
                info['episodes'].append({
                    'timestamp': node['data']['timestamp'],
                    'text': node['data'].get('text', '')
                })
        
        return info

    def cleanup(self):
        """Nettoyage avant fermeture"""
        print("🧹 Nettoyage de l'agent...")
        if hasattr(self, 'face_app') and self.face_app:
            # InsightFace n'a pas de méthode cleanup explicite
            pass
        if hasattr(self, 'whisper_model') and self.whisper_model:
            del self.whisper_model
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None


# ========== POINT D'ENTRÉE POUR TEST ==========

if __name__ == "__main__":
    from utils.config import HCANNConfig
    
    config = HCANNConfig()
    agent = HCANN_Agent(config)
    
    print("\n🎬 Simulation de flux vidéo/audio...")
    
    try:
        for i in range(10):
            # Frame simulée (couleur aléatoire)
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Audio simulé (silence ou texte aléatoire)
            if i % 3 == 0:
                dummy_audio = np.zeros(16000, dtype=np.float32)  # Silence
            else:
                dummy_audio = np.random.randn(16000).astype(np.float32) * 0.01  # Bruit faible
            
            agent.process_step(dummy_frame, dummy_audio)
            time.sleep(0.3)  # ~3 FPS pour le test
            
        # Test de requête
        print("\n" + "="*50)
        response = agent.query_memory("Qui était présent ?")
        print(response)
        
    finally:
        agent.cleanup()