"""
tests/test_agent.py
Tests unitaires pour valider chaque connexion de l'agent HCANN
"""

import unittest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Imports du projet
from utils.config import HCANNConfig
from stream.agent import HCANN_Agent
from utils.entity_utils import EntityRegistry, EntityInfo
from memory.consolidation import HebbianGraph

class TestEntityRegistry(unittest.TestCase):
    """Tests pour EntityRegistry"""
    
    def setUp(self):
        self.registry = EntityRegistry(
            face_sim_threshold=0.8,
            voice_sim_threshold=0.7,
            min_observations=2
        )
        
    def test_register_new_face(self):
        """Test enregistrement d'un nouveau visage"""
        embedding = torch.randn(512)
        eid = self.registry.register_face(embedding, confidence=0.9)
        
        self.assertIn(eid, self.registry.entities)
        entity = self.registry.entities[eid]
        self.assertEqual(len(entity.faces), 1)
        self.assertEqual(entity.entity_id, eid)
        
    def test_register_matching_face(self):
        """Test reconnaissance d'un visage connu"""
        # Premier enregistrement
        embedding1 = torch.randn(512)
        embedding1 = embedding1 / embedding1.norm()  # Normaliser
        eid1 = self.registry.register_face(embedding1, confidence=0.9)
        
        # Deuxième enregistrement avec embedding similaire
        embedding2 = embedding1 + torch.randn(512) * 0.1  # Petit bruit
        embedding2 = embedding2 / embedding2.norm()
        eid2 = self.registry.register_face(embedding2, confidence=0.85)
        
        # Doit retourner le même eid
        self.assertEqual(eid1, eid2)
        self.assertEqual(len(self.registry.entities[eid1].faces), 2)
        
    def test_link_face_voice(self):
        """Test liaison face-voice"""
        face_eid = self.registry.register_face(torch.randn(512), 0.9)
        voice_eid = self.registry.register_voice(torch.randn(256), 0.9)
        
        # Liaison
        merged_eid = self.registry.link_face_voice(face_eid, voice_eid, confidence=0.95)
        
        self.assertEqual(merged_eid, face_eid)
        self.assertNotIn(voice_eid, self.registry.entities)  # Supprimé après merge
        
        entity = self.registry.entities[face_eid]
        self.assertEqual(len(entity.faces), 1)
        self.assertEqual(len(entity.voices), 1)
        
    def test_cleanup_old_entities(self):
        """Test suppression des entités anciennes"""
        # Créer des entités avec timestamps anciens
        old_entity = EntityInfo(entity_id="old_0001")
        old_entity.first_seen = 0  # Très ancien
        old_entity.add_face(torch.randn(512), 0.5)  # Faible confiance, peu d'observations
        self.registry.entities["old_0001"] = old_entity
        
        # Entité récente à conserver
        new_entity = EntityInfo(entity_id="new_0002")
        new_entity.add_face(torch.randn(512), 0.9)
        new_entity.add_face(torch.randn(512), 0.9)  # 2 observations
        self.registry.entities["new_0002"] = new_entity
        
        # Cleanup
        removed = self.registry.cleanup_old_entities(max_age_hours=1, min_observations=2)
        
        self.assertEqual(removed, 1)
        self.assertIn("new_0002", self.registry.entities)
        self.assertNotIn("old_0001", self.registry.entities)


class TestHebbianGraph(unittest.TestCase):
    """Tests pour le graphe hebbien"""
    
    def setUp(self):
        self.config = HCANNConfig()
        self.graph = HebbianGraph(self.config)
        
    def test_add_node(self):
        """Test ajout de nœud"""
        embedding = np.random.randn(self.config.hpc_size)
        node_id = self.graph.add_node(embedding, metadata={'test': True})
        
        self.assertIn(node_id, self.graph.nodes)
        self.assertEqual(self.graph.nodes[node_id]['metadata']['test'], True)
        
    def test_hebbian_update(self):
        """Test mise à jour hebbienne"""
        # Ajouter des nœuds
        n1 = self.graph.add_node(np.random.randn(self.config.hpc_size))
        n2 = self.graph.add_node(np.random.randn(self.config.hpc_size))
        
        # Mise à jour hebbienne
        self.graph.update_hebbian_weights(n1, [n2], weight=0.1)
        
        # Vérifier le poids de l'arête
        weight = self.graph.edges.get(n1, {}).get(n2, 0)
        self.assertGreater(weight, 0)
        
    def test_spreading_activation(self):
        """Test activation propagée"""
        # Créer un petit graphe en chaîne: A -> B -> C
        a = self.graph.add_node(np.random.randn(self.config.hpc_size))
        b = self.graph.add_node(np.random.randn(self.config.hpc_size))
        c = self.graph.add_node(np.random.randn(self.config.hpc_size))
        
        # Créer des arêtes
        self.graph.update_hebbian_weights(a, [b], weight=0.5)
        self.graph.update_hebbian_weights(b, [c], weight=0.5)
        
        # Query proche de A
        query = np.random.randn(self.config.hpc_size)
        # Forcer similarité avec A
        self.graph.nodes[a]['embedding'] = query
        
        results = self.graph.spreading_activation(query, top_k=3, beta=0.3)
        
        # A doit être premier, B et C devraient apparaître via propagation
        result_ids = [r['node_id'] for r in results]
        self.assertIn(a, result_ids)


class TestHCANNAgent(unittest.TestCase):
    """Tests d'intégration pour HCANN_Agent"""
    
    def setUp(self):
        self.config = HCANNConfig()
        self.config.input_dim = 128  # Réduire pour les tests
        self.config.hpc_size = 64
        
        # Mock des dépendances externes
        with patch('stream.agent.FaceAnalysis'), \
             patch('stream.agent.whisper.load_model'):
            self.agent = HCANN_Agent(self.config)
            
    def tearDown(self):
        if hasattr(self.agent, 'stop'):
            self.agent.stop()
            
    def test_agent_initialization(self):
        """Test initialisation de l'agent"""
        self.assertIsNotNone(self.agent.model)
        self.assertIsNotNone(self.agent.memory_graph)
        self.assertIsNotNone(self.agent.entity_registry)
        
    def test_process_step_mock(self):
        """Test traitement d'un step avec mocks"""
        # Mock des outils de perception
        with patch.object(self.agent, '_detect_faces', return_value=[]), \
             patch.object(self.agent, '_transcribe_audio', return_value=("", None)):
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.agent.process_step(frame, audio_data=None)
            
            # Vérifier que le buffer a été mis à jour
            self.assertGreaterEqual(len(self.agent.clip_buffer), 0)
            
    def test_query_memory_empty(self):
        """Test requête avec mémoire vide"""
        response = self.agent.query_memory("Test question")
        # Doit retourner une réponse par défaut, pas crasher
        self.assertIsInstance(response, str)
        
    def test_entity_resolution(self):
        """Test résolution d'entités cross-modales"""
        from utils.entity_utils import resolve_identity_conflicts
        
        # Créer deux entités avec même nom
        face_entity = EntityInfo(entity_id="face_001")
        face_entity.attributes['name'] = "Alice"
        face_entity.confidence = 0.9
        
        voice_entity = EntityInfo(entity_id="voice_001")
        voice_entity.attributes['name'] = "Alice"
        voice_entity.confidence = 0.85
        
        # Mock registry
        mock_registry = Mock()
        mock_registry.get_entity.side_effect = lambda eid: face_entity if eid == "face_001" else voice_entity
        mock_registry.link_face_voice = lambda f, v, c: f
        
        context = {'timestamp': time.time()}
        resolved_id, confidence = resolve_identity_conflicts(
            mock_registry, "face_001", "voice_001", context
        )
        
        self.assertEqual(resolved_id, "face_001")
        self.assertGreater(confidence, 0.6)


class TestIntegration(unittest.TestCase):
    """Tests d'intégration end-to-end"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = HCANNConfig()
        self.config.input_dim = 64
        self.config.hpc_size = 32
        
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    @patch('stream.agent.FaceAnalysis')
    @patch('stream.agent.whisper.load_model')
    def test_full_pipeline_mock(self, mock_whisper, mock_face):
        """Test pipeline complet avec mocks"""
        # Setup mocks
        mock_face.return_value.get.return_value = []
        mock_whisper.return_value.transcribe.return_value = {'text': ''}
        
        agent = HCANN_Agent(self.config)
        
        try:
            # Simuler un flux court
            for i in range(3):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                audio = np.random.randn(16000).astype(np.float32) * 0.01
                agent.process_step(frame, audio)
                
            # Tester une requête
            response = agent.query_memory("Que s'est-il passé ?")
            self.assertIsInstance(response, str)
            
        finally:
            agent.stop()


def run_tests():
    """Exécute tous les tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)