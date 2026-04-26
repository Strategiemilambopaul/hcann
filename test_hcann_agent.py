#!/usr/bin/env python3
"""
Test d'intégration complet pour l'architecture HCANN.
Simule un cycle de vie complet : Perception -> Encodage -> Stockage -> Retrieval -> Consolidation.
"""

import torch
import numpy as np
from models.hcann import HCANN
from models.cortex import MultimodalEncoder, WorkingMemory
from memory.replay_buffer import ExperienceReplayBuffer
from memory.consolidation import HebbianConsolidation
from utils.config import HCANNConfig

def test_hcann_pipeline():
    print("🚀 Initialisation de l'architecture HCANN...")
    
    # 1. Initialisation des composants
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Utilisation du dispositif: {device}")

    # Configuration
    config = HCANNConfig()
    
    # Modèle principal HCANN
    hcann_model = HCANN(config).to(device)
    
    # Mémoire
    episodic_buffer = ExperienceReplayBuffer(capacity=100)
    hebbian_graph = HebbianConsolidation(config, device)
    
    # Encodage multimodal (standalone pour le test)
    mm_encoder = MultimodalEncoder(config)

    print("   ✅ Composants initialisés.\n")

    # 2. Simulation de perceptions (Scénario : Rencontre avec Alice)
    scenarios = [
        {
            "text": "Alice sourit et dit bonjour dans le salon.",
            "emotion": "joy",
            "timestamp": "T0"
        },
        {
            "text": "Alice parle de son voyage en Italie hier.",
            "emotion": "neutral",
            "timestamp": "T-1"
        },
        {
            "text": "Bob est arrivé et a serré la main d'Alice.",
            "emotion": "neutral",
            "timestamp": "T0-bis"
        }
    ]

    print("🧠 Phase 1 : Perception et Encodage Hippocampique")
    stored_indices = []
    
    for i, scene in enumerate(scenarios):
        print(f"   --- Scène {i+1}: '{scene['text']}' ---")
        
        # Passage par HCANN (encodage rapide) - le texte est encodé en interne
        with torch.no_grad():
            hpc_state, sem_rep, wm_state = hcann_model.fast_encode(
                texts=[scene["text"]], 
                velocity=None
            )
        
        # Ajout en mémoire de travail (déjà fait dans fast_encode, mais on garde une trace)
        print(f"      État HPC généré: {hpc_state.shape}, Moyenne: {hpc_state.mean().item():.4f}")
        
        # Stockage dans le buffer épisodique
        episode_data = {
            "context": scene["text"],
            "hpc_embedding": hpc_state.cpu().squeeze(),
            "semantic": sem_rep.cpu().squeeze(),
            "emotion": scene["emotion"],
            "timestamp": scene["timestamp"]
        }
        episodic_buffer.push(
            sem_embed=sem_rep.cpu().squeeze(),
            hpc_state=hpc_state.cpu().squeeze(),
            metadata={"context": scene["text"], "emotion": scene["emotion"], "timestamp": scene["timestamp"]}
        )
        print(f"      Épisode stocké (buffer size: {len(episodic_buffer)})")
        
        # Ajout au graphe hebbien (création de nœuds et liens)
        # La méthode add_node attend: node_id, data (dict), embedding (np.ndarray)
        hebbian_graph.add_node(
            node_id=f"node_{i}",
            data={"text": scene["text"], "time": scene["timestamp"], "emotion": scene["emotion"]},
            embedding=hpc_state.cpu().squeeze().numpy()
        )
        
        # Création de liens temporels simples (i <-> i-1)
        if i > 0:
            hebbian_graph.graph.update_hebbian_weights(f"node_{i-1}", f"node_{i}", weight=0.9)
            
    print(f"\n   📊 Buffer épisodique: {len(episodic_buffer)} épisodes.")
    print(f"   🕸️ Graphe hebbien: {len(hebbian_graph.graph.nodes)} nœuds, {len(hebbian_graph.graph.edges)} arêtes.")

    # 3. Test de Retrieval (Rappel)
    print("\n🔍 Phase 2 : Retrieval par Spreading Activation")
    query_text = "Qui a parlé de voyage ?"
    
    with torch.no_grad():
        query_hpc, _, _ = hcann_model.fast_encode(texts=[query_text])
    
    # Recherche dans le graphe
    retrieved_nodes = hebbian_graph.retrieve(
        query_embedding=query_hpc.cpu().squeeze().numpy(),
        k=2
    )
    
    print(f"   Requête: '{query_text}'")
    print(f"   Résultats trouvés ({len(retrieved_nodes)}):")
    for node in retrieved_nodes:
        meta = node.get('metadata', {})
        score = node.get('activation_score', 0)
        print(f"      - [{meta.get('time', '?')}] '{meta.get('text', '')}' (Score: {score:.4f})")

    # 4. Test de Consolidation (Simulation Sommeil)
    print("\n💤 Phase 3 : Consolidation (Simulation Sommeil)")
    # On force une consolidation pour vérifier la détection de hubs
    stats = hebbian_graph.consolidate(llm_callback=None)
    
    print(f"   Statistiques de consolidation:")
    print(f"      - Hubs détectés: {len(stats) if stats else 0}")

    # 5. Vérification finale
    print("\n✅ Validation Finale")
    assert len(episodic_buffer) == 3, "Le buffer devrait contenir 3 épisodes."
    assert len(hebbian_graph.graph.nodes) == 3, "Le graphe devrait avoir 3 nœuds."
    assert len(retrieved_nodes) > 0, "Le retrieval devrait retourner des résultats."
    
    print("   🎉 Tous les tests sont PASSÉS ! L'architecture HCANN est fonctionnelle.")
    print("\n💡 Prochaines étapes suggérées:")
    print("   - Intégrer de vrais encodeurs visage/voix.")
    print("   - Connecter un LLM pour la récapitulation sémantique.")
    print("   - Implémenter la boucle temporelle réelle (veille/sommeil).")

if __name__ == "__main__":
    try:
        test_hcann_pipeline()
    except Exception as e:
        print(f"\n❌ Erreur critique: {e}")
        import traceback
        traceback.print_exc()
