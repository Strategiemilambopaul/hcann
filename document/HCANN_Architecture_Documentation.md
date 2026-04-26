# Architecture HCANN : Mémoire Épisodique Multimodale Bio-inspirée

## 🎯 Vision Globale

HCANN (Hippocampal-Cortical Adaptive Neural Network) est une architecture de mémoire épisodique pour agents conversationnels incarnés, s'inspirant des mécanismes biologiques de la mémoire humaine tout en intégrant les avancées récentes en IA (M3-Agent, MemVerse, HeLa-Mem).

### Principes Fondamentaux

1. **Dualité Hippocampe-Cortex** : Apprentissage rapide (hippocampe) vs consolidation lente (cortex)
2. **Encodage Multimodal** : Intégration vision-audio-texte dans un espace sémantique unifié
3. **Graphe Hebbien Dynamique** : Associations contextuelles auto-organisées
4. **Consolidation Hiérarchique** : Transformation des épisodes en connaissances sémantiques
5. **Retrieval par Propagation** : Activation spreading inspirée des réseaux neuronaux biologiques

---

## 🏗️ Architecture Détaillée

### 1. Flux de Traitement Principal

```
[Perception] → [Encodage Multimodal] → [Mémoire Hippocampique] → [Consolidation] → [Cortex Sémantique]
     ↓                ↓                        ↓                      ↓                  ↓
  Vision           Fusion                 Graphe Hebbien         Distillation       Connaissances
  Audio            Vector-HaSH            (épisodes)             (LLM-assisted)     (hubs)
  Texte            Entorhinal                                                     Long-term
```

### 2. Modules Clés

#### A. Module de Perception (Stream Agent)

**Rôle** : Capturer et pré-traiter les flux multimodaux en temps réel.

**Composants** :
- **Vision** : InsightFace pour détection/reconnaissance faciale
  - Embeddings normalisés (cosine similarity)
  - Suivi temporel des identités (face_id)
  - Landmarks faciaux pour expressions

- **Audio** : Whisper pour transcription + ERes2NetV2 pour reconnaissance vocale
  - Transcription multilingue
  - Embeddings vocaux (voice_id)
  - Détection d'locuteurs multiples

- **Texte** : SentenceTransformer (paraphrase-multilingual)
  - Encodage sémantique des dialogues
  - Projection vers espace HCANN

**Innovation HCANN** : Résolution d'entités unifiées (face ↔ voice ↔ entity) via associations hebbiennes temporelles.

```python
# Pseudo-code de résolution d'entités
def resolve_entities(faces, voices, temporal_window=5s):
    entities = {}
    for face in faces:
        if face.id in entity_map:
            entities[entity_map[face.id]].add(face)
        else:
            # Créer entité temporaire
            entities[new_id()] = Entity(type='face', data=face)
    
    for voice in voices:
        # Associer voix au visage le plus proche temporellement
        best_face = find_temporal_match(voice, faces, window=temporal_window)
        if best_face:
            unify(best_face.id, voice.id)
    
    return entities
```

#### B. Encodage Multimodal (HCANN Core)

**Inspiration** : Vector-HaSH + Entorhinal Gateway (VLEM/M3-Agent)

**Architecture** :
1. **Encodage Indépendant** :
   - Visuel : CNN/ViT → features spatiales (256D)
   - Textuel : Transformer → embeddings sémantiques (256D)
   - Vocal : Speaker encoder → embeddings prosodiques (256D)

2. **Fusion Entorhinale** :
   ```
   fused = Attention([visual, textual, vocal, positional])
   ```
   - Intègre la position temporelle (timestamp encoding)
   - Pondère les modalités selon la saillance (attention dynamique)

3. **Projection Hippocampique** :
   ```
   hpc_state = W_sens→hpc · fused + grid_encoding(velocity)
   ```
   - `grid_encoding` : Cellules de grille periodiques (periods=[3,5,7])
   - `velocity` : Vitesse de déplacement dans l'espace temporel

**Avantage HCANN** : Séparation claire entre encodage sensoriel (rapide) et consolidation sémantique (lente).

#### C. Mémoire Hippocampique (Graphe Hebbien)

**Inspiration** : HeLa-Mem + spatial scaffolds hippocampiques

**Structure de Données** :
```python
class HebbianMemoryGraph:
    nodes: Dict[node_id, {
        'embedding': Tensor[hpc_size],      # État hippocampique
        'data': EpisodeMetadata,             # Timestamp, entités, texte
        'timestamp': float,                  # Temps absolu
        'access_count': int,                 # Fréquence de rappel
        'consolidated': bool                 # Statut de consolidation
    }]
    
    edges: Matrix[N, N]                      # Poids hebbiens symétriques
    adjacency: SparseMatrix                  # Pour propagation efficace
```

**Mécanismes** :

1. **Ajout de Nœuds** :
   - Encodage de l'épisode courant
   - Initialisation des connexions (poids = 0)

2. **Règle de Hebb** :
   ```
   w_ij(t+1) = (1-λ)·w_ij(t) + η·I(co-activation)
   ```
   - `λ` : Taux d'oubli naturel (decay=0.995)
   - `η` : Taux d'apprentissage hebbien (lr=0.02)
   - `I` : Indicatrice de co-activation (entités partagées)

3. **Spreading Activation** :
   ```python
   def retrieve(query_embedding, k=5):
       # Phase 1: Similarité directe (cosine)
       base_scores = cosine_similarity(query, all_nodes)
       
       # Phase 2: Propagation itérative
       boosted = base_scores.clone()
       for node_i in active_nodes(base_scores > threshold):
           for neighbor_j in neighbors(node_i):
               boosted[j] += strength * edge_weight(i,j)
       
       return top_k(boosted)
   ```

4. **Détection de Hubs** :
   - Nœuds avec degré > threshold (hub_threshold=10)
   - Candidats pour consolidation sémantique

5. **Oubli Adaptatif** :
   ```
   Si (weight_total < prune_thresh) ET (age > 7 jours) ET (access_count == 0):
       → Supprimer le nœud
   ```

**Innovation HCANN** : Combinaison de retrieval sémantique (similarité cosinus) et associatif (propagation hebbienne).

#### D. Consolidation Hippocampe→Cortex

**Inspiration** : Théorie CLS (Complementary Learning Systems) + M3-Agent memory distillation

**Processus** :

1. **Trigger** : Tous les N épisodes (consolidation_freq=10) ou pendant le repos (replay)

2. **Distillation des Hubs** :
   ```python
   def consolidate_hub(hub_node):
       # Récupérer tous les épisodes connectés au hub
       related_episodes = get_neighbors(hub_node, depth=2)
       
       # Synthèse LLM (optionnel mais puissant)
       prompt = f"""
       Résume les points communs entre ces {len(related_episodes)} épisodes:
       {[ep.text for ep in related_episodes]}
       
       Extrais:
       - Les entités récurrentes
       - Les lieux mentionnés
       - Les relations sociales
       - Les événements marquants
       """
       semantic_summary = llm.generate(prompt)
       
       # Créer un attracteur cortical
       cortex.attractors.create(
           type='semantic',
           embedding=text_encoder(semantic_summary),
           source_hubs=[hub_node.id],
           confidence=len(related_episodes)
       )
   ```

3. **Transfert de Connaissance** :
   - Les hubs deviennent des "attracteurs" dans le cortex
   - Représentations stables, indépendantes du contexte
   - Accessibles via retrieval sémantique direct

4. **Replay Nocturne** (option avancée) :
   - Réactivation sélective des souvenirs récents
   - Renforcement des traces faibles
   - Intégration dans les schémas existants

**Avantage HCANN** : Le cortex stocke des connaissances générales, l'hippocampe garde les détails épisodiques.

#### E. Cortex Sémantique (Mémoire Long Terme)

**Rôle** : Stocker les connaissances consolidées sous forme d'attracteurs.

**Types d'Attracteurs** :
1. **Sémantiques** : Concepts abstraits (noms, lieux, relations)
2. **Procéduraux** : Patterns d'interaction appris
3. **Épisodiques Génériques** : Scénarios types ("prendre un café", "réunion travail")

**Structure** :
```python
class CorticalAttractor:
    embedding: Tensor[attractor_dim]      # Point fixe dans l'espace sémantique
    basin_of_attraction: float            # Rayon d'influence
    activation_function: Callable         # Dynamique de convergence
    associated_hubs: List[node_id]        # Liens vers hippocampe
```

**Dynamique** :
- Convergence itérative vers l'attracteur le plus proche
- Généralisation à partir d'exemples multiples
- Mise à jour lente (learning rate faible)

---

## 🔄 Cycle de Vie d'un Souvenir

### Phase 1 : Encodage (Temps Réel)
```
t=0s : Perception (visage + voix + texte)
t=1s : Encodage multimodal → état hippocampique
t=2s : Ajout au graphe hebbien
t=3s : Renforcement des connexions avec épisodes similaires
```

### Phase 2 : Consolidation (Périodique)
```
t=N épisodes : Détection des hubs
t=N+1 : Distillation LLM → résumé sémantique
t=N+2 : Création d'attracteur cortical
t=N+3 : Liaison hippocampe↔cortex
```

### Phase 3 : Retrieval (À la demande)
```
Query : "Qui était présent hier ?"
1. Encodage de la requête → embedding
2. Spreading activation dans le graphe
3. Activation des hubs pertinents
4. Projection vers cortex si besoin
5. Synthèse de la réponse
```

### Phase 4 : Oubli (Continu)
```
Critères :
- Faible poids hebbien total
- Âge > 7 jours
- Aucun accès récent
→ Pruning du nœud
```

---

## 📊 Comparaison avec les Travaux Existant

| Fonctionnalité | HCANN | M3-Agent | MemVerse | HeLa-Mem |
|---------------|-------|----------|----------|----------|
| **Encodage multimodal** | ✅ Vision+Audio+Texte | ✅ Vision+Texte | ✅ Multimodal | ❌ Texte seul |
| **Graphe hebbien** | ✅ Dynamique + decay | ❌ | ✅ Static graph | ✅ Dynamique |
| **Consolidation LLM** | ✅ Distillation de hubs | ✅ Memory compression | ❌ | ❌ |
| **Cellules de grille** | ✅ Vector-HaSH periods | ❌ | ❌ | ❌ |
| **Spreading activation** | ✅ 2 phases (base+propagation) | ❌ | ✅ PageRank | ✅ Simple |
| **Oubli adaptatif** | ✅ Multi-critères | ❌ | ❌ | ✅ Basé sur poids |
| **Entités unifiées** | ✅ Face↔Voice↔Entity | ❌ | ❌ | ❌ |
| **Attracteurs corticaux** | ✅ Bassins de convergence | ❌ | ❌ | ❌ |

**Positionnement HCANN** : 
- Plus biologique que M3-Agent (hippocampe/cortex séparés)
- Plus dynamique que MemVerse (graphe évolutif + oubli)
- Plus multimodal que HeLa-Mem (vision+audio+texte)
- Unique dans l'intégration Vector-HaSH + consolidation LLM

---

## 🛠️ Implémentation Actuelle

### Arborescence du Projet
```
HCANN_Memory/
├── models/
│   ├── hcann.py              # Architecture complète
│   ├── hippocampus.py        # Scaffold hippocampique (Vector-HaSH)
│   └── cortex.py             # Encodeurs + Working Memory + Attracteurs
├── memory/
│   ├── consolidation.py      # Graphe hebbien + spreading activation
│   └── replay_buffer.py      # Buffer pour consolidation différée
├── stream/
│   ├── agent.py              # Agent central (perception → mémoire)
│   ├── capture/              # Modules de capture (webcam, micro)
│   └── utils/                # Utilitaires de traitement
├── train/
│   ├── continual_learner.py  # Boucle d'apprentissage continu
│   └── metrics.py            # Accuracy, Forgetting Measure
├── utils/
│   ├── config.py             # Configuration unifiée
│   └── data_utils.py         # Dataloaders multimodaux
└── document/                 # Articles de recherche (PDF)
    ├── MemVerse.pdf
    ├── HeLa-Mem.pdf
    ├── M3-Agent.pdf
    └── HippocampalScaffolds.pdf
```

### Points Forts de l'Implémentation

1. **Modularité** : Chaque composant est testable indépendamment
2. **Extensibilité** : Ajout facile de nouvelles modalités (depth, IMU, etc.)
3. **Efficacité** : Graphes clairsemés pour propagation rapide
4. **Bio-réalisme** : Respect des principes neuroscientifiques (CLS, Hebb, grid cells)

---

## 🚀 Axes d'Amélioration Futurs

### Court Terme (1-2 semaines)
- [ ] **Intégration ERes2NetV2** pour reconnaissance vocale robuste
- [ ] **Optimisation du spreading activation** (version CUDA parallèle)
- [ ] **Tests unitaires complets** pour chaque module
- [ ] **Benchmark de retrieval** (précision/rappel sur dataset synthétique)

### Moyen Terme (1-2 mois)
- [ ] **Replay nocturne** : Réactivation offline des souvenirs récents
- [ ] **Attention cross-modale** : Mécanisme attentionnel pour fusion optimale
- [ ] **Hiérarchie de hubs** : Hubs de premier niveau → super-hubs abstraits
- [ ] **Interface de visualisation** : Graphe interactif (nodes/edges/hubs)

### Long Terme (3-6 mois)
- [ ] **Apprentissage non-supervisé des attracteurs** (auto-organisation)
- [ ] **Génération de scénarios** : Prédiction d'épisodes futurs
- [ ] **Mémoire collective** : Partage de connaissances entre agents HCANN
- [ ] **Embodiment robotique** : Intégration sur plateforme mobile

---

## 📚 Références Scientifiques

### Architecture HCANN
1. **Vector-HaSH** : Grid cells pour encodage spatio-temporel
2. **CLS Theory** : McClelland et al. (1995) - Hippocampe vs Cortex
3. **Règle de Hebb** : "Neurons that fire together, wire together"

### Inspirations Externes
4. **M3-Agent** (Seeing, Listening, Remembering, and Reasoning)
   - Mémoire multimodale hiérarchique
   - Compression par LLM
   
5. **MemVerse** (Multimodal Memory for Lifelong Learning)
   - Graphe de connaissances évolutif
   - Retrieval par PageRank modifié

6. **HeLa-Mem** (Hebbian Learning for LLM Agents)
   - Graphe hebbien dynamique
   - Oubli adaptatif basé sur les poids

7. **Hippocampal Scaffolds** (Episodic memory from spatial scaffolds)
   - Rôle des cellules de grille dans l'encodage épisodique
   - Structure géométrique de la mémoire

---

## 💡 Philosophie HCANN

> *"La mémoire n'est pas un stockage passif, mais un processus actif de reconstruction."*

HCANN incarne cette vision en combinant :
- **Biologie** : Mécanismes neuronaux validés expérimentalement
- **IA Moderne** : LLMs pour abstraction, transformers pour fusion
- **Ingénierie** : Systèmes robustes, testables et déployables

L'objectif ultime : Créer un agent capable d'**apprendre continuellement** de ses interactions, de **se souvenir contextuellement**, et d'**oublier intelligemment** — comme un humain.

---

## 📝 Guide de Démarrage Rapide

```bash
# 1. Installation
pip install -r requirements.txt

# 2. Configuration
# Éditer utils/config.py selon vos besoins

# 3. Lancer l'agent en mode stream
python stream/main.py

# 4. Tester la mémoire
python run_test.py

# 5. Visualiser les représentations
python visualize_representations.py
```

---

**Document rédigé par** : Assistant HCANN  
**Date** : Avril 2026  
**Version** : 1.0  
**Statut** : En développement actif
