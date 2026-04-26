# HCANN Memory - Architecture de Mémoire Épisodique Bio-inspirée

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Vue d'Ensemble

**HCANN** (Hippocampal-Cortical Adaptive Neural Network) est une architecture de mémoire épisodique multimodale pour agents conversationnels incarnés, inspirée des mécanismes biologiques de la mémoire humaine et intégrant les avancées récentes en IA (M3-Agent, MemVerse, HeLa-Mem).

### ✨ Caractéristiques Principales

- **🧠 Dualité Hippocampe-Cortex** : Apprentissage rapide (hippocampe) vs consolidation lente (cortex)
- **👁️ Encodage Multimodal** : Intégration vision-audio-texte dans un espace sémantique unifié
- **🕸️ Graphe Hebbien Dynamique** : Associations contextuelles auto-organisées avec oubli adaptatif
- **🔄 Consolidation Hiérarchique** : Transformation des épisodes en connaissances sémantiques via LLM
- **🔍 Retrieval par Propagation** : Activation spreading inspirée des réseaux neuronaux biologiques

---

## 📚 Documentation Complète

Pour une description détaillée de l'architecture, consultez :
- **[Documentation Architecture HCANN](document/HCANN_Architecture_Documentation.md)** - Guide complet avec schémas, comparaisons et références scientifiques

---

## 🏗️ Architecture du Projet

```
HCANN_Memory/
├── models/
│   ├── __init__.py
│   ├── hippocampus.py          # Module hippocampique (Vector-HaSH, cellules de grille)
│   ├── cortex.py               # Module cortical (encodeurs, working memory, attracteurs)
│   └── hcann.py                # Architecture complète HCANN
├── memory/
│   ├── __init__.py
│   ├── replay_buffer.py        # Buffer d'expérience pour consolidation différée
│   └── consolidation.py        # Graphe hebbien + spreading activation + oubli
├── train/
│   ├── __init__.py
│   ├── continual_learner.py    # Boucle d'apprentissage continu
│   └── metrics.py              # Métriques (Accuracy, Forgetting Measure)
├── utils/
│   ├── __init__.py
│   ├── data_utils.py           # Dataloaders incrémentaux multimodaux
│   └── config.py               # Configuration unifiée
├── stream/
│   ├── __init__.py
│   ├── agent.py                # Agent central (perception → encodage → mémoire)
│   ├── main.py                 # Point d'entrée mode stream
│   ├── capture/                # Modules de capture (webcam, microphone)
│   ├── tests/                  # Tests unitaires
│   └── utils/                  # Utilitaires de traitement
├── document/                   # Articles de recherche & documentation
│   ├── HCANN_Architecture_Documentation.md
│   ├── MemVerse_Multimodal Memory for Lifelong Learning Agents.pdf
│   ├── HeLa-Mem Hebbian Learning and Associative Memory for LLM Agents.pdf
│   ├── Seeing, Listening, Remembering, and Reasoning.pdf (M3-Agent)
│   └── Episodic and associative memory from spatial scaffolds in the hippocampus.pdf
├── main.py                     # Point d'entrée principal
├── run_test.py                 # Script de test rapide
├── visualize_representations.py # Visualisation des représentations
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

---

## 🚀 Démarrage Rapide

### 1. Installation des Dépendances

```bash
# Environnement virtuel recommandé
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installation
pip install -r requirements.txt

# Pour la visualisation (optionnel)
pip install -r requirements_visualization.txt
```

### 2. Préparation des Données (Optionnel)

```bash
# Générer des données synthétiques pour tester
python generate_synthetic_multimodal_data.py
```

Structure attendue des données :
```
data/multimodal_episodes/
├── episodes.json           # Métadonnées des épisodes
└── images/                 # Images associées
```

### 3. Lancer l'Entraînement

```bash
python main.py
```

### 4. Mode Stream (Agent Temps Réel)

```bash
# Lancer l'agent avec capture webcam/micro
python stream/main.py

# Ou tester avec des données simulées
python stream/test_stream.py
```

### 5. Exécuter les Tests

```bash
# Test rapide
python run_test.py

# Benchmark complet
python benchmark.py
```

### 6. Visualiser les Représentations

```bash
python visualize_representations.py
```

---

## 🔬 Composants Clés

### A. Modèle HCANN (`models/hcann.py`)

Architecture neuro-inspirée combinant :
- **Hippocampe** : Encodage rapide avec cellules de grille (Vector-HaSH)
- **Cortex** : Encodeurs multimodaux et mémoire de travail
- **Porte Entorhinale** : Interface entre hippocampe et cortex
- **Associations Hétéro-modales** : Liaisons bidirectionnelles

```python
from models.hcann import HCANN
from utils.config import HCANNConfig

config = HCANNConfig()
model = HCANN(config)

# Encodage rapide
hpc_state, semantic, wm = model.fast_encode(images=img, texts=text)

# Consolidation lente
loss = model.slow_consolidate(hpc_target, sem_input)
```

### B. Graphe Hebbien (`memory/consolidation.py`)

Mémoire épisodique dynamique avec :
- **Règle de Hebb** : Renforcement des connexions co-activées
- **Spreading Activation** : Retrieval par propagation dans le graphe
- **Détection de Hubs** : Identification des nœuds centraux
- **Oubli Adaptatif** : Pruning basé sur âge, poids et accès

```python
from memory.consolidation import HebbianMemoryGraph

graph = HebbianMemoryGraph(config, device)

# Ajouter un épisode
node_id = graph.add_node(embedding)

# Mise à jour hebbienne
graph.update_hebbian(current_idx, retrieved_indices)

# Retrieval
results = graph.spreading_activation(query_embedding, k=5)

# Oubli
graph.adaptive_forgetting()
```

### C. Agent Stream (`stream/agent.py`)

Agent central pour traitement temps réel :
- **Perception** : Vision (InsightFace), Audio (Whisper), Texte (SentenceTransformer)
- **Résolution d'Entités** : Unification face ↔ voice ↔ entity
- **Encodage Multimodal** : Fusion des modalités via HCANN
- **Stockage** : Ajout au graphe hebbien avec associations

```python
from stream.agent import HCANN_Agent
from utils.config import HCANNConfig

config = HCANNConfig()
agent = HCANN_Agent(config)

# Traitement d'un pas de temps
agent.process_step(video_frame, audio_data, timestamp)

# Requêter la mémoire
response = agent.query_memory("Qui était présent ?")
print(response)
```

---

## ⚙️ Configuration

Éditez `utils/config.py` pour personnaliser :

```python
@dataclass
class HCANNConfig:
    # Multimodal
    use_vision: bool = True
    use_language: bool = True
    semantic_dim: int = 256
    
    # Hippocampe
    grid_periods: List[int] = None  # [3, 5, 7] par défaut
    hpc_size: int = 256
    scaffold_lr: float = 0.01
    
    # Cortex
    wm_slots: int = 7
    attractor_dim: int = 128
    
    # Graphe hebbien
    hebbian_lr: float = 0.02
    hebbian_decay: float = 0.995
    hub_threshold: int = 10
    prune_age_thresh: float = 86400 * 7  # 7 jours
    
    # Entraînement
    consolidation_freq: int = 10
```

---

## 📊 Comparaison avec l'État de l'Art

| Fonctionnalité | HCANN | M3-Agent | MemVerse | HeLa-Mem |
|---------------|-------|----------|----------|----------|
| Encodage multimodal | ✅ Vision+Audio+Texte | ✅ Vision+Texte | ✅ Multimodal | ❌ Texte seul |
| Graphe hebbien dynamique | ✅ | ❌ | ⚠️ Statique | ✅ |
| Consolidation LLM | ✅ | ✅ | ❌ | ❌ |
| Cellules de grille | ✅ Vector-HaSH | ❌ | ❌ | ❌ |
| Spreading activation | ✅ 2 phases | ❌ | ⚠️ PageRank | ✅ Simple |
| Oubli adaptatif | ✅ Multi-critères | ❌ | ❌ | ✅ Poids |
| Entités unifiées | ✅ Face↔Voice | ❌ | ❌ | ❌ |
| Attracteurs corticaux | ✅ | ❌ | ❌ | ❌ |

---

## 📈 Axes de Développement

### Court Terme
- [ ] Intégration ERes2NetV2 pour reconnaissance vocale
- [ ] Optimisation CUDA du spreading activation
- [ ] Tests unitaires complets
- [ ] Benchmark de retrieval

### Moyen Terme
- [ ] Replay nocturne (consolidation offline)
- [ ] Attention cross-modale pour fusion optimale
- [ ] Hiérarchie de hubs (super-hubs abstraits)
- [ ] Interface de visualisation interactive

### Long Terme
- [ ] Apprentissage non-supervisé des attracteurs
- [ ] Génération de scénarios futurs
- [ ] Mémoire collective multi-agents
- [ ] Embodiment robotique

---

## 📚 Références Scientifiques

HCANN s'inspire des travaux suivants :

1. **Vector-HaSH** - Cellules de grille pour encodage spatio-temporel
2. **CLS Theory** (McClelland et al., 1995) - Complémentarité hippocampe/cortex
3. **Règle de Hebb** - "Neurons that fire together, wire together"
4. **M3-Agent** - "Seeing, Listening, Remembering, and Reasoning"
5. **MemVerse** - "Multimodal Memory for Lifelong Learning Agents"
6. **HeLa-Mem** - "Hebbian Learning and Associative Memory for LLM Agents"
7. **Hippocampal Scaffolds** - "Episodic memory from spatial scaffolds"

Les PDF de ces articles sont disponibles dans le dossier `document/`.

---

## 💡 Philosophie HCANN

> *"La mémoire n'est pas un stockage passif, mais un processus actif de reconstruction."*

HCANN incarne cette vision en combinant :
- **Biologie** : Mécanismes neuronaux validés expérimentalement
- **IA Moderne** : LLMs pour abstraction, transformers pour fusion
- **Ingénierie** : Systèmes robustes, testables et déployables

L'objectif ultime : Créer un agent capable d'**apprendre continuellement**, de **se souvenir contextuellement**, et d'**oublier intelligemment** — comme un humain.

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Veuillez :
1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/amelioration`)
3. Committer vos changements (`git commit -m 'Ajout fonctionnalité'`)
4. Pusher (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

---

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 📞 Contact

Pour toute question ou collaboration, veuillez ouvrir une issue GitHub.

---

**Développé avec ❤️ par l'équipe HCANN**  
**Version** : 1.0.0  
**Dernière mise à jour** : Avril 2026