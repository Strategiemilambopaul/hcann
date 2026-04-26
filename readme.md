HCANN_Memory/
├── models/
│   ├── __init__.py
│   ├── hippocampus.py          # Module hippocampique (apprentissage rapide)
│   ├── cortex.py               # Module cortical (apprentissage lent)
│   └── hcann.py                # Architecture complète HCANN
├── memory/
│   ├── __init__.py
│   ├── replay_buffer.py        # Buffer d'expérience pour consolidation
│   └── consolidation.py        # Mécanisme de transfert hippocampe→cortex
├── train/
│   ├── __init__.py
│   ├── continual_learner.py    # Boucle d'apprentissage continu
│   └── metrics.py              # Métriques (Accuracy, Forgetting)
├── utils/
│   ├── data_utils.py           # Dataloaders incrémentaux
│   └── config.py               # Configuration
├── main.py                     # Point d'entrée
└── requirements.txt

# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Préparer les données (format JSON + images)
# data/multimodal_episodes/
#   ├── episodes.json
#   └── images/

# 3. Lancer l'entraînement
python main.py