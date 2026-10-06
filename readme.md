# HCANN — journal d’épisodes vécus

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
L’unité de mémoire est un **épisode** : image + texte + temps, avec une trace structurée à côté de la légende.

Le geste distinctif : **ne pas fusionner deux visites trop proches**. Identité dans la requête → `identity`. Sinon complétion CA3 si les attracteurs sont séparés, sinon paire `ambiguous_twins`. Le cosine RAG choisit un jumeau.

Le récit complet : [document/HCANN_concept.md](document/HCANN_concept.md). Le **mètre** (épisode comme unité de mesure des IA) : [document/episodic_unit.md](document/episodic_unit.md).

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Linux/Mac : source venv/bin/activate
pip install -r requirements.txt
```

Poids CLIP locaux (démo visuelle) : `checkpoints/clip-vit-base-patch32/`.

## Démo fondatrice

```bash
python scripts/demo_nonconfusion.py --use-clip
python scripts/demo_completion.py --use-clip
```

Export : `results/episodic/nonconfusion.json` et `results/episodic/completion.json`. Sans CLIP, encodeur mock — le protocole tourne, ce n’est pas une preuve visuelle.

```python
from integrations.episodic_memory import EpisodicMemory

mem = EpisodicMemory.create(use_clip=True)
mem.encode(
    "vet_momo",
    "Cat at the vet clinic. This is Momo.",
    image=img,
    created_at="2026-04-02",
    extra={"identity": "Momo"},
)
print(mem.deja_vu(text="Cat at the vet clinic. This is Momo.", image=img))
hits = mem.recall(query_text="Cat at the vet clinic")
# path : semantic | identity | ca3 | ambiguous_twins
```

## Ce que le code est

- **Mètre** [`integrations/episodic_bench.py`](integrations/episodic_bench.py)
- **Scènes** [`integrations/episodic_scenes.py`](integrations/episodic_scenes.py)
- **Journal** [`integrations/episodic_memory.py`](integrations/episodic_memory.py)
- **Réseau** `models/` — CLIP, EC, DG, CA3, CA1, Subiculum
- **Graphe** `memory/consolidation.py`

Tests : `python -m pytest tests/test_episodic_memory.py tests/test_episodic_bench.py tests/test_trisynaptic.py tests/test_semantic_boosted.py`.

Les contributions passent par une branche et une pull request.
