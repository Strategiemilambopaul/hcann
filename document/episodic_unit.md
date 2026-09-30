# L’épisode comme unité de mesure

Le mètre n’est pas HCANN. Le mètre est **un journal d’épisodes vécus** (image + texte + temps) et trois questions :

1. La mémoire **fusionne-t-elle** deux visites trop proches ? (confusion exclusive)
2. Retrouve-t-elle la **paire** quand elle refuse de choisir ? (paire@2)
3. Avec un **indice d’identité**, vise-t-elle la bonne visite ? (hit)

N’importe quelle IA qui implémente `encode` + `recall` peut être mesurée. Quatre barres sur le banc jumeaux : **lexical**, **merge** (fusion Jaccard, geste Mem0/Zep), **RAG cosine**, **HCANN**.

Un second protocole mesure la **complétion de motif**. Deux cues :

- **CLIP** (crop, texte partiel, image bruitée) : ce n’est pas un fragment du motif Hopfield. Diagnostic, pas le test CA3.
- **DG stocké** bruité ou masqué, comparé au k-NN dans le même espace. C’est le test Hopfield.

## Contrat

```python
class EpisodicStore:
    name: str
    def encode(episode_id, text, image=None, created_at=None, extra=None) -> str: ...
    def recall(query_text=None, image=None, k=5) -> list[dict]:  # [{id, score, path?, ambiguous?}]
```

Une réponse **ambiguë** (`path="ambiguous_twins"` ou `ambiguous=True`) n’est pas une confusion : le système n’a pas commis de vainqueur unique.

## Lancer les bancs

```bash
python scripts/demo_nonconfusion.py --use-clip   # jumeaux : fusionner ou pas
python scripts/demo_completion.py --use-clip     # [A] cue CLIP  [B] motif DG stocké
```

Code : [`integrations/episodic_bench.py`](../integrations/episodic_bench.py) (mètre), [`integrations/episodic_scenes.py`](../integrations/episodic_scenes.py) (scènes), [`integrations/episodic_memory.py`](../integrations/episodic_memory.py) (référence HCANN).

Pour brancher une mémoire externe : wrapper `encode` / `recall`, l’ajouter au dict `stores`. `MergeStore` est le stand-in in-repo du geste « fusionner les visites trop proches ». Ce n’est pas l’API Mem0.
