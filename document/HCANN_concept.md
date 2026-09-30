# HCANN — le concept

HCANN (Hippocampal-Cortical Adaptive Neural Network) est un **journal d’épisodes vécus**. L’unité n’est pas un document (RAG) ni un fait de conversation (Mem0, Zep). C’est un épisode : **image + texte + temps**.

Le geste distinctif n’est pas de mieux répondre à des questions factuelles. C’est de **ne pas fusionner deux visites trop proches**.

## C’est

- Encoder un souvenir : `EpisodicMemory.encode`
- Demander si on l’a déjà vécu : `deja_vu`
- Rappeler : `recall`
- Comparer au cosine plat : `rag_recall`

Quand CLIP colle deux jumeaux :

| Requête | HCANN | RAG |
|---------|-------|-----|
| Légende générique | CA3 si attracteurs séparés, sinon paire ambiguë | un jumeau au hasard |
| Indice d’identité (`… Momo`) | départage `identity` | CLIP texte sépare déjà souvent |
| Image dégradée (sans texte) | paire ambiguë — pas de pari CA3 | cosine CLIP |
| CLIP déjà séparé (marge ≥ 0,05) | `semantic` | idem |

API :

```python
from integrations.episodic_memory import EpisodicMemory

mem = EpisodicMemory.create(use_clip=True)
mem.encode("vet_momo", "Cat at the vet. This is Momo.", image=img, created_at="2026-04-02")
hits = mem.recall(query_text="Cat at the vet clinic")
# hits[0]["path"] in {"semantic", "identity", "ca3", "ambiguous_twins"}
```

## Ce n’est pas

- Un SOTA Memory-QA (PENSIEVE et consorts restent hors sujet).
- Un produit MaaS. Le [business plan](Business_Plan_HCANN.md) décrit une ambition ; ce n’est pas la preuve.
- De la *pattern separation* magique dans le gyrus denté. **La preuve** est la politique de rappel (jumeaux inférés, identité, CA3 `complete_among`, paire ambiguë), pas un SOTA hippocampique.
- Un hit unique garanti sur légende générique. Sans indice, CA3 ne parie que si les attracteurs DG sont assez séparés (marge 0,2). Sinon la paire reste ambiguë.

## Protocole (preuve actuelle)

```bash
python scripts/demo_nonconfusion.py --use-clip
```

Quatre paires d’images quasi identiques, légendes à fort recouvrement, plus des distracteurs. Quatre requêtes par cible : légende partagée, indice d’identité, image + légende, **image dégradée** (amorce visuelle sans texte).

CLIP, seed 0, 32 requêtes — `results/episodic/nonconfusion.json` :

- confusion exclusive RAG 28,1 % → merge **37,5 %** → HCANN **12,5 %**
- hit RAG 65,6 % → merge **37,5 %** → HCANN **81,2 %**
- paire@2 RAG 78,1 % → merge **0 %** (un seul nœud fusionné) → HCANN **93,8 %**
- 8 `ca3` (légende avec texte), 16 `ambiguous_twins`, 8 `identity`

`merge` est le geste Mem0/Zep in-repo : Jaccard de légendes ≥ 0,5 → une seule visite. Sur l’indice d’identité, 50 % de confusion (Luna est devenue Momo).

## Statut de CA3

CA3 n’est pas le retriever du journal. Trois jobs, trois verdicts (CLIP, seed 0) :

| Job | Ce qui est câblé | Verdict |
|-----|------------------|---------|
| Encodage | `store(DG, episode_id)` + `complete()` à chaque `fast_encode` | Substrat. Pas la preuve. |
| Rappel jumeaux | `complete_among` si légende + CLIP collé, marge 0,2. Interdit en image seule. | 8 paris / 32 requêtes, **4 hits / 4 confusions** — toujours le même membre de la paire. Le gain du banc est le **refus** (16 `ambiguous_twins`), pas ces 8 paris. |
| Complétion | `complete` sur un motif DG | Cue CLIP (crop) : 25 %, pire que RAG. Cue = DG stocké altéré : 100 %, **égal au k-NN DG**. |

Statut : **attracteur DG + juge d’indécision**. Pas un encodeur visuel. Pas un SOTA hippocampique. Ne pas citer le hit jumeaux ni le hit crop comme preuve de complétion.

Image seule / dégradée : les deux visites ont la même silhouette, CA3 ne parie pas. Sur légende générique, CA3 ne commet que si les attracteurs sont assez séparés.

Sans `--use-clip`, le script tourne en encodeur mock : le protocole s’exécute, ce n’est pas une preuve visuelle.

Le **mètre** (n’importe quelle IA, pas seulement HCANN) : [episodic_unit.md](episodic_unit.md). Jumeaux : lexical, merge, RAG cosine, HCANN. Complétion (visites distinctes) : `python scripts/demo_completion.py --use-clip`.

## Où s’arrête la preuve

- Les jumeaux ne sont plus étiquetés à l’encodage. Un jumeau = cosine CLIP élevé **et** légendes quasi doubles (Jaccard). Deux visites au même magasin avec des légendes différentes ne collapsent pas. `pair_id` reste optionnel si on l’écrit dans les métadonnées.
- `models/` : CA3 Hopfield est **branché** sur le rappel jumeaux (`complete_among`) et mesuré à part en complétion (`complete` sur motifs DG), pas réécrit. DG / CA1 restent le substrat.
- Memory-QA, le mode stream, `main.py` sont d’**autres expériences**. Ils ne fondent pas le concept.
