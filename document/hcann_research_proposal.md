# HCANN — proposition de recherche

Mémoire épisodique légère, externe au modèle, pour un agent qui reste avec une entité.

HCANN s’inspire de principes de la mémoire épisodique humaine. Il ne prétend pas reproduire le cerveau. Chaque mécanisme ci-dessous a une fonction computationnelle. Un module sans gain mesuré sort du cœur.

État au 1er octobre 2026. Les chiffres cités viennent de `results/episodic/nonconfusion.json` et `results/episodic/completion.json` (CLIP, seed 0). Ce qui n’a pas été mesuré est marqué **ouvert**.

---

## 1. Définition

HCANN est un journal d’épisodes. L’unité est une expérience vécue par une entité : qui a vécu quoi, quand, où, avec qui, dans quel contexte, avec quelle intention et quel résultat, et comment cette visite se rapporte aux précédentes.

L’embedding sert d’index. Il n’est pas le souvenir. Le souvenir est la trace (`episode_id` + champs). Un modèle de langue peut lire les traces rappelées. Ses paramètres ne sont pas le lieu de la mémoire.

La trace câblée porte l’identifiant, le texte, l’image optionnelle, la date, et un objet `trace` (entité, lieu, participants, contexte, événement, intention, résultat, importance, liens). Ces champs sont une métadonnée du nœud. Ils ne sont pas concaténés à la légende et n’ajoutent pas de couche neuronale. Le rappel ne s’en sert pas encore pour classer.

## 2. Problème

Les mémoires d’agents stockent des faits (« Paul travaille sur l’IA ») ou des vecteurs de documents. Deux visites trop proches — même lieu, légendes presque identiques, issues différentes — y sont fusionnées ou départagées au hasard. L’agent répond alors comme s’il n’y avait eu qu’une expérience.

La question n’est pas d’agrandir un réseau. C’est : quelle représentation minimale, et quels mécanismes minimaux, permettent à une IA de garder une mémoire de ses expériences.

## 3. Hypothèse principale

Une mémoire externe, structurée en traces d’épisodes, avec une politique qui refuse de fusionner deux visites trop proches, permet à un agent de rappeler une expérience personnelle, datée et contextuelle, sans modifier les poids du modèle de langue.

Forme testable : à journal égal, cette politique baisse la confusion exclusive par rapport au cosine et par rapport à une fusion de faits, et elle retrouve la paire quand elle refuse de choisir.

Résultat partiel déjà obtenu (32 requêtes, 12 épisodes, 4 paires quasi identiques) :

| Système | Confusion exclusive | Hit | Paire@2 |
|---|---|---|---|
| Fusion de légendes (Jaccard ≥ 0,5) | 37,5 % | 37,5 % | 0 % |
| Cosine CLIP | 28,1 % | 65,6 % | 78,1 % |
| HCANN | 12,5 % | 81,2 % | 93,8 % |

Le gain vient surtout des 16 rappels laissés ambigus, pas des 8 paris CA3 (4 hits, 4 confusions, toujours le même membre de la paire).

## 4. Contributions visées

1. L’épisode comme unité de mémoire d’un agent, distincte du fait et du document.
2. Une politique de rappel qui engage un vainqueur unique seulement si un indice le justifie, et qui rend la paire sinon.
3. Un mètre (`encode` / `recall`) sur lequel d’autres mémoires passent les mêmes visites.
4. Un résultat négatif assumé : le Hopfield sur codes DG n’est pas un retriever visuel (cue image recodée : hit 25 % contre 75 % pour le cosine ; cue = motif DG déjà stocké : 100 %, égal au plus proche voisin).
5. **Ouvert** : le même schéma pour plusieurs entités, l’ordre temporel, la consolidation qui n’efface pas les épisodes, l’oubli vérifiable.

## 5. Architecture minimale

```text
Expérience
    → Episode Encoder (CLIP gelé, ou texte seul)
    → Episodic Trace (l’enregistrement)
    → Memory Store (un nœud par visite, isolé par entité)
    → Recall policy (identité, sinon paire, sinon cosine)
    → traces rendues
    → LLM ou agent, en lecture seule
```

Consolidation, oubli et lien entre épisodes sont des opérations sur les traces. Ils n’entrent dans le cœur que s’ils améliorent une métrique du banc sans relever la confusion exclusive.

DG, CA3, CA1 restent un substrat d’encodage déjà écrit. Ils ne sont pas étendus. CA3 sert de juge d’indécision (marge 0,2) et reste interdit quand la requête n’a pas de texte.

## 6. Vie d’un épisode

1. Une entité vit quelque chose : image, note, date, lieu, participants, intention, issue.
2. L’encodeur produit un vecteur de recherche. La trace est écrite sous un identifiant stable.
3. Une seconde visite proche crée un second identifiant. Aucune fusion à l’écriture.
4. Une requête réactive des traces. Si deux traces collent et qu’aucun indice ne les sépare, les deux sont rendues, marquées ambiguës.
5. Un lecteur (humain ou LLM) voit les traces, leur date et le chemin (`semantic`, `identity`, `ambiguous_twins`). Il peut dire « je ne sais pas » si le chemin est ambigu ou si le score est sous un seuil.
6. Une consolidation éventuelle écrit une connaissance dérivée à part. Elle ne supprime pas les épisodes sources.
7. Un oubli demandé efface la trace, le vecteur et les liens de cet identifiant.

## 7. Encodage

Fonction : rendre une expérience cherchable sans en faire le souvenir.

Implémenté : CLIP gelé (vision et texte projetés, moyenne si les deux sont présents) puis code sparse entorhinal et gyrus denté. Le code DG est stocké avec le nœud. Le banc tourne avec `enable_dreaming=False`.

À ne pas ajouter tant qu’un banc ne le demande pas : world model, diffusion, cellules de lieu, second encodeur appris.

## 8. Stockage

Fonction : une visite, un enregistrement, une entité propriétaire.

Implémenté : nœud de graphe hébbien (`episode_id`, texte, date, embedding sémantique, code DG, objet `trace`) et motif CA3 indexé par le même identifiant. `identity` reste à part, pour l’indice lexical. Le schéma ci-dessous est celui écrit à l’encodage (`integrations/episodic_trace.py`).

Schéma, sans nouveau réseau :

```json
{
  "trace_id": "E_2026_09_30_017",
  "entity_id": "user_001",
  "timestamp": "2026-09-30T20:14",
  "place": null,
  "participants": [],
  "context": "research",
  "event": "refonte de HCANN",
  "intent": "mémoire épisodique légère",
  "outcome": "priorité à la trace, pas aux couches",
  "importance": null,
  "links": []
}
```

L’universalité est ce schéma : personne, agent, robot ou groupe sont des `entity_id`. **Ouvert** : l’isolation est un filtre sur `entity_id`, pas encore un banc.

## 9. Consolidation

Fonction : dériver une connaissance stable sans écraser les visites.

Trois états proposés : trace récente, épisode conservé, connaissance générale écrite à côté (`pattern_id` qui cite ses `trace_id`).

Le code contient un rêve hébbien (`memory/dreaming.py`). Il est éteint sur le banc fondateur. Le mesurer consiste à vérifier deux choses ensemble : la connaissance dérivée est juste, et la confusion exclusive sur les jumeaux ne remonte pas. Une fusion qui résume « Paul simplifie souvent » et jette les dates échoue ce second test. C’est le geste déjà perdu par `MergeStore`.

## 10. Oubli

Fonction : ne pas traiter toute trace comme également disponible, et permettre une suppression réelle.

Pistes, toutes **ouvertes** : importance (nouveauté, répétition, utilité future déclarée), dépriorisation au rappel, obsolescence temporelle, effacement sur demande.

Critère d’effacement : après `forget(trace_id)`, l’identifiant est absent du rappel, du déjà-vu et du graphe. Effacer le fichier image seul ne compte pas.

Le seuil de refus au rappel est le premier oubli utile déjà compatible avec le code : sous le seuil, la réponse est « pas assez sûr », pas un faux épisode.

## 11. Retrieval

Fonction : réactiver une expérience à partir d’un indice, et refuser un vainqueur unique quand l’indice ne tranche pas.

Politique implémentée :

| Condition | Chemin | Engagement |
|---|---|---|
| Pas de jumeau, ou cosine séparé (marge ≥ 0,05) | `semantic` | oui |
| Jumeaux et recouvrement lexical différent | `identity` | oui, la visite nommée |
| Jumeaux, texte, attracteurs séparés (marge CA3 ≥ 0,2) | `ca3` | oui, et souvent le mauvais jumeau |
| Jumeaux sans texte, ou CA3 indécis | `ambiguous_twins` | non |

Similarité : cosine CLIP. Temporalité, entité, liens et importance : **ouverts**, comme filtres ou réordonnancements sur les traces, pas comme nouveau réseau. La causalité n’est pas inférée tant qu’un lien n’a pas été écrit.

## 12. Personal Episodic Trace

Une entité n’a pas un profil. Elle a une suite de traces.

```text
entity_id = user_001
  E_2026_01  première discussion
  E_2026_03  début du projet
  E_2026_06  décision intermédiaire
  E_2026_09  refonte : alléger HCANN
```

La question « pourquoi avons-nous simplifié HCANN ? » doit ramener `E_2026_09` et, si on le demande, les voisins temporels. Elle ne doit pas ramener seulement le fait « Paul aime l’IA ».

Deux entités n’ont pas les mêmes nœuds. Une requête porte `entity_id`. Une trace d’une autre entité qui sort dans le top-k est un échec d’isolation.

## 13. Banc

Le banc actuel est la tâche « ne pas fusionner deux visites ». Il couvre une partie du rappel épisodique, du contexte et de l’identité lexicale. Le reste est un protocole à construire sur le même contrat `encode` / `recall`.

| Tâche | Question | Statut |
|---|---|---|
| Episodic recall | La bonne visite est-elle retrouvée ? | Mesuré (hit) |
| Non-fusion | Deux visites proches restent-elles deux traces ? | Mesuré (confusion exclusive, paire@2) |
| Temporal recall | L’ordre et la date sont-ils justes ? | Ouvert |
| Contextual recall | Le lieu, les participants, l’intention reviennent-ils avec l’événement ? | Stockés et rendus avec le hit. Pas encore un signal de classement. |
| Entity-centric recall | Seules les traces de l’entité demandée reviennent-elles ? | Ouvert |
| Longitudinal consistency | Après N visites, les anciennes paires tiennent-elles ? | Ouvert |
| Consolidation | Une connaissance dérivée est-elle juste sans effacer les sources ? | Ouvert |
| Controlled forgetting | L’effacement et le refus sont-ils effectifs ? | Ouvert |
| Personalization | Deux entités divergent-elles après les mêmes questions ? | Ouvert |
| Efficiency | Taille du journal et coût d’un rappel | À reporter (12 épisodes aujourd’hui) |
| Privacy / isolation | Fuite inter-entités nulle, suppression totale | Ouvert |

Les scènes actuelles : 4 paires d’images quasi identiques, légendes à fort recouvrement, 4 distracteurs, 4 types de requêtes (`shared_caption`, `identity_cue`, `image+shared_caption`, `degraded_image`). Voir `integrations/episodic_scenes.py`.

## 14. Métriques

Déjà définies dans `integrations/episodic_bench.py` :

- confusion exclusive : vainqueur unique commis sur le jumeau ;
- hit : bonne cible si engagé, cible dans le top-2 si ambigu ;
- paire@2 : les deux visites sont dans les deux premiers ;
- taux de non-engagement.

À ajouter avec les tâches ouvertes, sans changer les trois premières :

- exactitude d’ordre (paires temporelles) ;
- fuite : fraction de traces d’une autre entité ;
- survie des sources après consolidation ;
- rappel après `forget` (doit être nul) ;
- faux souvenir : réponse affirmative sous le seuil de refus ;
- coût : nombre de traces, temps d’un `recall`.

Un hit sur paire ambiguë signifie « la paire est retrouvée », pas « l’agent a reconnu la bonne visite ».

## 15. Baselines

Même journal, même requêtes.

| Baseline | Rôle | Déjà dans le banc |
|---|---|---|
| Cosine CLIP (`RagStore`) | mémoire vectorielle | oui |
| Recouvrement lexical | texte seul, vainqueur unique | oui |
| Fusion Jaccard ≥ 0,5 (`MergeStore`) | mémoire de faits qui fusionne | oui |
| k-NN dans l’espace DG | même espace que CA3, sans Hopfield | oui, banc complétion |
| Contexte de conversation seul | pas de mémoire externe | ouvert |
| Résumé glissant | compression qui peut effacer une visite | ouvert |
| Mémoire d’agent publiée (Mem0, Zep) | fait de dialogue, autre unité | à brancher par `encode` / `recall`, pas à réimplémenter |

Ne pas déclarer une victoire sur LoCoMo ou Memory-QA. Ces bancs mesurent des faits de chat et des questions sur des photos. Le mètre HCANN mesure la non-fusion d’épisodes. Un échec y est un résultat.

## 16. Ablations

Déjà lisibles dans les chemins de rappel :

- sans refus d’ambiguïté, HCANN retombe sur le cosine ;
- sans indice lexical, l’identité ne se fait pas ;
- CA3 engagé sur image seule dégrade le banc (politique actuelle : interdit) ;
- CA3 sur jumeaux textuels : 4/8, le mécanisme n’explique pas le gain global ;
- complétion Hopfield contre k-NN DG : pas de gain sur motif stocké bruité, masqué ou mélangé.

Ablations **ouvertes** : rappel avec date seule, avec `entity_id` seul, consolidation avec et sans conservation des sources, oubli par dépriorisation contre effacement.

Chaque gain doit nommer le mécanisme. Le gain actuel se nomme : refus de commettre un vainqueur unique.

## 17. Données

Disponibles : scènes synthétiques du dépôt (images dessinées, légendes contrôlées). Suffisant pour la non-fusion. Insuffisant pour une personne réelle sur des mois.

À construire, dans cet ordre :

1. Journal à deux entités, visites jumeaux plantées dans chacune, pour l’isolation.
2. Journal longitudinal court (quelques dizaines de visites datées) avec questions d’ordre.
3. Traces réelles volontaires (carnet d’étude), avec paires de séances presque identiques, consentement, et droit d’effacement.

LoCoMo, LongMemEval et Memory-QA restent des contrôles externes : ils disent si HCANN échoue quand la tâche n’est pas la sienne. Ils ne sont pas le critère de succès.

## 18. Cas d’usage

Le cas que le mécanisme justifie : un assistant personnel qui répond à « pourquoi avons-nous simplifié HCANN ? » par l’épisode, la date et les visites voisines, et qui dit « deux séances se ressemblent » au lieu d’en inventer une.

Même mécanisme, plus tard, pour un tuteur (deux séances sur le même exercice), une équipe (deux décisions proches), un robot (deux passages au même lieu). Le médical seulement avec isolation et effacement vérifiés. Aucun de ces déploiements n’est une preuve. La preuve reste le banc.

## 19. Limites

- Un seul journal synthétique, 32 requêtes, une graine.
- Les jumeaux visuels partagent une silhouette. CA3 ne les sépare pas de façon fiable.
- Lieu, participants, intention et résultat sont sur la trace. Ils n’ont pas encore changé un classement : le banc jumeaux, relancé après le schéma (CLIP, seed 0), reste à 12,5 % de confusion exclusive.
- Pas de modèle de langue dans la boucle. La personnalisation sans fine-tuning est une conséquence de l’architecture, pas une expérience.
- Le graphe hébbien relie des nœuds proches à l’encodage. Ce n’est pas une causalité.
- « Universel » signifie « même schéma d’entité ». Cela n’a pas été montré sur un robot ni sur un groupe.

## 20. Risques

Scientifiques : prendre le hit ambigu pour une reconnaissance ; prendre un profil fusionné pour une mémoire ; publier un gain de consolidation qui augmente la confusion exclusive ; comparer des unités différentes (fait de chat contre visite).

Éthiques : une trace personnelle est une histoire, pas un cache. Écrire sans consentement, mélanger deux entités, ou laisser un vecteur après un effacement sont des échecs du système. Affirmer un épisode sous le seuil est pire qu’un refus. Pas de promesse clinique.

## 21. Feuille de route

1. **Fait.** Politique de non-fusion, trois baselines, statut de CA3, dépôt réduit au journal.
2. **Fait.** Trace structurée à côté de la légende, sans nouveau module. Banc jumeaux relancé : confusion exclusive inchangée (12,5 %). Le schéma n’a pas encore de rôle au classement.
3. **Isolation.** Deux `entity_id`, même images. Métrique de fuite. Cible : fuite nulle.
4. **Temps.** Questions d’ordre sur les dates déjà stockées. Baseline : cosine sans date.
5. **Consolidation.** Écrire un pattern qui cite ses sources. Succès seulement si le pattern est juste et si la non-fusion tient.
6. **Oubli.** `forget` efface trace, vecteur et motif. Test d’absence.
7. **Lecteur.** Un petit modèle local lit les traces. Comparer modèle seul, modèle + cosine, modèle + HCANN, sur les mêmes questions d’épisode. Le modèle n’est pas réentraîné.
8. Arrêt. Si une étape n’apporte rien à sa métrique, le mécanisme correspondant ne reste pas dans le cœur.

## 22. Article possible

**Titre.** A lightweight episodic trace that refuses to merge similar visits.

**Résumé.** Les mémoires d’agents réduisent l’expérience à un fait ou à un vecteur. Nous définissons l’unité comme la visite (contenu, temps, entité) et une politique de rappel qui n’engage un vainqueur que si un indice sépare deux visites trop proches. Sur un journal synthétique de paires quasi identiques, cette politique ramène la confusion exclusive de 28,1 % (cosine) et 37,5 % (fusion de légendes) à 12,5 %, surtout en rendant la paire. Un Hopfield sur codes visuels compressés n’explique pas ce gain. La mémoire est externe au modèle de langue.

**Claim.** Le plus petit mécanisme qui fait la différence, sur ce banc, est le refus de fusionner. Le reste est une feuille de route, pas un résultat.
