# Audit latence retrieval Danann — verdict mesuré (2026-06-12)

> ⚠️ **Document historique (point-in-time).** Sa conclusion « reranker = aucun gain
> fiable, à désactiver » a été **dépassée par la Phase 2D** (PRs #51→#55). La cause
> racine du non-gain était la combinaison testée — modèle `ms-marco` **anglais** +
> fenêtre 8 + troncature 512 — pas le reranker en soi. Avec le modèle **multilingue
> `mmarco`**, fenêtre 16 et cut 1000, le gain FR est mesuré et reproductible
> (hit@3 48/56 vs 43/56, `scripts/eval_rag.py`) : le reranker est désormais **ON par
> défaut**. Le diagnostic *latence* ci-dessous (le reranker domine le coût) reste
> valide ; seule la recommandation de le couper est caduque. Détails dans le CHANGELOG
> (entrées Phase 2D) et `AGENTS.md`.

**Contexte.** Hypothèse de départ : « Danann est trop lent à l'échelle, et un
mini-RAG fragmenté serait plus efficace qu'un gros corpus monolithique. »
Méthode imposée : mesurer avant de proposer (`time.perf_counter`), vérifier
l'IVF, puis A/B monolithique vs fragmenté. Aucune optimisation à l'aveugle.

**Banc d'essai.** Index `data/models/index_code` = **46 569 chunks / 9 langages**
(python, bash, shell, git, javascript, css, html, docker, sql), compression
int8, embedder `intfloat/multilingual-e5-small` (CPU). Machine dev :
i5-10210U, 7 Go RAM (swap actif), GPU MX330 inutilisable (sm_61 non supporté
par le torch installé). 16 requêtes FR couvrant les 9 langages, top_k=5.
(L'index Wikipédia 38k a été écarté : bâti avec l'ancien embedder MiniLM,
désaligné de l'embedder actuel.)

---

## Étape 1 — Où part le temps (pipeline réel, int8 flat)

| Étage | p50 | p95 |
|---|---|---|
| Embedding de la requête | 30 ms | 40 ms |
| Recherche vectorielle (scan int8 complet) | 56 ms | 74 ms |
| **Rerank cross-encoder CPU (15 paires)** | **1 753 ms** | **2 291 ms** |
| **Total pipeline** | **1 830 ms** | **2 380 ms** |

**Verdict : le goulot du retrieval est le reranker à ~96 %.** La recherche
vectorielle elle-même coûte 56 ms sur 46k chunks — ce n'est pas elle le
problème.

Constats annexes (lecture du code, confirmés à la mesure) :

1. **Le reranker ne tourne même pas en prod actuellement** :
   `CrossEncoderReranker.load()` instancie `CrossEncoder(model_name)` sans
   `device` → sentence-transformers choisit CUDA si dispo → erreur
   `no kernel image` sur la MX330 → exception silencieuse → les candidats
   sont rendus **non re-classés**. Le pipeline effectif aujourd'hui ≈ 86 ms,
   reranker inopérant.
2. Le commentaire du code (`reranker.py`) annonce « 5-15 ms/paire CPU » ;
   mesuré : **~117 ms/paire** (chunks p50 = 568 chars, CPU U-series).
3. `Int8Index.search` (`quantization.py:114`) fait
   `codes.astype(np.float32) @ q` → re-matérialise **71,5 Mo de float32 à
   chaque requête** (46 569×384×4 o) sur une machine qui swappe.
4. **Perspective** : la génération RWKV est à ~12,7 s p50 (plafond matériel
   documenté dans `docs/benchmarks.md`). Le retrieval (86 ms) n'est PAS ce
   qui rend Morrigan lente de bout en bout.

## Étape 2 — IVF : bien calibré, mais jamais branché au runtime

- **L'IVF est inactif en prod** : `core/knowledge.py::build_danann` ne passe
  jamais `ann="ivf"` → défaut `flat` → scan complet à chaque requête.
- Calibration intrinsèque saine : **215 clusters = exactement √N**,
  `n_probe` défaut 26 (≈ C/8, ~23 % du corpus scanné), build 2,1 s (lazy).

Sweep probes (recall@5 vs le top-5 exact du flat int8) :

| probes | recall@5 | p50 | p95 | candidats scannés |
|---|---|---|---|---|
| 4 | 0.613 | 0,4 ms | 2,3 ms | 2 128 (5 %) |
| 8 | 0.725 | 1,6 ms | 4,9 ms | 3 899 (8 %) |
| 16 | 0.812 | 3,5 ms | 8,3 ms | 7 085 (15 %) |
| 26 (défaut) | 0.925 | 5,7 ms | 10,0 ms | 10 625 (23 %) |
| 32 | 0.938 | 7,8 ms | 12,1 ms | 12 624 (27 %) |
| **64** | **0.988** | **13,3 ms** | 17,8 ms | 22 064 (47 %) |

**Verdict : l'IVF fonctionne** (à 64 probes : recall 99 %, 4× plus rapide que
le flat ; à 26 : 10× plus rapide, recall 92,5 %). L'activer = un kwarg. Mais
il optimise un étage à 56 ms — c'est une brique pour la croissance du corpus,
pas un gain de ressenti.

## Étape 3 — A/B monolithique vs mini-RAG fragmenté

Protocole : shards = partition de l'index int8 par métadonnée `language`
(9 shards, simples slices — zéro re-embedding). Routeur testé = **centroïde de
shard** (produit scalaire requête ↔ 9 centroïdes, coût ~0).

> ⚠️ Correction au brief : **le routage par Brigid est impossible tel quel** —
> Brigid classifie l'*intention* (factual/reasoning/creative/conversation/
> complex/code), pas le langage ni le domaine. Le routeur par centroïdes est
> l'équivalent fonctionnel le moins cher.

Métrique « bonne réponse » = le top-5 contient un chunk dont l'origine matche
le document attendu (ex. « annuler un commit git » → `man/git-commit` ou
`git-reset`). 16 requêtes FR.

| Mode | Bonne réponse | Bon langage dans top-5 | p50 | Erreurs de routage |
|---|---|---|---|---|
| Monolithique int8 flat | 13/16 | 14/16 | 67 ms | — |
| Monolithique IVF (probes=26) | 12/16 | 14/16 | 6 ms | — |
| **Fragmenté top-1 shard** | **15/16** | **16/16** | **4,6 ms** | **0/16** |
| Fragmenté top-2 shards | 14/16 | 15/16 | 7 ms | 0/16 |
| Fragmenté top-3 shards | 14/16 | 15/16 | 13,6 ms | 0/16 |

**Verdict : hypothèse mini-RAG confirmée — mais pour la QUALITÉ, pas pour la
vitesse.**

- Latence : gain négligeable vs IVF (4,6 ms vs 6 ms). Ce n'est pas l'argument.
- Qualité : **+2 bonnes réponses vs flat, +3 vs IVF**. Le shard top-1 corrige
  les pièges cross-langage connus (ex. « trier un *tableau* en JavaScript »
  qui remontait des éléments `<table>` HTML — faux ami tableau=array/table).
- Routage : **0 erreur sur 16** avec le simple centroïde, et le top-1 dur fait
  MIEUX que top-2/top-3 — réélargir réintroduit le bruit cross-langage.
- Réserve : jeu de 16 requêtes mono-langage propres. Des requêtes ambiguës
  (« centrer une div » → css ou html ?) stresseraient davantage le routeur →
  prévoir un **repli monolithique quand le routeur hésite** (écart de score
  centroïde faible), pour ne jamais créer de faux « je ne sais pas » en RAG
  strict.

## Complément — leviers de coût du reranker (CPU forcé)

| Configuration | p50 |
|---|---|
| 15 paires, passages complets (état « réparé » naïf) | 1 786 ms |
| 15 paires, tronqués à 512 chars | 832 ms |
| 15 paires, tronqués à 256 chars | 547 ms |
| 5 paires, passages complets | 616 ms |
| **5 paires, tronqués à 512 chars** | **305 ms** |

S'ajoute une question de fond : `cross-encoder/ms-marco-MiniLM-L-6-v2` est un
modèle **anglais** — requêtes FR = hors domaine. Son apport qualité en
cross-lingue est à prouver avant de payer ses 300 ms+.

**Éval d'apport (même jeu de 16 requêtes FR, candidats int8 flat)** :

| Configuration rerankée | Bonne réponse (baseline 13/16) |
|---|---|
| fenêtre 15 candidats, tronqués 512 | 12/16 (**dégrade**) |
| fenêtre 15 candidats, complets | 12/16 (**dégrade**) |
| fenêtre 8 candidats, tronqués 512 | 14/16 (+1) |

Variation de ±1 selon la fenêtre = niveau du bruit sur n=16. **Le
cross-encoder anglais n'apporte pas de gain fiable en FR → défaut OFF**,
en gardant l'implémentation réparée (device CPU, troncature, fenêtre
configurable) pour ré-évaluation sur un jeu plus large en Phase 2D.

---

## Recommandations (ordonnées)

1. **Reranker** : forcer `device="cpu"` (fix du bug silencieux), tronquer les
   passages (512 chars) et borner la fenêtre re-classée. **Défaut OFF** —
   l'éval ci-dessus montre que le modèle EN n'apporte pas de gain fiable en
   FR ; à ré-évaluer sur un jeu plus large (ou avec un cross-encoder
   multilingue) en Phase 2D.
2. **Mini-RAG par langage** : implémenter comme pré-filtre dur par shard,
   routé centroïde **top-1**, avec repli monolithique si confiance de routage
   faible. C'est le vrai gain qualité identifié — à intégrer au chantier de
   tuning RAG strict (Phase 2D).
3. **IVF** : exposer `ann="ivf"` dans `build_danann` (+ variable d'env).
   Quasi gratuit, prépare la montée en charge du corpus.
4. **Ne PAS micro-optimiser le scan vectoriel** (56 ms) ni chercher la latence
   perçue dans le retrieval : elle est dans la génération RWKV (~12,7 s,
   plafond matériel). Le streaming (TTFT ~1,4 s) reste le levier de ressenti.

*Scripts de mesure reproductibles : `docs/audit-retrieval-2026-06-12/`
(`bench_danann.py` = étapes 1-2, `bench_minirag.py` = étape 3 + reranker),
à lancer depuis la racine avec `.venv-uv/bin/python`. À intégrer à
`scripts/benchmark.py` lors de la Phase 2D.*
