# Ingestion à l'échelle — mesures (Phase 5)

- **Date** : 2026-05-25
- **Machine** : Linux x86_64, Python 3.12 (i5-10210U, 7 Go RAM, swap utilisé)
- **Pipeline** : `scripts/ingest_wikipedia.py` (streaming Wikipédia FR) →
  Danann **compressé int8** → `save_index` sur disque → servi au runtime
  via `MORRIGAN_INDEX` (`core.knowledge.build_danann` → `Danann.load_index`)

Objectif : vérifier la thèse Phase 5 — **un gros corpus encyclopédique
réel tenu et servi sur PC modeste**, sans GPU, sans réembedding au boot.

## Synthèse (run réel : 500 articles FR)

| Métrique | Valeur |
|---|---|
| Articles ingérés | 500 |
| Chunks indexés | **37 967** |
| Temps de build (stream + embed + quantize) | 655 s (~11 min) |
| **Index en RAM (int8)** | **14.7 MB** |
| Équivalent float32 | ~58.3 MB (**×4.0**) |
| Taille sur disque | 27 MB (`corpus.json` 14.1 + `vectors.npz` 13.6) |
| **Chargement au runtime** (`load_index`) | **0.29 s** (zéro réembedding) |
| Recherche — 1ʳᵉ requête (cold) | ~10.5 s (lazy-load embedder + 1er scan) |
| Recherche — requêtes suivantes (warm) | **~130 ms** |

Exemple d'ancrage (top-1, requête « Qu'est-ce que l'algorithmique ? ») :

> « Un algorithme est une suite finie et non ambiguë d'instructions […].
> Le domaine qui étudie les algorithmes est appelé l'algorithmique. »
> — score 0.845, `source=Algorithme`, `domain=wikipedia`.

## Lecture honnête

- Le **build est CPU-bound** et lent sur cette machine (~55 chunks/s,
  swap sous pression) : c'est un coût **ponctuel**. Il scale ~linéairement
  (un run de 2000 articles ≈ 110 k chunks ≈ ~33 min, même machine).
- Le **chargement au runtime est quasi instantané** (0.29 s) et
  indépendant de la taille du corpus à l'embedding près : on relit des
  codes int8 (`vectors.npz`), on ne réembedde jamais. C'est tout
  l'intérêt de la persistance compressée — réembedder 38 k chunks au boot
  coûterait ~11 min.
- La **1ʳᵉ requête est lente** (lazy-load du modèle d'embeddings pour
  encoder la query + premier scan), les suivantes sont rapides.
- **Recherche `flat` = O(n)** : à ~38 k chunks c'est encore ~130 ms, mais
  au-delà de ~100 k chunks il faudra l'ANN IVF (Phase 4). **Limite
  connue** : `ann="ivf"` requiert `compression="none"` pour l'instant ;
  combiner IVF + int8 (re-score quantizé sur candidats IVF) est
  l'optimisation naturelle pour servir un index Wikipedia complet.
- La RAM de l'**index** est petite (14.7 MB / 38 k chunks) ; le poste
  mémoire dominant au runtime reste le modèle d'embeddings + (si activé)
  la génération RWKV.

## Reproduire

```bash
# 1. Construire l'index compressé (streaming, sans télécharger le dump)
.venv-uv/bin/python scripts/ingest_wikipedia.py \
    --max-articles 500 --compression int8 --output data/models/index_wiki

# 2. Servir l'index au runtime (CLI ou bot)
MORRIGAN_INDEX=data/models/index_wiki .venv-uv/bin/python interfaces/cli.py
```
