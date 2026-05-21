# Benchmarks de génération — Morrigan / Scáthach RWKV

- **Date** : 2026-05-21
- **Machine** : Linux x86_64, Python 3.12.13
- **Modèle** : rwkv-6-world-1.6b-Q4_K.gguf
- **Backend** : rwkv (llama.cpp) (strict_rag=True)

## Synthèse

| Métrique | Valeur |
|---|---|
| Cas testés | 8 (5 générés, 3 hors-corpus) |
| **Taux de refus (hors-corpus)** | **100%** (cible 100% — 0 hallucination) |
| **Taux d'ancrage (générés)** | **100%** |
| Latence génération p50 | 12.71 s |
| Latence génération p95 | 18.56 s |
| Latence génération moyenne | 10.97 s |
| Latence génération max | 18.56 s |
| Cible README < 1 s | ❌ non atteinte (voir note) |

## Détail par cas

| Query | Type | generated_by | Latence | Ancré |
|---|---|---|---|---|
| Qu'est-ce que le protocole TCP ? | in-corpus | rwkv | 12.80s | oui |
| Quelle est la différence entre TCP et UDP ? | in-corpus | rwkv | 4.74s | oui |
| Qu'est-ce qu'un réseau neuronal liquide ? | in-corpus | rwkv | 12.71s | oui |
| Qui est la déesse Brigid ? | in-corpus | rwkv | 6.04s | oui |
| Comment trier une liste en Python ? | in-corpus | rwkv | 18.56s | oui |
| Quelle est la recette du cassoulet toulous… | hors-corpus | template | 0.00s | — |
| Quel temps fera-t-il demain à Tokyo ? | hors-corpus | template | 0.00s | — |
| Qui a gagné la coupe du monde 2074 ? | hors-corpus | template | 0.00s | — |

## Lecture honnête

- Le **refus hors-corpus est déterministe** : sans contexte fiable, Scáthach n'appelle pas le LLM (latence quasi nulle) et renvoie un « je ne sais pas ». C'est le cœur du « 0 hallucination ».
- La **cible < 1 s n'est pas atteinte** pour une génération complète sur ce CPU contraint : un RWKV-6 1.6B Q4_K génère à ~10-12 tok/s, soit plusieurs secondes pour une réponse de quelques phrases. Pistes : quantization plus agressive (Q3_K), modèle plus petit (0.4B), streaming (afficher au fil de l'eau), ou réponses plus courtes. À traiter en optimisation (Phase 4).
- L'**ancrage** est mesuré par une heuristique de recouvrement lexical réponse↔contexte — indicatif, pas une preuve d'absence d'hallucination.

_Régénérer : `.venv-uv/bin/python scripts/benchmark.py --output docs/benchmarks.md`_
