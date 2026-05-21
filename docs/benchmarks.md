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
| **Time-to-first-token p50 (streaming)** | **1.40 s**  |
| Time-to-first-token p95 (streaming) | 1.52 s |
| Latence génération complète p50 | 11.07 s |
| Latence génération complète p95 | 19.73 s |
| Latence génération moyenne | 10.36 s |
| Latence génération max | 19.73 s |
| Cible README < 1 s (réponse complète) | ❌ non atteinte (voir note) |

## Détail par cas

| Query | Type | generated_by | TTFT | Total | Ancré |
|---|---|---|---|---|---|
| Qu'est-ce que le protocole TCP ? | in-corpus | rwkv | 1.52s | 11.77s | oui |
| Quelle est la différence entre TCP et U… | in-corpus | rwkv | 1.46s | 3.95s | oui |
| Qu'est-ce qu'un réseau neuronal liquide ? | in-corpus | rwkv | 1.40s | 19.73s | oui |
| Qui est la déesse Brigid ? | in-corpus | rwkv | 1.34s | 5.29s | oui |
| Comment trier une liste en Python ? | in-corpus | rwkv | 1.24s | 11.07s | oui |
| Quelle est la recette du cassoulet toul… | hors-corpus | template | — | 0.00s | — |
| Quel temps fera-t-il demain à Tokyo ? | hors-corpus | template | — | 0.00s | — |
| Qui a gagné la coupe du monde 2074 ? | hors-corpus | template | — | 0.00s | — |

## Lecture honnête

- Le **refus hors-corpus est déterministe** : sans contexte fiable, Scáthach n'appelle pas le LLM (latence quasi nulle) et renvoie un « je ne sais pas ». C'est le cœur du « 0 hallucination ».
- **Streaming + contexte réduit (2 chunks)** : le time-to-first-token est le levier de latence *ressentie*. En affichant la réponse au fil de l'eau, l'utilisateur voit le 1er mot bien avant la fin de la génération.
- La **cible < 1 s sur la réponse COMPLÈTE n'est pas atteinte** sur ce CPU contraint : un RWKV-6 1.6B Q4_K génère à ~10-13 tok/s, soit plusieurs secondes pour quelques phrases. Le plafond est matériel (RAM saturée + CPU U-series). Pistes restantes : meilleur matériel, modèle plus petit (écarté : qualité), ou réponses plus courtes.
- L'**ancrage** est mesuré par une heuristique de recouvrement lexical réponse↔contexte — indicatif, pas une preuve d'absence d'hallucination.

_Régénérer : `.venv-uv/bin/python scripts/benchmark.py --output docs/benchmarks.md`_
