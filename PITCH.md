# 🐺 Morrigan — architecture IA modulaire post-LLM

**TL;DR** : pari archi sur "remplacer un LLM monolithique par des modules
spécialisés", tourne 100 % local sur CPU modeste (sans GPU, sans cloud,
sans hallucination). Repo perso, Python. **Phases 1 à 4 livrées (cœur)**,
**Phase 5 (passage à l'échelle) en cours** 🎉

## L'idée

Un LLM = un bloc opaque de centaines de milliards de params. Morrigan
décompose l'intelligence conversationnelle en **6 modules nommés d'après
la mytho celte**, chacun implémenté par la techno la + efficiente pour
sa tâche :

- **An Dagda** — orchestrateur (routing query → modules)
- **Brigid** — réseau neuronal liquide (CfC via ncps), classification d'intent
- **Ogham** — moteur symbolique + knowledge graph (extraction d'entités/relations)
- **Danann** — mémoire vectorielle (sentence-transformers + cosine + reranker,
  index compressé int8/binary persistable)
- **Scáthach** — génération texte (RWKV Q4 via llama.cpp, RAG strict, streaming)
- **Cauldron** — mémoire de travail (session-based, condensation à venir)
- **Morrigan-Code** — agent spécialisé code (bonus, vérif AST 6 langages)

## Où on en est (Phases 1-4 livrées, Phase 5 en cours)

- ✅ Pipeline end-to-end CPU, ~50-150 ms/query sur i5
- ✅ Reranker cross-encoder **multilingue** (mmarco, ON par défaut, gain FR mesuré) + métadonnées riches sur les chunks
- ✅ **Morrigan-Code 6/6 langages** (Python `ast`, Bash `bash -n`, JS
  `node --check`, SQL `sqlparse`, HTML `html.parser`, CSS `tinycss2`).
  Subprocess sécurisé, stdin only, timeout 5s.
- ✅ **Brigid entraînée** : CfC ncps 384→16→6 neurones, 504 ex FR curatés
  équilibrés, **val_acc 88.2 %** reproductible local↔CI (seed 42),
  checkpoint 240 KB
- ✅ **Knowledge graph** wiré dans Ogham : **267 entités / 430 triplets**,
  requêtes `compare(X,Y)` / `facts_about(X)`, pure-Python (zéro NLP dep)
- ✅ **Génération neuronale RWKV** (Phase 3) : RWKV-6 1.6B Q4_K via
  llama.cpp, **RAG strict** (réponse ancrée sur le corpus OU "je ne sais
  pas" — **100 % de refus hors-corpus**, 0 hallucination), **streaming**
  token par token (1er token ~1.4 s)
- ✅ **Compression d'index** (Phase 4) : quantization int8 (×4) / binary
  (×32), float32 jamais conservé, **persistance disque** (rechargée sans
  réembedder), ANN IVF sous-linéaire — le tout en NumPy pur
- 🚧 **Ingestion à l'échelle** (Phase 5) : streaming Wikipédia FR
  (`scripts/ingest_wikipedia.py`) → index compressé persisté, **servi au
  runtime** via `MORRIGAN_INDEX`. Mesuré : 500 articles → **37 967
  chunks, index int8 14.7 MB (×4.0), chargé en 0.29 s** (zéro
  réembedding au boot). Cf. `docs/ingestion.md`.
- ✅ **5 workflows CI** : `tests`, `version-sync-check`, `release` (tag →
  GitHub Release auto), `brigid-train` (réentraîne le CfC), `kg-build`
  (re-ingère le corpus)

## Stack

Python 3.12 · PyTorch + ncps (LNN) · sentence-transformers (e5-small multilingue 384-D) ·
llama-cpp-python (RWKV Q4, CPU) · networkx · pyDatalog · Jinja2 ·
datasets (streaming Wikipédia) · Supabase pgvector (optionnel) ·
python-telegram-bot

## Le pari

> "L'information utile qu'un LLM stocke dans ses poids tient dans une
> fraction de l'espace, si on la décompose correctement."

Qualité conversationnelle ciblée avec **~100× moins de paramètres**,
0 cloud, 0 hallucination (réponse depuis corpus vérifiable OU "je ne
sais pas" honnête).

## Démo concrète : `Compare TCP et UDP`

```
An Dagda  → Brigid (LNN) classifie "reasoning" confidence 0.85
          → fallback heuristiques NON déclenché (au-dessus du seuil)
          → route [danann, ogham, scathach]
Danann    → top-3 chunks via embedding + reranker, filtrés domain=reseau
Ogham     → kg.compare(tcp, udp) :
              common  = [protocole]
              tcp_only = [fiable, transmission_control_protocol, …]
              udp_only = [rapide, user_datagram_protocol, …]
Scáthach  → génération RWKV ancrée sur les chunks + faits KG (RAG strict)
```

Le tout en CPU local, sans appeler un seul LLM externe.

## Compteurs sympas

- **411 tests pytest verts**, 0 xfailed
- **55 PRs mergées**, CI verte de bout en bout
- **Déterminisme parfait local ↔ CI** sur l'entraînement Brigid (seed 42)
- Gros corpus encyclopédique réel tenu **sur PC modeste** : index
  Wikipédia int8 chargé en 0.29 s, RAM réduite ×4
- License propriétaire, repo privé

## Et après ? (Phase 5 — production)

Déjà livré : **API HTTP FastAPI + SSE**, **ANN IVF + int8** (index compressé
servi sans scan linéaire), **Dockerisation** (image CPU + compose, validée en
CI). Reste :

- Intégration avec l'écosystème (notamment Gungnir, client HTTP loose-coupling)
- Déploiement VPS (runner self-hosted + cadrage RAM)
- Monitoring/observabilité (Prometheus optionnel)
- Benchmarks vs LLMs commerciaux sur domaines ciblés (réseau, code,
  mytho celte 🌿)

C'est là que le pari se transforme en outil. À suivre.

---

Pas un produit, projet de recherche perso. Mais le pipeline tourne de
bout en bout — génération neuronale ancrée, corpus compressé à l'échelle,
0 hallucination. Curieux d'avoir vos retours d'archi, vos crash-tests,
vos idées d'extraction sémantique sans NLP lourd 🐺
