# Morrigan — contexte repo pour assistant IA

> Fichier d'embarquement pour travailler sur ce repo avec une IA.
> Source de vérité fine : `CHANGELOG.md` (historique), `README.md` (usage), `PITCH.md` (vision).

## C'est quoi

**Morrigan** est une architecture IA modulaire **post-LLM** : qualité conversationnelle
**100 % locale, CPU seul, sans cloud, sans hallucination**. Le pari : remplacer le
monolithe LLM par des modules spécialisés orchestrés, avec un petit modèle génératif
(RWKV-6 1.6B quantizé) strictement contraint par du retrieval (RAG strict : **sans
contexte fiable, le LLM n'est jamais appelé** — refus déterministe « je ne sais pas »).

- Auteur : Kevin (solo maintainer, répondre en **français**).
- Licence : **propriétaire, tous droits réservés** (pas MIT/Apache).
- Repo : `github.com/kevinggraphiste-hub/morrigan`, branche `main`, version `0.5.0`.
- État : Phases 1–5 livrées + Phase 2 « corpus code » complète (2C) + **Phase 2D tuning RAG strict
  en cours** (reranker multilingue ON, gate recalibré sur cosinus pur, mini-RAG fragmenté par
  langage, IVF exposé — cf. roadmap en bas). **411 tests, 0 xfailed.**

## Architecture — 6 modules celtiques

Le flux d'une requête : `An Dagda` route → modules sollicités → `Scáthach` génère.

| Module | Rôle | Implémentation |
|---|---|---|
| **An Dagda** (`core/dagda.py`) | Orchestrateur : classification d'intention, routage, assemblage | `_ROUTING_MAP` partagé ; fence markdown = priorité absolue → `morrigan_code` |
| **Brigid** (`modules/brigid/`) | Classification d'intention neuronale | LNN/CfC (ncps, 384→16→6, ~60 K params), embeddings e5, val acc **0.971** ; fallback heuristiques si confidence < 0.5 |
| **Danann** (`modules/danann/`) | Mémoire vectorielle / RAG | Embedder `intfloat/multilingual-e5-small` (384-D, préfixes `query:`/`passage:`), compression **int8**/binary, ANN IVF pur-NumPy, index persistés (`corpus.json` + `vectors.npz`), mini-RAG fragmenté par langage (`shard_by`, ON), reranker CrossEncoder **multilingue mmarco ON par défaut** (fenêtre 16, cut 1000, `MORRIGAN_RERANKER=off` pour désactiver) |
| **Ogham** (`modules/ogham/`) | Raisonnement symbolique | Knowledge graph networkx (compare / facts_about), patterns FR |
| **Scáthach** (`modules/scathach/`) | Génération | Backends `template` (Jinja2) / `rwkv` (llama.cpp) / `auto` ; **RAG strict par défaut** (`strict_rag=True`) |
| **Cauldron** (`modules/cauldron/`) | Mémoire de travail | Contexte de session (`session_id`) |
| + **Morrigan-Code** (`modules/morrigan_code/`) | Vérification de code | 6 langages : Python (ast), Bash, JS, SQL, HTML, CSS — subprocess sécurisé stdin/timeout |

Pièces transverses dans `core/` : `embedder_cache.py` (**une seule instance** du modèle
d'embedding partagée Danann↔Brigid — ne pas casser ça, machine à 7 Go de RAM),
`knowledge.py` (`build_danann` : charge `MORRIGAN_INDEX` sinon ré-ingère `data/knowledge/`),
`env.py` (auto-load `.env`), `types.py`, `pipeline.py`.

## Interfaces (`interfaces/`)

- `api.py` — FastAPI : `POST /query`, `POST /query/stream` (SSE), `GET /health`, `GET /stats`.
  Durcie : rate-limit/concurrence (`MORRIGAN_API_MAX_CONCURRENT`), bornes d'entrée,
  inférence offloadée hors event-loop (`asyncio.to_thread`), erreurs non leakées,
  clé API optionnelle (`MORRIGAN_API_KEY` → header `X-API-Key`).
- `openai_compat.py` — **shim OpenAI** : `POST /v1/chat/completions` (+ stream) et
  `GET /v1/models` (modèle unique `morrigan`). Branchable tel quel comme provider
  custom d'un client OpenAI (c'est l'intégration **Gungnir**). Module additif et
  isolé : le retirer = 1 ligne dans `create_app`.
- `cli.py`, `telegram_bot.py` — streaming token par token (TTFT ~1.4 s).

## Génération (RWKV)

- Modèle : **RWKV-6 World 1.6B GGUF Q4_K** (~993 Mo, gitignoré) via `llama-cpp-python`
  (wheel CPU index abetlen). Fetch : `scripts/fetch_rwkv_model.py`.
- Format prompt RWKV World `User:/Assistant:`, `repeat_penalty=1.3` **obligatoire** (sinon boucle).
- Perf mesurée (machine dev i5-10210U, 7 Go RAM) : ~10-12 tok/s, latence p50 ~12.7 s,
  refus hors-corpus 100 %, ancrage 100 % (`docs/benchmarks.md`). La lenteur est ~70 %
  **matérielle** — ne pas chercher une optim code miracle.

## Corpus & index (gitignorés, régénérables)

| Index | Contenu | Script |
|---|---|---|
| `data/models/index_code/` | **46 569 chunks / 9 langages** (python 7 907, bash, shell, git, js 10 331, css 11 627, html, docker 9 723, sql 2 834), int8 ×4.0, ~17,6 Mo RAM | `scripts/ingest_code_docs.py` |
| `data/models/index_wiki/` | 37 967 chunks Wikipédia FR (⚠️ bâti avec l'ancien embedder MiniLM → **à rebâtir avec e5** avant de le resservir) | `scripts/ingest_wikipedia.py` |
| `data/knowledge/` (tracké) | Petit corpus curé FR (+ docs code `code_*.md`) | ré-encodé au boot via `build_danann` |

`scripts/ingest_code_docs.py` = **registre de sources multi-langage** (`iter_source`,
`--sources`) : `python` (bundle officiel + pydoc), `man` (pages locales), `mdn`
(sparse clone js/css/html), `docker` (sparse clone `docker/docs`), `postgresql`
(tarball docs officiel `postgresql-X.Y-docs.tar.gz` — le tarball *source* ne contient
plus le HTML). Chunker **code-aware** (`chunk_code_doc` : préserve indentation/newlines,
mode `markdown=True` pour titres `#` hors code-fences). Ajouter un langage = brancher
une fonction source dans le registre, le reste est partagé.

Servir : `MORRIGAN_INDEX=data/models/index_code .venv-uv/bin/python -m interfaces.api`

## Lancer / tester (⚠️ pièges)

```bash
# VENV : utiliser .venv-uv/ (Python 3.12, uv). Le .venv/ historique est CASSÉ.
.venv-uv/bin/python -m pytest -q              # 411 passed attendu (~50 s)
.venv-uv/bin/python -m interfaces.cli          # CLI streaming
MORRIGAN_INDEX=data/models/index_code .venv-uv/bin/python -m interfaces.api
```

- Tests : convention **`asyncio.run` partout, PAS pytest-asyncio** (non installé).
- Tests RWKV réels : tournent en local si le GGUF est présent, **skippés en CI**.
- Machine dev : **7 Go RAM, swap déjà utilisé** → ne pas lancer suite complète + build
  d'index en même temps. Un build complet de l'index code ≈ **1 h 40**.
- GPU local (MX330 sm_61) **non supporté** par le torch installé → tout forcer CPU.
  Le CrossEncoder reranker est désormais `device="cpu"` par défaut (bug CUDA corrigé en 2D, PR #51).
- ⚠️ **3 tests réseau** (`test_brigid_inference`) téléchargent le modèle e5 depuis HuggingFace :
  ils échouent **hors-ligne** (« Temporary failure in name resolution ») et passent dès que le
  modèle est en cache / le réseau dispo (CI verte). Idem reranker mmarco (~135 Mo) au 1er usage.
- Docker : `Dockerfile` + `docker-compose.yml` (publie **8100:8000** — le port hôte
  8000 appartient à Gungnir, ne pas le proposer). Image validée par CI
  (`docker-build.yml`), pas de runtime Docker local.

## CI (GitHub Actions, 6 workflows)

`tests.yml` (pytest 3.12), `version-sync-check.yml` (version ↔ CHANGELOG),
`release.yml` (tag `v*` → GitHub Release avec notes du CHANGELOG),
`brigid-train.yml` (réentraîne le CfC, gate val_acc ≥ 0.65 ; checkpoint gitignoré,
régénéré — déterminisme parfait seed 42), `kg-build.yml`, `docker-build.yml`.

## Conventions de travail (à respecter)

1. **Brainstormer avant d'agir** sur tout sujet non trivial (options/tradeoffs/reco).
2. **Branche AVANT de coder** ; jamais de commit direct sur `main` (sauf release mécanique).
3. **Staging explicite** — jamais `git add -A` (`data/` grouille d'artefacts runtime).
4. 2 commits par PR quand applicable : `feat(...)` puis `docs(changelog,readme,...)`.
5. CHANGELOG : entrée dans `[Non publié]` à chaque PR ; **clôturer `[Non publié]`
   (numéro + date) AVANT de poser un tag** `vX.Y.Z`, sinon release sans notes.
6. Bump `core/__version__.py` ⇒ tag + push immédiat (vérifier `check_version_sync.py`).
7. PR mergées en **squash** + suppression de branche ; attendre la CI verte.
8. Compteurs à tenir synchro en PR : nb de tests dans `README.md` + `PITCH.md`.

## Invariants techniques (ne pas casser)

- **RAG strict** : sans chunks Danann / faits Ogham pertinents, le LLM n'est PAS appelé.
  C'est LE pari du projet (« 0 hallucination ») — toute feature doit le préserver.
- **Embeddings L2-normalisés** (`normalize_embeddings=True`) : toute la chaîne
  (int8/binary/IVF) suppose norme = 1 (dot = cosinus).
- **Préfixes e5** : `query:` à la recherche, `passage:` à l'indexation
  (`core.embedder_cache.text_prompt_prefix`). Un index bâti sans = désaligné.
- **Une seule instance d'embedder en RAM** (`core/embedder_cache.py`), partagée
  Danann↔Brigid. `LABELS` Brigid = tuple gelé (stabilité checkpoint).
- Brigid absent/checkpoint invalide ⇒ **dégradation gracieuse** (heuristiques), jamais d'exception.
- API `Danann.search` retourne `List[Tuple[str text, float score, Dict meta]]` (pas des dicts).

## Roadmap — Phase 2D (tuning RAG strict, en cours)

**Phase 2D — transformer « pertinent » en « bon » sur le corpus code.** Largement livrée
(chaîne PRs #51→#55, toutes mergées) :
1. ✅ Reranker `device="cpu"` (fix bug CUDA, #51), modèle **multilingue mmarco ON par défaut**,
   fenêtre 16 / cut 1000 — gain FR mesuré hit@3 48/56 vs 43/56 (#55).
2. ✅ `MIN_RELEVANCE_SCORE` recalibré sur le **cosinus pur** (0.84, embedder e5) — le seuil
   0.42 datait de MiniLM et était inopérant (#54).
3. ✅ Jeu de Q/R code committé (`scripts/eval_rag.py`, 56 requêtes in-corpus) — sert de
   harnais d'éval reproductible (les 2 ratés FR historiques « trier un *tableau* en JS » et
   « balise lien hypertexte » y figurent).
4. ✅ Mini-RAG fragmenté par langage (`shard_by`, ON) + IVF exposé (`MORRIGAN_ANN`, `MORRIGAN_IVF_PROBES`).

Reste en 2D : élargir le jeu Q/R au-delà des 56 requêtes (couverture par langage), étendre
`scripts/benchmark.py` au corpus code, éventuellement re-tester un reranker multilingue plus
gros si le gain le justifie.

Parkés (ne pas relancer sans Kevin) : déploiement VPS (gated sur runner self-hosted +
vérif RAM, la stack Gungnir tourne déjà dessus), backend Supabase (mort au runtime mais
**conservé volontairement**, ne pas proposer sa suppression), élargissement Stack Overflow.
