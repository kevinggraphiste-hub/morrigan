# Morrigan

> Architecture IA modulaire post-LLM — un pari : atteindre la qualité
> conversationnelle des LLMs sur des domaines ciblés, en tournant localement
> sur du matériel modeste, sans GPU, sans cloud, sans hallucination.

**Statut : Phases 1 à 4 livrées (cœur).** Pipeline end-to-end CPU,
6 modules + agent Morrigan-Code, classifieur LNN entraîné (Brigid CfC),
knowledge graph (Ogham), génération neuronale RWKV avec RAG strict
(0 hallucination) en streaming, et **compression d'index vectoriel**
(quantization int8/binary ×4-32, persistance disque, ANN IVF). Reste
l'optimisation de la latence de génération et l'ingestion à l'échelle
Wikipedia (Phase 5 → production). Projet de recherche, pas un produit fini.

---

## Philosophie

Les LLMs actuels sont puissants mais massifs, opaques, sujets aux hallucinations,
et dépendants d'infrastructures cloud coûteuses. Morrigan explore une autre
voie : **décomposer l'intelligence conversationnelle en modules spécialisés**,
chacun implémenté par le composant le plus efficient pour sa tâche.

- **Réseau neuronal liquide** (LNN/CfC) pour l'intuition et la classification
- **Moteur symbolique** (pyDatalog) pour le raisonnement logique
- **Mémoire vectorielle** (pgvector + embeddings) pour les connaissances
- **Petit modèle de langage** (templates Jinja2 → RWKV → Llama Q4) pour la génération
- **Mémoire de travail** pour le contexte conversationnel

L'hypothèse centrale : **l'information utile qu'un LLM stocke dans ses poids
tient dans une fraction de l'espace, si on la décompose correctement**.
Les techniques de quantization et de pruning montrent que la majorité des
paramètres d'un LLM sont redondants. Morrigan essaie de prouver qu'une
architecture décomposée, à 100× moins de paramètres, peut rivaliser sur
des domaines bien définis.

Contraintes de design :

- **Tout doit tourner en local sur un PC modeste** (CPU, 8-16 Go de RAM)
- **Pas d'hallucination** : Morrigan répond à partir de son corpus vérifiable
  ou admet qu'elle ne sait pas
- **Chaque module est indépendant et remplaçable** (clean boundaries, API claire)
- **Interprétabilité** : chaque réponse est traçable vers sa source

---

## Architecture

Six modules nommés d'après la mythologie celtique :

```
                        [ Utilisateur ]
                              ↓
                   ┌──────────────────────┐
                   │      An Dagda        │  Orchestrateur
                   │  (routing, assembly) │  (classification + pipeline)
                   └──────────┬───────────┘
                              │
         ┌────────────┬───────┴────────┬────────────┐
         ↓            ↓                ↓            ↓
    ┌────────┐   ┌────────┐       ┌────────┐   ┌──────────┐
    │ Brigid │   │ Ogham  │       │ Danann │   │ Cauldron │
    │  LNN   │   │symbol. │       │ vector │   │ working  │
    │ CfC    │   │pyDatal.│       │ memory │   │ memory   │
    └────────┘   └────────┘       └────────┘   └──────────┘
         │            │                │            │
         └────────────┴────────┬───────┴────────────┘
                               ↓
                         ┌──────────┐
                         │ Scáthach │  Génération de texte
                         │ template │  (templates → RWKV → Llama)
                         │  / LM    │
                         └────┬─────┘
                              ↓
                         [ Réponse ]
```

| Module | Rôle | Phase 1 | Roadmap |
|---|---|---|---|
| **An Dagda** | Orchestration, routing des queries | Heuristiques mots-clés | → classification LNN |
| **Brigid** | Intuition, pattern recognition, classification | Squelette | → CfC entraîné (200-500 ex) |
| **Ogham** | Raisonnement symbolique, règles | pyDatalog de base | → knowledge graph |
| **Danann** | Mémoire vectorielle, retrieval | sentence-transformers + cosine + boost lexical | → multi-level index, reranker |
| **Scáthach** | Génération de texte | Templates Jinja2 | → RWKV / Llama 3.2 Q4 |
| **Cauldron** | Mémoire de travail, contexte | Session-based historique | → condensation, épisodique |

---

## Stack technique

- **Python 3.11+**
- **PyTorch** + **ncps** (Liquid Neural Networks / CfC)
- **sentence-transformers** (`intfloat/multilingual-e5-small`, 384-D, ~470 Mo, 50+ langues — retrieval cross-lingue FR↔EN)
- **pyDatalog** (raisonnement symbolique)
- **Jinja2** (templates de génération — Phase 1)
- **Supabase pgvector** (persistance optionnelle)
- **python-telegram-bot v22** (interface Telegram)
- **numpy** (store en mémoire + cosine similarity)

---

## État actuel (Phases 1-4 livrées, Phase 5 en cours)

### Ce qui marche

- Pipeline end-to-end : input → classification → retrieval → raisonnement → génération → output
- **Classification d'intention par LNN** : Brigid (CfC ncps 384→16→6) entraînée sur 504 exemples FR, val_acc ~88 %, avec fallback heuristiques sous le seuil de confiance
- **Génération neuronale en streaming** : Scáthach v2 backend RWKV-6 1.6B Q4_K (llama.cpp, CPU), mode template en fallback, affichage token par token (CLI + Telegram)
- **RAG strict (0 hallucination)** : génération ancrée sur chunks Danann + faits KG ; refus déterministe « je ne sais pas » sans contexte fiable
- **Knowledge graph** (Ogham) : 267 entités / 430 relations extraites du corpus, requêtes `compare` / `facts_about`
- **Agent Morrigan-Code** : vérification de syntaxe 6 langages (Python, Bash, JS, SQL, HTML, CSS)
- **Compression d'index vectoriel** (Danann) : quantization int8 (×4) / binary (×32 + 2 étages), persistance disque sans float32, ANN IVF sous-linéaire — pour tenir un gros corpus sur PC modeste
- Retrieval hybride Danann : cosine + boost lexical + reranker cross-encoder, filtrage par domaine, garde anti-faux-positif
- Observabilité `/stats` (CLI + Telegram) : routage, probas Brigid, generated_by, latence
- Interfaces CLI + Telegram (streaming), `.env` auto-load, backends Danann branchables (memory / Supabase pgvector), scripts d'ingestion
- **API OpenAI-compatible** (`/v1/chat/completions`, `/v1/models`) : branchable tel quel comme provider custom dans un client OpenAI (dont Gungnir)
- **Corpus de documentation code multi-langage** (Phases 2B/2C) : registre de sources (Python officiel + pydoc, pages man bash/git/CLI), chunker code-aware, index `int8` interrogeable en FR
- **375 tests** (pytest), 6 workflows CI (tests, version-sync, release, brigid-train, kg-build, docker-build)

### Performances mesurées

- **Retrieval/classification** : ~50-150 ms par requête (CPU, i5-10210U), empreinte ~300 Mo
- **Génération RWKV** (1.6B Q4_K, CPU contraint) : ~10-12 tok/s, latence p50 ~12.7 s pour une réponse de quelques phrases — cf. `docs/benchmarks.md`
- **0 hallucination** : 100 % de refus sur les queries hors-corpus (benchmark)
- **Ingestion à l'échelle** : 500 articles Wikipédia FR → 37 967 chunks, index int8 14.7 MB (×4.0), **chargé au runtime en 0.29 s** (zéro réembedding) — cf. `docs/ingestion.md`

### Ce qui ne marche pas encore

- **Latence de génération élevée** : un 1.6B sur CPU contraint reste loin de la cible < 1 s (p50 ~12.7 s ; le streaming masque le ressenti avec un 1er token ~1.4 s). Plafond surtout matériel (RAM saturée + CPU U-series).
- **Pas de zero-shot** : Morrigan ne répond que depuis son corpus. Par design — fallback honnête plutôt qu'hallucination.
- **Passage à très grande échelle (Wikipedia complet)** : l'ingestion à l'échelle est branchée, mesurée et servie au runtime (`MORRIGAN_INDEX`, cf. `docs/ingestion.md`), et la recherche **IVF + compression int8** est disponible (sous-linéaire sur index compressé) ; reste à éprouver le tout sur un corpus de plusieurs millions de chunks.
- **Benchmarks vs LLMs commerciaux** : pas encore réalisés.

---

## Installation

### Prérequis

- Python 3.11+
- Environ 500 Mo de disque (venv + modèle d'embeddings)

### Setup

```bash
git clone <repo>
cd morrigan
python -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

Le premier lancement télécharge `intfloat/multilingual-e5-small` depuis
HuggingFace (~470 Mo).

### Variables d'environnement

Copier `.env.example` en `.env` et remplir au besoin :

```env
MORRIGAN_TELEGRAM_TOKEN=         # Pour l'interface Telegram
SUPABASE_URL=                    # Pour la persistance vectorielle (optionnel)
SUPABASE_KEY=
HF_TOKEN=                        # Pour éviter le rate limit HuggingFace (optionnel)
MORRIGAN_INDEX=                  # Dossier d'index compressé à servir (optionnel)
```

---

## Usage

### CLI

```bash
.venv/Scripts/python interfaces/cli.py
```

Interface interactive dans le terminal. Tape ta question, `quit` pour sortir.

### Telegram

```bash
set MORRIGAN_TELEGRAM_TOKEN=ton_token_botfather
.venv/Scripts/python interfaces/telegram_bot.py
```

Commandes disponibles dans le bot :

- `/start` — présentation du projet
- `/help` — aide sur les modules
- `/reset` — efface la mémoire de conversation
- `/stats` — statistiques des modules

### API HTTP (FastAPI + SSE)

```bash
.venv/Scripts/python -m uvicorn interfaces.api:app --host 0.0.0.0 --port 8000
```

Endpoints natifs :

- `POST /query` — JSON `{"query": "...", "session_id": "..."}` → réponse
  complète + routage (`query_type`, `modules`, `domain_hint`,
  `generated_by`, `latency_s`).
- `POST /query/stream` — même body, réponse **SSE** : un event `data`
  par fragment de génération, puis un event `done` portant le routage.
- `GET /health` — `{"status": "ok", "modules": [...]}`.
- `GET /stats` — observabilité (texte + compteurs JSON).

Surface **OpenAI-compatible** (cf. `interfaces/openai_compat.py`) — pour
brancher Morrigan dans n'importe quel client OpenAI sans adaptation :

- `POST /v1/chat/completions` — `{model, messages:[...], stream}` standard.
  Le **dernier message `user`** sert de requête ; le champ optionnel `user`
  sert de `session_id` (mémoire Cauldron). Non-stream et stream (chunks
  `chat.completion.chunk` + `[DONE]`).
- `GET /v1/models` — un seul modèle (`morrigan`).
- Auth : `Authorization: Bearer <MORRIGAN_API_KEY>` (ou `X-API-Key`).

Le dagda de prod (Brigid + Ogham + Danann via `MORRIGAN_INDEX` ou
`data/knowledge` + Scáthach RWKV + Cauldron) est composé au démarrage.

#### Brancher Morrigan dans Gungnir (provider custom)

Aucune modification de code Gungnir : Réglages → Clés API → **Custom**, puis
`name=morrigan`, `base_url=http://localhost:8100`, une clé API au choix (=
`MORRIGAN_API_KEY`), `default_model=morrigan`. Gungnir détecte un provider
inconnu avec `base_url` → bascule sur son `OpenAIProvider` et appelle
`POST {base_url}/v1/chat/completions`. ⚠️ En **RAG strict**, Morrigan répond
« je ne sais pas » hors de son corpus : la richesse des réponses suit la
taille du corpus indexé.

### Tests

```bash
python -m pytest tests/
```

Lance la suite complète (orchestration, retrieval, génération, API…). Utile
pour valider une modification sans casser les cas connus.

### Ingestion de corpus

```bash
.venv/Scripts/python scripts/ingest_knowledge.py \
    --source data/knowledge \
    --backend memory
```

Ingère récursivement tous les `.md` et `.txt` du dossier donné. Backend
`memory` par défaut, `supabase` pour la persistance.

#### Corpus de documentation **code** (Phase 2B)

`scripts/ingest_code_docs.py` construit un index de docs **code** via un
**registre de sources multi-langage** (`--sources`) — chunker *code-aware*
(préserve l'indentation, garde code + explication ensemble, sous ~512 tokens e5),
index `int8` unique, métadonnée `language` par chunk. Corpus EN **interrogeable
en FR** (embedder multilingue, Phase 2A). Sources actuelles :

- **`python`** : bundle texte officiel (`docs.python.org`, tutorial/howto/faq/
  `library`) + introspection `pydoc` de modules stdlib → langage `python`.
- **`man`** : pages man locales (bash, git, grep, sed, awk, find…) — **offline,
  souverain** → langages `bash`/`git`/`shell`.

```bash
# Multi-langage : Python complet (avec la stdlib) + man (bash/git/CLI)
.venv-uv/bin/python scripts/ingest_code_docs.py \
    --sources python,man --categories tutorial,library,howto,faq \
    --output data/models/index_code

# Python scopé seul (rapide, sans library/)
.venv-uv/bin/python scripts/ingest_code_docs.py --sources python --output data/models/index_code

# Servir l'index code au runtime
MORRIGAN_INDEX=data/models/index_code .venv-uv/bin/python -m interfaces.api
```

À venir : MDN (js/html/css), PostgreSQL (sql), Docker. Docs téléchargées et
index sont **gitignorés** (régénérables via le script).

### Servir un index compressé persisté

Pour servir un gros corpus sans le réembedder à chaque démarrage, construis
un index compressé sur disque, puis pointe `MORRIGAN_INDEX` dessus :

```bash
# Corpus local → index int8 sur disque
.venv/Scripts/python scripts/build_compressed_index.py ^
    --source data/knowledge --output data/models/index_default

# … ou Wikipédia FR en streaming (Phase 5)
.venv/Scripts/python scripts/ingest_wikipedia.py --max-articles 5000

# Lancer le CLI (ou le bot) en pointant l'index
set MORRIGAN_INDEX=data/models/index_default
.venv/Scripts/python interfaces/cli.py
```

Le CLI et le bot chargent alors l'index via `Danann.load_index` (RAM
réduite, zéro réembedding). Si `MORRIGAN_INDEX` est absent ou invalide,
le runtime retombe sur l'ingestion de `data/knowledge`.

---

## Structure du projet

```
morrigan/
├── core/
│   ├── dagda.py          # Orchestrateur central
│   ├── types.py          # Types partagés (ModuleInput/Output, etc.)
│   └── pipeline.py
├── modules/
│   ├── brigid/           # LNN (classification, intuition)
│   ├── ogham/            # Moteur symbolique pyDatalog
│   ├── danann/           # Mémoire vectorielle (memory + Supabase)
│   ├── scathach/         # Génération de texte (templates Jinja2)
│   └── cauldron/         # Mémoire de travail
├── interfaces/
│   ├── cli.py
│   ├── telegram_bot.py
│   └── api.py
├── data/
│   └── knowledge/        # Corpus (.md, .txt)
├── scripts/
│   ├── ingest_knowledge.py
│   ├── ingest_wikipedia.py
│   ├── build_compressed_index.py
│   ├── train_brigid.py
│   └── supabase_schema.sql
├── config/
│   ├── morrigan.yaml
│   ├── ogham_rules.yaml
│   └── routing_rules.yaml
└── tests/
```

---

## Roadmap

### Phase 1 — Squelette fonctionnel ✅

- [x] An Dagda orchestrateur
- [x] 5 modules enregistrés et branchés
- [x] Danann retrieval hybride (cosine + lexical)
- [x] Scáthach templates Jinja2
- [x] Interfaces CLI + Telegram
- [x] Corpus d'exemple, script d'ingestion
- [x] Tests d'intégration

### Phase 2 — Indexation multi-niveaux et spécialisation ✅

- [x] Métadonnées riches par chunk (type, domaine, source, version, confiance)
- [x] Reranker cross-encoder sur les top candidats
- [x] Premier agent spécialisé : **Morrigan-Code** (Python, JS, Bash, SQL, HTML, CSS)
  - [x] Vérifieur de syntaxe par langage (`ast`, `bash -n`, `node --check`, `sqlparse`, `html.parser`, `tinycss2`)
  - [x] Corpus dédié code (6 docs FR curatés, auto-tagués `domain=code`)
- [x] Brigid entraînée (CfC ncps 384→16→6 sur 504 exemples, val_acc ~88 %)
- [x] Knowledge graph via extraction d'entités (networkx, 267 entités / 430 relations)

### Phase 3 — Génération neuronale ✅ *(cœur livré — optimisation latence à suivre)*

- [x] Scáthach v2 avec backend **RWKV** (RWKV-6 World 1.6B Q4_K via llama.cpp, CPU)
- [x] Mode RAG strict (génération ancrée sur chunks Danann + faits KG Ogham ; refus déterministe hors-corpus → **0 hallucination**)
- [x] Harnais de benchmarks + rapport (`scripts/benchmark.py`, `docs/benchmarks.md`)
- [ ] Benchmarks vs LLMs commerciaux sur domaines cibles *(à faire)*
- [ ] Latence cible : < 1s par réponse complète sur CPU — ⚠️ **non atteinte** (p50 ~12.7s sur 1.6B Q4_K CPU contraint, cf. `docs/benchmarks.md`) → chantier d'optimisation (quant plus agressive, modèle plus petit, streaming)

### Phase 4 — Corpus étendu et compression ✅ *(cœur livré)*

- [x] **Compression d'index binary/int8** : quantization vectorielle
  pure-NumPy (`Int8Index` ×4, `BinaryIndex` ×32 + recherche 2 étages),
  branchée dans Danann (`compression=int8/binary`, float32 jamais
  conservé). Recall ≥ 0.9 (int8) / ≥ 0.8 (binary two-stage).
- [x] **Index sur disque** : `Danann.save_index` / `load_index`
  (corpus.json + vectors.npz, chargement sans float32 ni réembedding)
  + `scripts/build_compressed_index.py`. Démo : 75 chunks, ×4.0.
- [x] **Recherche scalable (ANN)** : `IVFIndex` pur-NumPy (k-means +
  probes, sous-linéaire), `Danann(ann="ivf")`.
  - DiskANN/SPANN (graph-ANN natifs C++/Rust) **délibérément reportés** :
    build lourd hors philo « PC modeste, deps minimales ». L'IVF
    pur-NumPy couvre le retrieval scalable sans dépendance native.
- [ ] Ingestion Wikipedia FR à grande échelle — pipeline prêt
  (`build_compressed_index.py`, ingestion incrémentale) ; reste à
  brancher une source Wikipedia et mesurer à l'échelle.
- [ ] Matryoshka embeddings — **non retenu** : la troncature de
  dimension imposerait un embedder Matryoshka dédié et un réentraînement
  Brigid + rebuild des index à chaque ajustement. Le 2-étages
  **binary → int8** en est l'équivalent fonctionnel (recherche grossière
  puis fine) sans toucher au modèle.
- [x] Cible « < 5 Go corpus » : atteignable via compression (×4 à ×32)
  + persistance disque ; validé sur le corpus actuel, à confirmer à
  l'échelle Wikipedia.

### Phase 5 — Production

- [x] **Ingestion à l'échelle Wikipedia FR** (`scripts/ingest_wikipedia.py`,
  streaming sans dump local) + **index persisté servi au runtime**
  (`MORRIGAN_INDEX` → `Danann.load_index`, zéro réembedding au boot) +
  **IVF combiné à int8** (sous-linéaire ET compressé). Validé sur 500 art →
  37 967 chunks, chargé en 0.29 s. Cf. `docs/ingestion.md`.
- [x] **API HTTP FastAPI + SSE** (`interfaces/api.py`) : `POST /query`
  (JSON) + `POST /query/stream` (SSE, token par token via
  `process_stream`), `GET /health`, `GET /stats`. Démarrage :
  `uvicorn interfaces.api:app --host 0.0.0.0 --port 8000`.
- [x] **Dockerisation** (Dockerfile CPU `python:3.12-slim` non-root, torch
  CPU-only ; `docker-compose.yml` `8100:8000`, modèle GGUF + index + `.env`
  en volumes ; CI `docker-build.yml` build + smoke d'import). Test runtime
  `/health` reporté au déploiement VPS.
- [x] **Surface OpenAI-compatible** (`/v1/chat/completions` + `/v1/models`,
  `interfaces/openai_compat.py`) : Morrigan se branche tel quel comme provider
  custom dans Gungnir (aucun code Gungnir à toucher). Validation usage à suivre.
- [ ] Intégration Gungnir côté UX (skill/tool dédié, au-delà du provider brut)
- [ ] Monitoring et observabilité (Prometheus optionnel)

---

## Pourquoi "Morrigan" ?

La Morrigan est une déesse celtique de la guerre, de la transformation et de la
prophétie. Elle apparaît souvent sous la forme d'un corbeau et fait partie
d'une triade (Morrigan, Badb, Macha). Elle incarne l'intelligence qui voit
au-delà du visible et qui transforme le chaos en structure — une métaphore
qui colle bien à ce qu'on essaie de faire : structurer l'intelligence en
modules clairs, au lieu d'un bloc opaque et monolithique.

Chaque module porte le nom d'une figure celtique liée à sa fonction :
An Dagda le père orchestrateur, Brigid la forge et l'inspiration, Ogham
l'écriture sacrée et le savoir druidique, Danann la mère des Tuatha Dé
(les connaissances ancestrales), Scáthach la guerrière-maîtresse (qui
forme et articule), Cauldron le chaudron inépuisable (la mémoire qui
se renouvelle).

---

## Auteur

**Kevin** — dev no-code, passionné de mythologie celtique, projets IA
alternatifs. Morrigan fait partie d'une constellation de projets explorant
des architectures IA post-LLM (Ogma, Munnin, Vargr, Gungnir).

---

## Licence

**Propriétaire — tous droits réservés.** Copyright © 2026 Kevin (Scarlet Wolf).
Voir le fichier [`LICENSE`](LICENSE). Aucun usage, copie, modification ou
distribution n'est autorisé sans accord écrit préalable du titulaire.

---

## Contribuer

Projet en développement actif, en Phase 1. Les contributions, suggestions,
critiques d'architecture, et idées de corpus ciblés sont les bienvenues.

Les axes où le feedback est particulièrement utile :

- **Curation de corpus** sur les domaines cibles (code, réseau, culture générale)
- **Vérifieurs de cohérence** par langage de programmation
- **Benchmarks honnêtes** contre les LLMs commerciaux sur domaines restreints
- **Compression d'index vectoriels** (binary, Matryoshka, DiskANN)
- **Petits modèles génératifs** locaux adaptés au français
