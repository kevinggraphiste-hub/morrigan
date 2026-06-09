# Changelog

Toutes les modifications notables de ce projet sont consignées dans ce
fichier.

Le format suit [Keep a Changelog 1.1.0](https://keepachangelog.com/fr/1.1.0/),
et le projet respecte le versionnage sémantique [SemVer 2.0.0](https://semver.org/lang/fr/).

Les sections versionnées (`## [X.Y.Z] - YYYY-MM-DD`) sont la source de
vérité parsée par `.github/workflows/release.yml` pour générer les
notes de release. Avant de poser un tag `vX.Y.Z`, **clôturer la section
`[Non publié]` en y donnant le numéro et la date** — sinon la release
GitHub sortira sans notes (cf. mémoire `gungnir-release-changelog-gotcha`).

## [Non publié]

### Ajouté — surface API OpenAI-compatible (branchement Gungnir)
Nouveau module `interfaces/openai_compat.py`, **purement additif** (les routes
natives `/query` sont intactes ; retirer l'appel `add_openai_compat_routes`
suffit à le désactiver). Permet à tout client OpenAI standard — dont **Gungnir**
via son provider custom — de parler à Morrigan sans adaptation côté client.
- `POST /v1/chat/completions` : format Chat Completions standard (non-stream +
  stream SSE `chat.completion.chunk` → `[DONE]`). Le **dernier message `user`**
  devient la requête Morrigan ; le champ optionnel `user` → `session_id`
  (mémoire Cauldron) ; `usage` estimé (mots, le modèle local n'est pas tokenisé
  ici). Le cœur **RAG strict est préservé** (hors corpus → « je ne sais pas »).
- `GET /v1/models` : liste à un seul modèle (`morrigan`).
- Auth : `Authorization: Bearer <MORRIGAN_API_KEY>` (ce qu'envoie le SDK OpenAI)
  **ou** `X-API-Key`, et réutilise le sémaphore de concurrence existant.
- Branchement Gungnir documenté (README) : provider custom, **zéro code Gungnir**.
- +`tests/test_openai_compat.py` (10 tests : shape non-stream/stream, auth
  Bearer + X-API-Key, extraction dernier message user, `/v1/models`).

### Corrigé — 4 bugs de correction (backlog audit)
- **`KnowledgeGraph.from_dict` non idempotent** : l'ancien chargement rejouait
  `add_relation` `count` fois, ce qui couplait à tort les sources au compteur et
  **perdait toute source au-delà de `count`** (et coûtait O(count) par arête).
  Désormais restauration **verbatim** des agrégats (`count`/`confidence`/
  `sources`), avec invariant `count >= nb sources`. Round-trip stable.
- **`domain_hint` arbitraire en cas d'égalité** (`AnDagda._detect_domain_hint`) :
  à égalité de hits, le `max()` tranchait selon l'ordre du dict (arbitraire) ; ce
  domaine douteux servait ensuite de filtre dur. Une égalité ⇒ domaine **ambigu**
  ⇒ on ne contraint plus le retrieval (`None`).
- **Filtre domaine/type Danann en best-effort** : si le filtre vidait entièrement
  la fenêtre de candidats (hint erroné ou corpus sans ce domaine), le RAG strict
  produisait un **faux « je ne sais pas »**. On retombe maintenant sur les
  candidats non filtrés plutôt que de dégrader le rappel à zéro.
- **`process_stream` n'enregistrait pas la requête (/stats) si le stream levait** :
  passage en `try/finally` → la latence/le compteur sont toujours consignés, même
  quand la génération plante en cours de route (`process()` non-stream catchait
  déjà).
- +`tests/test_correction_bugs.py` : un garde-fou par bug (8 tests).

### Performance — MiniLM mutualisé entre Danann et Brigid
Danann (`EmbeddingEngine`) et Brigid (`IntentEmbedder`) chargeaient chacun leur
propre `SentenceTransformer(all-MiniLM-L6-v2)` → le même modèle **2× en RAM**.
Nouveau `core/embedder_cache.py` (cache process, double-checked locking, noms
canoniques normalisés) → **une seule instance** partagée. Levier RAM #1 sur
machine modeste.

### Docker — image CPU + compose pour l'API
Conteneurisation de l'API HTTP (Phase 5, production) :
- **`Dockerfile`** single-stage `python:3.12-slim`, non-root. **torch CPU-only**
  installé avant `requirements.txt` (pip voit `torch>=2.0` satisfait → pas de
  libs CUDA, ~plusieurs Go évités) ; llama-cpp-python CPU via l'index abetlen
  déjà dans `requirements.txt`. Aucune compilation native → pas de multi-stage.
  Lancement `python -m interfaces.api`, bind `0.0.0.0:8000` en interne.
- **`docker-compose.yml`** réécrit (le stub Phase 0 était cassé) : service
  unique `morrigan-api`, port **`8100:8000`** (Gungnir possède le 8000 hôte),
  modèle GGUF + `index_wiki` montés en **volumes** (pas bakés),
  `MORRIGAN_INDEX` servi au boot, cache HuggingFace persistant (volume nommé),
  `env_file` optionnel, healthcheck `/health` en `urllib` (pas de `curl` dans
  l'image) avec `start_period` 90s (boot = chargement RWKV 1.6B + index).
  Suppression de `version: "3.8"`, du stub redis et de `TELEGRAM_TOKEN`.
- **`.dockerignore`** : exclut `.env` (secrets non bakés), `data/` (monté en
  volume), venvs, `.git`, caches — contexte de build léger.
- **`.github/workflows/docker-build.yml`** : CI qui build l'image sur les
  runners GitHub + smoke d'import (torch CPU / llama-cpp / app) + `docker
  compose config` — valide la partie risquée du build sans Docker local ni
  VPS, à chaque changement des fichiers Docker. Le test runtime `/health`
  (modèle GGUF requis) est reporté au déploiement VPS.

## [0.5.0] - 2026-06-03

Phase 2 livrée (reranker, Morrigan-Code 6 langages, Brigid CfC, knowledge
graph, corpus code). Phase 3 livrée (génération RWKV + RAG strict +
streaming). Phase 4 livrée — corpus étendu et compression d'index.
**Phase 5 démarrée** — ingestion à l'échelle.

### Sécurité — inférence hors event-loop (lot 2/2)
Corrige le défaut **F2** de l'audit : l'inférence RWKV (synchrone, llama.cpp)
était exécutée dans la boucle asyncio → une seule génération **gelait toute
l'API** (`/health` compris, sonde Docker incluse).
- **Offload systématique** : `Scathach.process` exécute la génération
  bloquante via `asyncio.to_thread` ; `Scathach.stream` pompe le générateur
  synchrone `answer_stream` dans un thread et relaie les tokens en async via
  une queue (helper `_aiter_in_thread`). Le retrieval bloquant (embeddings)
  est lui aussi offloadé. L'event loop reste réactif pendant la génération.
- **Sécurité thread du modèle** : `RWKVBackend` sérialise désormais l'accès
  au contexte llama.cpp (unique, non concurrent-safe) via un
  `threading.Lock` autour de `generate`/`generate_stream` — l'inférence peut
  être appelée depuis plusieurs threads sans corruption (les générations se
  sérialisent, ce qui est correct sur un modèle CPU mono-contexte).
- Comportement fonctionnel **inchangé** (mêmes morceaux, même ordre, mêmes
  fallbacks) pour la CLI, Telegram et l'API. +3 tests (`_aiter_in_thread` :
  ordre, propagation d'exception, exécution hors thread principal).

### Sécurité — durcissement de l'API HTTP (lot 1/2)
Suite à l'audit du 2026-05-29, durcissement de `interfaces/api.py`
(la couche HTTP ; l'offload de l'inférence hors event-loop suivra) :
- **Concurrence bornée** : un sémaphore limite les générations RWKV
  simultanées (CPU-bound, coûteuses) ; au-delà → **HTTP 503** plutôt
  qu'une file d'attente illimitée. Configurable via
  `MORRIGAN_API_MAX_CONCURRENT` (défaut 2).
- **Bornes d'entrée (anti-DoS)** : `query` plafonnée
  (`MORRIGAN_API_MAX_QUERY_CHARS`, défaut 4000) et `session_id` bornée en
  longueur + charset (`^[A-Za-z0-9._-]+$`) → rejet **422**.
- **SSE robuste** : la génération s'**arrête si le client se déconnecte**
  (`request.is_disconnected()`) au lieu de continuer dans le vide.
- **Erreurs non divulguantes** : message générique renvoyé au client
  (plus de `str(exc)` exposant chemins/détails internes) ; le détail est
  loggé côté serveur. `/query` non-stream encapsule aussi ses erreurs en
  500 générique.
- **Auth optionnelle par clé API** : si `MORRIGAN_API_KEY` est défini,
  `/query`, `/query/stream` et `/stats` exigent l'en-tête `X-API-Key`
  (→ 401 sinon). `/health` reste ouvert (sonde de vivacité).
- **Bind 127.0.0.1 par défaut** (`main()`) : exposer sur le réseau devient
  un choix explicite (`MORRIGAN_API_HOST=0.0.0.0`). Variables documentées
  dans `.env.example`. +4 tests (longueur, charset, clé API, 503).

### Supprimé
- **Dépendances mortes** retirées de `requirements.txt` : `markovify`
  (jamais importé — Scáthach génère via Jinja2/RWKV, pas de chaîne de
  Markov), `redis` (aucun usage), et la ligne `asyncio` (stdlib, n'a rien
  à faire dans les requirements). `supabase` conservé (backend pgvector
  optionnel, non-défaut).
- **Scripts smoke pré-pytest** supprimés (`scripts/test_phase1.py`,
  `scripts/test_pipeline.py`, `scripts/hello_lnn.py`,
  `scripts/hello_danann.py`, `scripts/hello_ogham.py`) : démos/tests
  manuels Phase 0 redondants avec la suite `tests/` (31 fichiers). README
  mis à jour (section « Tests » → `pytest`, arborescence `scripts/`).

### Corrigé
- **Embeddings Danann normalisés L2** (`modules/danann/embeddings.py`) :
  `EmbeddingEngine.encode` passe désormais `normalize_embeddings=True`.
  Tout le module (quantization int8/binary, ANN IVF, `store` mode `none`)
  suppose des vecteurs de norme 1 pour assimiler produit scalaire et
  cosinus, mais l'encodeur ne normalisait pas — biaisant **silencieusement**
  le ranking RAG vers les chunks de grande norme dans les chemins
  compressés, et désalignant Danann de Brigid (qui normalisait déjà).
  Le mode `none` de `store.search` est simplifié en conséquence (produit
  scalaire direct, plus de recalcul des normes du corpus à chaque requête).
  +1 test garde-fou (`test_embeddings_are_l2_normalized`).

### Ajouté — Phase 5 (mise en production)
- **API HTTP FastAPI + SSE** (`interfaces/api.py`) : `POST /query`
  (JSON in/out) renvoie la réponse complète + le routage (type, modules,
  domain_hint, generated_by, latence). `POST /query/stream` streame la
  génération **token par token via Server-Sent Events** par-dessus
  `AnDagda.process_stream` (latence perçue alignée sur la CLI/Telegram),
  termine par un event `done` portant le routage. `GET /health` liste
  les modules opérationnels ; `GET /stats` expose le format texte
  observabilité **et** les compteurs JSON. Dagda de prod composé au
  startup via le lifespan (Brigid + Ogham + `build_danann` + Scáthach
  RWKV + Cauldron) ; `create_app(dagda=...)` permet d'injecter un dagda
  factice pour les tests. Tests via `httpx.AsyncClient` + `ASGITransport`
  (zéro réseau, CI-safe). Démarrage : `uvicorn interfaces.api:app`.
  +6 tests. Nouvelles deps : `fastapi`, `uvicorn[standard]`.

### Ajouté — Phase 5 (ingestion à l'échelle)
- **ANN IVF combiné à la compression** (`IVFIndex.build_from_int8`) :
  `Danann(ann="ivf", compression="int8"|"binary")` est désormais
  possible (la contrainte `ann="ivf" ⇒ compression="none"` est levée).
  Les cellules k-means sont calculées sur les vecteurs **déquantizés à
  la volée** (transitoire, jeté) ; on ne conserve que centroïdes +
  listes, et le re-score des candidats se fait sur les **codes int8** →
  **zéro float32 matérialisé**. Recherche sous-linéaire ET compressée.
  Validé sur l'index Wikipédia réel (37 967 chunks) : 194 cellules,
  ~23 % du corpus scanné, top-1 pertinent, `vectors=None`. Top-1
  IVF+int8 == flat float sur requêtes nettes. +4 tests (l'ancien
  garde-fou « ivf interdit si compressé » est remplacé).
- **Validation de l'ingestion à l'échelle** (`docs/ingestion.md`) : run
  réel sur 500 articles Wikipédia FR → **37 967 chunks**, index int8
  **14.7 MB (×4.0** vs float32), 27 MB sur disque. **Chargé au runtime
  en 0.29 s** (zéro réembedding) et servi via `MORRIGAN_INDEX` ;
  retrieval ancré (top-1 Wikipedia pertinent), ~130 ms à chaud. Mesures
  + lecture honnête (build CPU-bound ; `flat` O(n) → ANN IVF à combiner
  avec int8 au-delà de ~100 k chunks) documentées.
- **Index persisté chargé au runtime** (`core/knowledge.build_danann`) :
  CLI et bot chargent un index compressé sur disque via
  `Danann.load_index` quand `MORRIGAN_INDEX` (ou `index_path`) pointe
  vers un dossier valide (`corpus.json` + `vectors.npz`) → gros corpus
  servi avec RAM réduite, **zéro réembedding** au démarrage. Sinon,
  fallback gracieux sur l'ingestion de `data/knowledge`. C'est le
  **consommateur runtime** des index produits par
  `build_compressed_index.py` / `ingest_wikipedia.py`. Logique
  centralisée et partagée CLI/bot — corrige au passage un **trou : le
  CLI n'ingérait rien** (Danann vide → refus en RAG strict).
  `MORRIGAN_INDEX` documenté dans `.env.example`. +6 tests.
- **Ingestion Wikipédia FR à l'échelle** (`scripts/ingest_wikipedia.py`) :
  **stream** le dataset `wikimedia/wikipedia` (sans télécharger les
  ~20 Go du dump), chunke les articles, les indexe dans un Danann
  **compressé** (int8 par défaut, compression Phase 4) et persiste
  l'index sur disque (rechargé sans réembedder). Filtre les articles
  trop courts (`MIN_ARTICLE_CHARS`), borne via `--max-articles`,
  indexation par lots (`--batch`), choix de compression
  (`none`/`int8`/`binary`). **Tolérance aux pannes** : sur erreur
  réseau/dataset, le lot en cours est *flushé* puis sauvé (rien n'est
  perdu) au lieu d'être abandonné. Validé en réel : 5 articles FR →
  170 chunks, index int8 **×4.0**. +8 tests CI-safe (monkeypatch de
  `_iter_articles` + faux module `datasets` → zéro réseau).
- `datasets` ajouté à `requirements.txt` (dépendance Phase 5, utilisée
  uniquement par le script d'ingestion).

### Ajouté — Phase 4 (corpus étendu et compression)
- **Index ANN IVF sous-linéaire** (`modules/danann/ann.py`, pure
  NumPy) : partitionne le corpus en cellules (k-means) ; à la requête,
  ne sonde (`n_probe`) que les cellules les plus proches → recherche
  **sous-linéaire** sans dépendance native. Recall ≥ 0.8 avec assez de
  probes ; petit corpus (≤4 cellules) sondé entièrement (exact).
  `Danann(ann="ivf")` (avec `compression="none"`) : IVF bâti
  paresseusement, invalidé à chaque `index()`, partage le tableau
  float (zéro copie). **Choix assumé vs DiskANN/SPANN** (graph-ANN à
  build C++/Rust lourd) : l'IVF pur-NumPy couvre le retrieval scalable
  sur matériel modeste sans build natif. +14 tests.
- **Persistance disque de l'index compressé** (`Danann.save_index` /
  `Danann.load_index`) : sauve `corpus.json` (chunks + metadata +
  config) et `vectors.npz` (codes quantizés). Le chargement
  reconstruit l'index **sans réembedder ni matérialiser de float32**
  → gros corpus servi avec une RAM réduite. `scripts/
  build_compressed_index.py` : ingère un répertoire → index compressé
  sur disque (ingestion incrémentale fichier par fichier). Démo sur
  le corpus actuel : 75 chunks, index int8 **28 KB vs 112 KB float32
  (×4.0)**. +8 tests. GGUF/NPZ et dossiers `index*/` gitignorés.
- **Compression branchée dans Danann** : option `compression` au
  constructeur — `none` (float32, défaut, inchangé), `int8` (codes
  par-vecteur, **~4× moins de RAM**), `binary` (Hamming coarse + int8
  rerank, **~4.5× moins**). En mode compressé, le float32 n'est
  **jamais conservé** (quantization par lot, incrémentale). La
  recherche garde le boost lexical (sur la fenêtre de candidats), le
  filtrage par domaine/type et le reranker cross-encoder. `top-1`
  compressé == `top-1` exact sur des requêtes nettes (testé).
  `memory_bytes()` + `index_memory_bytes` dans capabilities. +13 tests.
- **`Int8Index` par-vecteur + `extend`** (incrémental) et
  `BinaryIndex.extend` ajoutés au module quantization.
- **Quantization vectorielle** (`modules/danann/quantization.py`,
  pure NumPy) : compresse l'index d'embeddings pour tenir un gros
  corpus sur PC modeste.
  - `Int8Index` : quantization scalaire symétrique → **4× plus
    compact** que float32, recall ≥ 0.9 (quasi sans perte).
  - `BinaryIndex` : 1 bit/dimension (signe) → **32× plus compact**,
    recherche par distance de Hamming. Filtre grossier.
  - `two_stage_search` : filtre binaire élargi (k × 16) → re-score
    float fin → recall ≥ 0.8 pour 32× moins de mémoire en coarse.
  - `exact_search` + `recall_at_k` (référence/mesure). +12 tests.

### Ajouté — Phase 3 (génération neuronale)
- **Observabilité `/stats`** : An Dagda accumule des compteurs (nb
  requêtes, répartition par `query_type`, par `generated_by` rwkv/
  template, latence moyenne) et une **trace de la dernière requête**
  (type, raison de routage, modules activés, domaine, `generated_by`,
  latence, **classification Brigid + top-3 probas**). `format_stats()`
  rend tout ça ; commande `/stats` (ou `stats`) en CLI et `/stats`
  Telegram. Scáthach expose `last_generated_by` pour tracer le chemin
  même en streaming. +6 tests. Idéal pour debug + démo.
- **Chargement auto du `.env`** (`core/env.py` → `load_env`) : plus
  besoin de coller le token Telegram (ni les clés Supabase/HF) à chaque
  lancement. Câblé dans la CLI et le bot Telegram. Dégradation
  gracieuse (no-op si python-dotenv absent ou .env manquant) ;
  `override=False` (l'env réel l'emporte sur le fichier). `.env`
  gitignoré, `.env.example` documente les clés. +4 tests.
- **Streaming Telegram** : le bot affiche la réponse au fil de l'eau
  via **édition progressive du message** (curseur `▌`), throttlée à
  ~1 édition/s (anti flood-control Telegram). Helper testable
  `stream_collect(pieces, edit, interval, clock)` qui accumule et
  édite, avale les erreurs d'édition ("message not modified", flood),
  garantit une édition finale. Débordement >4000 car. géré (1er bloc
  édité, reste envoyé en messages séparés). Scáthach passe en backend
  `rwkv` côté bot. `cmd_help` mis à jour (RWKV + RAG strict). +8 tests.
- **Optim prefill (contexte RWKV 4→2 chunks)** : mesure → le prefill
  RWKV est ~linéaire en longueur de prompt. Réduire le contexte de 4 à
  2 chunks (les mieux classés par Danann) fait chuter le
  **time-to-first-token de ~2.4s à ~1.4s p50** sans perte d'ancrage
  (toujours 100 % sur le benchmark). `n_threads` confirmé optimal à 4
  (8 = -30 % à cause de l'hyperthreading). Le benchmark mesure et
  rapporte désormais le **TTFT** (latence ressentie) en plus de la
  latence totale ; `docs/benchmarks.md` régénéré.
- **Streaming de génération (optimisation latence ressentie)** : la
  réponse RWKV s'affiche token par token au lieu d'attendre la
  génération complète. Diagnostic latence : sur la machine de dev
  (i5-10210U, RAM saturée — 2.6 Go de swap au repos), la lenteur vient
  surtout du matériel (RAM starvation + CPU U-series), pas du code ;
  ~10-12 tok/s est normal pour un 1.6B Q4 sur ce chip. Le streaming
  attaque le *ressenti* : **1er token en ~2.4s** au lieu de ~20s
  d'écran vide.
  - `RWKVBackend.generate_stream` / `answer_stream` (llama.cpp stream=True).
  - `Scathach.stream()` (async generator) : stream RWKV si contexte,
    sinon yield template/refus en un bloc. Helper `_rwkv_context`
    partagé avec `_generate_rwkv`.
  - `AnDagda.process_stream()` : exécute les modules amont normalement
    puis streame le dernier (Scáthach).
  - CLI (`interfaces/cli.py`) : affichage live token par token, et
    Scáthach passe en backend `rwkv` (fallback template si modèle absent).
  - +11 tests (backends factices).
- **Benchmarks de génération (PR D)** : `scripts/benchmark.py` complet
  (remplace le placeholder) + rapport `docs/benchmarks.md` avec chiffres
  réels mesurés sur la machine. Mesure : latence p50/p95/moyenne/max,
  **taux de refus hors-corpus** (cible 100% — 0 hallucination), **taux
  d'ancrage** (recouvrement lexical réponse↔contexte), comparaison à la
  cible README < 1 s. Premiers résultats (RWKV-6 1.6B Q4_K, CPU
  contraint) : **refus 100%, ancrage 100%, latence p50 ~12.7 s** —
  la cible < 1 s n'est PAS atteinte (documenté honnêtement, pistes
  d'optimisation listées : quant plus agressive, modèle plus petit,
  streaming). +10 tests de harnais (backend factice).
- **RAG strict (PR C) — le « 0 hallucination » de Morrigan** :
  `Scathach(strict_rag=True)` (défaut). En génération RWKV :
  - **Refus déterministe sans contexte** : si aucun chunk Danann
    pertinent ni fait Ogham n'est disponible, Scáthach **n'appelle pas
    le LLM** et renvoie un « [Morrigan] Je n'ai pas d'information… »
    via template. Pas d'invention possible.
  - **Génération ancrée avec contexte** : le prompt instruit RWKV de
    répondre UNIQUEMENT à partir du contexte fourni et de dire « Je ne
    sais pas » sinon (`RWKVBackend.format_prompt(strict=True)`).
  - **Contexte enrichi par le KG** : `_ogham_context()` convertit les
    `compare` / `facts` d'Ogham (PR #13) en lignes FR (prédicats
    humanisés : is_a → « est », uses → « utilise »…) et les ajoute aux
    chunks Danann dans le contexte RAG.
  - `strict_rag=False` pour un mode génération libre (sans grounding).
  - Smoke validé : « recette du cassoulet ? » sans corpus → refus sans
    appel LLM ; « TCP est-il fiable ? » + chunk → réponse ancrée sur
    le chunk. +14 tests.
- **Scáthach branché sur RWKV (PR B)** : `Scathach.process()` supporte
  3 backends — `template` (défaut, Jinja2), `rwkv` (génération RWKV),
  `auto` (alias rwkv). En mode rwkv, assemble le contexte depuis les
  chunks Danann pertinents (RAG souple) et génère via `RWKVBackend`.
  **Fallback gracieux** : si RWKV indisponible ou échoue, retombe sur
  les templates — zéro régression. La vérification de code reste sur
  template (sortie structurée). `metadata["generated_by"]` trace le
  chemin réellement emprunté. Constructeur accepte un `rwkv_backend`
  injectable (tests). +13 tests de wiring (backend factice, déterministe).
- **`modules/scathach/rwkv_backend.py`** : backend de génération RWKV
  via llama.cpp (`llama-cpp-python`, wheel CPU prebuilt — pas de build
  cmake). Modèle RWKV-6 World 1.6B quantizé GGUF (défaut Q4_K ~993 Mo).
  `RWKVBackend` : lazy load, `is_available()` pour le fallback,
  `format_prompt` (format RWKV World `User:/Assistant:`, injection de
  contexte RAG optionnelle), `generate` / `answer` avec defaults validés
  (`repeat_penalty=1.3` indispensable, sinon RWKV boucle). Dégradation
  gracieuse : sans lib ni modèle, le backend est indisponible et
  Scáthach (PR B) retombera sur ses templates Jinja2.
  **Validé en local** : génère du français cohérent à ~10-12 tok/s sur
  CPU contraint. Q2_K testé mais trop agressif (sortie dégénérée) → Q4_K
  est le plancher de qualité.
- **`scripts/fetch_rwkv_model.py`** : télécharge le GGUF depuis HF
  (`--quant` configurable). Le `.gguf` est gitignoré (option B,
  artefact reproductible).
- **`tests/test_rwkv_backend.py`** : 11 tests (format prompt, config,
  dégradation gracieuse modèle absent ; + 2 smoke de génération réelle
  gated par présence du GGUF + `importorskip llama_cpp`).

### Modifié
- `requirements.txt` : ajoute `llama-cpp-python` via l'index de wheels
  CPU prebuilt abetlen (`--extra-index-url`).
- `.gitignore` : ignore `data/models/*.gguf`.

### Ajouté
- **Reranker cross-encoder** (`modules/danann/reranker.py`) sur les
  top candidats du retriever Danann, avec métadonnées riches par chunk
  (type, domaine, source, version, confiance) et filtrage par domaine.
- **Module Morrigan-Code** (`modules/morrigan_code/`) — premier agent
  spécialisé. Vérifieurs syntaxiques par langage :
  - **Python** via stdlib `ast` (imports, fonctions sync + async,
    classes, `if __name__ == "__main__"`).
  - **Bash** via `bash -n` (fonctions, shebang).
  - **JavaScript** via `node --input-type=module --check` (fonctions,
    classes, imports ESM/CJS).
  - **SQL** via `sqlparse` (structure — pas validation stricte par
    dialecte, limitation assumée).
  - **HTML** via stdlib `html.parser` + suivi de pile (balises non
    fermées, fermetures orphelines, void elements).
  - **CSS** via `tinycss2` (erreurs de parse CSS3, at-rules).
  Registry avec alias usuels (py, sh/shell, js/node). Skip propre des
  langages non encore supportés (Rust, TypeScript, …).
- **Routage code** dans An Dagda : détection d'un fence markdown
  ` ```lang ... ``` ` dans la query → `QueryType.CODE` →
  pipeline `[morrigan_code, scathach]`.
- **Template `code_verification.j2`** dans Scáthach pour rendre la
  vérification Morrigan-Code (blocs valides, erreurs, structure).
- **`LICENSE` propriétaire — tous droits réservés** (Copyright © 2026
  Kevin / Scarlet Wolf) en remplacement du « À déterminer » du README.
- **CI pytest** (`.github/workflows/tests.yml`) — Python 3.12,
  installe `requirements.txt` + pytest, cache pip, tourne sur chaque
  PR et chaque push sur `main`.
- **`CHANGELOG.md`** (ce fichier) + workflow `version-sync-check.yml`
  qui vérifie la cohérence entre `core/__version__.py`, le tag posé et
  la première section versionnée de ce changelog.
- **`release.yml`** déclenché sur tag `v*`, qui extrait la section
  CHANGELOG correspondante et crée la GitHub Release.
- **Dataset de classification d'intention Brigid** : 504 exemples
  curatés en français dans `data/training/intent_classification.jsonl`,
  équilibrés 84 × 6 classes (`factual`, `reasoning`, `creative`,
  `conversation`, `complex`, `code`), 0 doublon. Couvre la borne haute
  de la roadmap « Brigid entraînée (CfC 200-500 ex) ».
- **`modules/brigid/dataset.py`** : `LabeledExample`, `load_dataset`,
  `split_train_val` *stratifié* et déterministe (chaque classe est
  garantie présente en train et en val), `class_balance`. Ordre des
  `LABELS` gelé pour stabilité des checkpoints futurs.
- **`modules/brigid/embedder.py`** : wrapper singleton autour de
  `sentence-transformers/all-MiniLM-L6-v2` (même modèle que Danann →
  cache HF partagé). Force CPU (cohérent avec la philo « PC modeste »).
- **`modules/brigid/model.py`** : `IntentClassifier` réel basé sur
  `ncps.torch.CfC` (Liquid Time-Constant Closed-form). Architecture
  compacte : 384-D → CfC(16) → Linear(6) ≈ **60 K paramètres**,
  checkpoint ~240 KB. Helpers `save_checkpoint` / `load_checkpoint`
  avec métadonnées (input_dim, hidden_dim, labels, embed_model_name,
  accuracy) et garde-fous : refuse de charger un checkpoint dont
  l'embedder ou l'ordre des LABELS ne correspond plus.
- **`scripts/train_brigid.py`** : entraînement complet (load JSONL →
  embed → split stratifié → AdamW + CrossEntropy → eval val ↔ best
  state → save). Métriques par classe. Code retour non nul si
  `val_accuracy < --min-accuracy` (0.65 par défaut) — utilisé comme
  garde-fou CI. Premier run réel : **val_acc 0.882** sur 80 époques /
  402 train / 102 val, 7s sur CPU.
- **`.github/workflows/brigid-train.yml`** : réentraîne le CfC sur
  chaque PR/push touchant dataset, modèle, embedder ou script de
  training. Cache HuggingFace Hub (MiniLM ~80 MB). Échec dur si
  val_accuracy < 0.65. Checkpoint uploadé en artefact GitHub
  (rétention 14 j) pour debug.
- **`tests/test_brigid_model.py`** : 13 tests CfC — construction,
  déterminisme par seed, forward shapes, save/load roundtrip, refus
  de checkpoints incompatibles (embedder, labels).
- **Brigid wirée dans An Dagda (PR C)** :
  - `Brigid.classify_intent(query)` (sync) — encode + forward + softmax
    → `IntentClassification(label, confidence, probabilities)`. Lazy-load
    du checkpoint au 1er appel. Renvoie `None` si checkpoint absent
    (dégradation gracieuse, pas d'exception).
  - `Brigid.process()` (async) délègue à `classify_intent` et renvoie
    un `ModuleOutput` cohérent — `phase=2` quand le modèle est chargé,
    `errors` non vide sinon.
  - `An Dagda.classify_query()` consulte Brigid en premier ; au-dessus
    de `BRIGID_CONFIDENCE_THRESHOLD = 0.5`, route via la prédiction
    LNN ; sinon (Brigid absent, checkpoint manquant, ou confidence
    faible) → fallback heuristiques mots-clés. Le shortcut fence
    markdown garde la priorité absolue.
  - Mapping unifié `_ROUTING_MAP: Dict[QueryType, List[str]]` —
    source de vérité partagée entre routages Brigid et heuristiques,
    évite les divergences silencieuses.
- **`tests/test_brigid_inference.py`** : 11 tests d'intégration
  (classify_intent shapes/confidence, process success/dégradé,
  health_check, An Dagda utilise Brigid quand confiant, fallback
  quand faible/absent, fence markdown garde priorité, mapping complet).
  Fixture session-scopée : réutilise `data/models/brigid_cfc.pt` s'il
  existe, sinon entraîne un mini-modèle (20 époques) à la volée.
- **`modules/ogham/knowledge_graph.py` (PR 1 du chantier KG)** :
  fondations du knowledge graph Ogham. Modèle de données
  (`Entity`, `Relation`) + `KnowledgeGraph` backé par `networkx.DiGraph`
  (déjà dans les deps). API de requête : `add_entity`/`add_relation`
  (avec agrégation des duplicates : count, max confidence, sources
  cumulées), `entities`/`relations` (filtres optionnels), `neighbors`,
  `facts_about`, `compare` (points communs, différences, relations
  directes). Persistance JSON versionnée (schema_version=1), lisible
  et diff-friendly. Pas encore d'extraction (PR 2), pas d'intégration
  Ogham (PR 4). +22 tests.
- **`modules/ogham/extractor.py` (PR 2 du chantier KG)** : extraction
  d'entités et relations depuis du texte français, pure-Python (zéro
  dep NLP externe). `slugify` pour ids stables (sans accents, lowercase),
  `split_sentences` basique sur `[.!?]` + maj. `extract_entities` détecte
  Title-Cased multi-mots + acronymes (TCP, HTTP, ADN…), filtre les
  stopwords usuels (déterminants, conjonctions). `extract_relations` :
  5 patterns FR (`est un` → `is_a`, `possède` → `has`, `utilise` →
  `uses`, `de` → `of`, co-occurrence par phrase → `co_occurs_with`
  confidence 0.3). `populate_graph(kg, text, source)` ingère dans le
  KG en respectant l'agrégation. **Sur le corpus actuel (4 docs) :
  129 entités, 282 relations**, `compare("tcp", "udp")` capture déjà
  les bonnes distinctions (`protocole` commun, `fiable`/`rapide`
  distinctifs). +27 tests.
- **`scripts/build_knowledge_graph.py` (PR 3 du chantier KG)** :
  ingère un corpus (Markdown/text récursif) → `data/models/
  knowledge_graph.json` versionné. Pré-traitement markdown (drop
  code fences, headers, bullets, flatten links) avant extraction.
  Garde-fous `--min-entities` / `--min-relations` pour échouer
  proprement si le corpus est vidé ou l'extracteur cassé. Top-N
  entités affichées pour validation à l'œil. CLI déterministe, exit
  non nul sur seuil non atteint. Build local : **129 entités,
  228 triplets, 71 KB, 0.0s** sur les 4 docs actuels. +11 tests.
- **`.github/workflows/kg-build.yml`** : reconstruit le KG sur chaque
  PR/push touchant corpus, extracteur, knowledge_graph ou le script.
  Échec si en dessous des seuils. KG uploadé en artefact GitHub
  (rétention 14 j) pour debug. Symétrique au workflow `brigid-train`.
- **Ogham branché sur le KG (PR 4 du chantier KG)** :
  `Ogham.process()` charge le KG en lazy au 1er appel et l'utilise
  pour enrichir les réponses. Détection des requêtes via patterns FR :
  - `Compare X et Y` / `Différence entre X et Y` / `X vs Y` →
    `kg.compare(a, b)` → structure `compare` avec `common_neighbors`,
    `a_only`, `b_only`, `direct_relations` (JSON-safe pour Scáthach).
  - `Qu'est-ce que X` / `C'est quoi X` / `Définition de X` /
    `Parle-moi de X` / `Qui est X` → `kg.facts_about(slug(X))` →
    structure `facts` avec liste de relations.
  - Résolution d'entités multi-candidats (slug complet + mots
    individuels) pour absorber « les protocoles TCP » → `tcp`.
  - Dégradation gracieuse : sans KG dispo (corpus pas indexé,
    `kg.json` absent), `compare`/`facts` restent `None` et Ogham
    fonctionne comme avant (templates + chunks). `metadata.kg_loaded`
    + `kg_load_error` exposent l'état pour debug.
  - Contrat stable : `result["type"]` vaut désormais
    **`"structured_response"`** (et plus `comparison`/`explanation`
    selon le sous-type — déplacé sous `result["structure_type"]`).
- **`tests/test_ogham_kg_integration.py`** : 12 tests E2E. Mini-KG
  TCP/UDP/IP/Brigid en fixture pour tests autonomes (pas de dépendance
  au build sur disque). Couvre dégradation gracieuse, compare avec/
  sans match, patterns FR multiples, capabilities.

### Réparé
- **`tests/test_ogham.py::test_ogham_process`** : **xfailed depuis le
  commit initial cdc66ce, passe enfin**. Le contrat stable
  `result["type"] == "structured_response"` est désormais respecté.
  Marker `@pytest.mark.xfail` retiré.
- **`tests/test_scathach.py::test_scathach_template_generation`** :
  **dernier xfailed du repo, résolu en PR B Phase 3**. `not_found.j2`
  préfixe désormais `[Morrigan]` (Morrigan s'identifie quand elle ne
  sait pas). **La suite n'a plus aucun xfailed** (208 passed).
- **Régression PR #13 (KG) corrigée** : `Scathach._extract_structure_type`
  lisait `ogham_result["type"]`, devenu `"structured_response"` depuis
  l'intégration KG → Scáthach tombait toujours sur `factual.j2`. Lit
  désormais `structure_type` (avec rétrocompat sur l'ancien `type`).

### Boucle Phase 2
- **`data/knowledge/code_*.md` — corpus dédié code (6 fichiers FR
  curatés)** : Python (boucles, comprehensions, decorateurs, async),
  JavaScript (let/const/arrow, promises, ESM vs CJS, closures), Bash
  (variables, conditions, fonctions, pipes, trap), SQL (SELECT, JOIN,
  CTE, index, EXPLAIN), HTML/CSS (sémantique, accessibilité, flexbox
  vs grid, srcset), Git/Docker (workflow, stash, multi-stage, compose).
  Chaque fichier ≥ 500 caractères, structure markdown propre, auto-
  tagué `domain="code"` par `scripts/ingest_knowledge.py` (existant).
  **Effet immédiat sur le KG** : 129 → **267 entités**, 228 → **430
  triplets** distincts (+138 / +202). Nouvelles entités top : SELECT
  (13 relations), JOIN (12). +19 tests garde-fous (nb fichiers,
  domain, taille, structure markdown).

### Modifié
- `core/types.py` : ajout `QueryType.CODE`.
- `.gitignore` : ajoute `data/models/*.json` (KG construit, option B
  comme pour `.pt` Brigid) et `.venv-uv/` / `.venv-*/` (envs uv).
- `tests/test_brigid.py` : mis à jour pour le nouveau contrat Brigid
  (phase 0 → 1 sans checkpoint, 2 chargé ; mode dégradé sans
  exception). Les vraies validations d'inférence sont dans
  `test_brigid_inference.py`.
- `tests/test_ogham.py::test_ogham_process` et
  `tests/test_scathach.py::test_scathach_template_generation` marqués
  `@pytest.mark.xfail(strict=False)` — pré-existants depuis le commit
  initial, documentés en code plutôt que re-vérifiés à chaque session.

## [0.1.0] - 2026-05-07

Phase 1 — squelette fonctionnel. Premier jalon SemVer. Pipeline end-to-
end opérationnel sur corpus d'exemple, mesuré sur i5-10210U CPU.

### Ajouté
- **An Dagda** orchestrateur (routing par mots-clés et heuristiques,
  normalisation accents, signaux interrogatifs).
- **Brigid** squelette LNN/CfC (modèle non entraîné, hooks prêts).
- **Ogham** moteur symbolique de base (pyDatalog chargé).
- **Danann** retrieval hybride : cosine similarity + boost lexical sur
  tokens rares, garde anti-faux-positif, backends `memory` et
  `Supabase pgvector` branchables.
- **Scáthach** génération par templates Jinja2 (factuel, comparaison,
  explication, conversation, not_found) avec nettoyage markdown des
  chunks.
- **Cauldron** mémoire de travail (historique session-based).
- **Interfaces** CLI + Telegram fonctionnelles + ébauche API HTTP.
- **Scripts** d'ingestion automatique (`scripts/ingest_knowledge.py`)
  et de smoke test (`scripts/test_phase1.py`).
- **Corpus d'exemple** : 28 chunks sur 4 domaines (réseau, mythologie
  celtique, IA, projet).
- **Tests d'intégration** end-to-end.
- **Versionnage SemVer** via `core/__version__.py` (`0.1.0`) et tag
  annoté `v0.1.0`.

### Performances mesurées
- Latence moyenne : **~50-150 ms par requête** (CPU, i5-10210U).
- Empreinte RAM : **~300 Mo** (embeddings + numpy + code).
- Précision factuelle : **10/10** sur les 10 requêtes du corpus de
  test après les correctifs de normalisation et de garde.
