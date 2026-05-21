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

Phase 2 livrée (reranker, Morrigan-Code 6 langages, Brigid CfC, knowledge
graph, corpus code). Phase 3 livrée (génération RWKV + RAG strict +
streaming). **Phase 4 démarrée** — corpus étendu et compression d'index.

### Ajouté — Phase 4 (corpus étendu et compression)
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
