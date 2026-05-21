# Morrigan

> Architecture IA modulaire post-LLM — un pari : atteindre la qualité
> conversationnelle des LLMs sur des domaines ciblés, en tournant localement
> sur du matériel modeste, sans GPU, sans cloud, sans hallucination.

**Statut : Phases 1, 2 et 3 livrées (cœur).** Pipeline end-to-end CPU,
6 modules + agent Morrigan-Code, classifieur LNN entraîné (Brigid CfC),
knowledge graph (Ogham), et génération neuronale RWKV avec RAG strict
(0 hallucination). Reste l'optimisation de la latence et le corpus
étendu (Phase 4). Projet de recherche appliquée, pas un produit fini.

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
- **sentence-transformers** (`all-MiniLM-L6-v2`, 384-D, ~80 Mo)
- **pyDatalog** (raisonnement symbolique)
- **Jinja2** (templates de génération — Phase 1)
- **Supabase pgvector** (persistance optionnelle)
- **python-telegram-bot v22** (interface Telegram)
- **numpy** (store en mémoire + cosine similarity)

---

## État actuel (Phases 1-3 livrées)

### Ce qui marche

- Pipeline end-to-end : input → classification → retrieval → raisonnement → génération → output
- **Classification d'intention par LNN** : Brigid (CfC ncps 384→16→6) entraînée sur 504 exemples FR, val_acc ~88 %, avec fallback heuristiques sous le seuil de confiance
- **Génération neuronale** : Scáthach v2 backend RWKV-6 1.6B Q4_K (llama.cpp, CPU), mode template en fallback
- **RAG strict (0 hallucination)** : génération ancrée sur chunks Danann + faits KG ; refus déterministe « je ne sais pas » sans contexte fiable
- **Knowledge graph** (Ogham) : 267 entités / 430 relations extraites du corpus, requêtes `compare` / `facts_about`
- **Agent Morrigan-Code** : vérification de syntaxe 6 langages (Python, Bash, JS, SQL, HTML, CSS)
- Retrieval hybride Danann : cosine + boost lexical + reranker cross-encoder, filtrage par domaine, garde anti-faux-positif
- Interfaces CLI + Telegram, backends Danann branchables (memory / Supabase pgvector), script d'ingestion
- **227 tests** (pytest), 5 workflows CI (tests, version-sync, release, brigid-train, kg-build)

### Performances mesurées

- **Retrieval/classification** : ~50-150 ms par requête (CPU, i5-10210U), empreinte ~300 Mo
- **Génération RWKV** (1.6B Q4_K, CPU contraint) : ~10-12 tok/s, latence p50 ~12.7 s pour une réponse de quelques phrases — cf. `docs/benchmarks.md`
- **0 hallucination** : 100 % de refus sur les queries hors-corpus (benchmark)

### Ce qui ne marche pas encore

- **Latence de génération élevée** : un 1.6B sur CPU contraint reste loin de la cible < 1 s. Chantier d'optimisation (quantization plus agressive, modèle plus petit, streaming) — Phase 4.
- **Pas de zero-shot** : Morrigan ne répond que depuis son corpus. Par design — fallback honnête plutôt qu'hallucination.
- **Corpus encore restreint** : quelques domaines curatés. L'ingestion massive (Wikipedia FR, compression) est l'objet de la Phase 4.
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

Le premier lancement télécharge `sentence-transformers/all-MiniLM-L6-v2` depuis
HuggingFace (~80 Mo).

### Variables d'environnement

Copier `.env.example` en `.env` et remplir au besoin :

```env
MORRIGAN_TELEGRAM_TOKEN=         # Pour l'interface Telegram
SUPABASE_URL=                    # Pour la persistance vectorielle (optionnel)
SUPABASE_KEY=
HF_TOKEN=                        # Pour éviter le rate limit HuggingFace (optionnel)
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

### Test d'intégration

```bash
.venv/Scripts/python scripts/test_phase1.py
```

Lance une batterie de 10 requêtes test sur le corpus réel et affiche latences +
réponses. Utile pour valider une modification sans casser les cas connus.

### Ingestion de corpus

```bash
.venv/Scripts/python scripts/ingest_knowledge.py \
    --source data/knowledge \
    --backend memory
```

Ingère récursivement tous les `.md` et `.txt` du dossier donné. Backend
`memory` par défaut, `supabase` pour la persistance.

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
│   ├── test_phase1.py
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

### Phase 4 — Corpus étendu et compression

- [ ] Ingestion Wikipedia FR (compression binary/int8)
- [ ] DiskANN ou SPANN pour index sur SSD
- [ ] Matryoshka embeddings pour recherche à deux temps
- [ ] Cible : < 5 Go de corpus total couvrant 90% des usages

### Phase 5 — Production

- [ ] API HTTP/gRPC stable
- [ ] Intégration avec d'autres projets (notamment Gungnir)
- [ ] Dockerisation complète
- [ ] Monitoring et observabilité

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
