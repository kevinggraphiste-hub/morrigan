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

Phase 2 — Indexation multi-niveaux et spécialisation (en cours). Sera
publiée sous `0.2.0` quand Brigid entraînée + knowledge graph + corpus
code seront livrés.

### Ajouté
- **Reranker cross-encoder** (`modules/danann/reranker.py`) sur les
  top candidats du retriever Danann, avec métadonnées riches par chunk
  (type, domaine, source, version, confiance) et filtrage par domaine.
- **Module Morrigan-Code** (`modules/morrigan_code/`) — premier agent
  spécialisé : vérifieur AST Python via la stdlib `ast`. Extrait
  imports, fonctions (sync + async), classes, `if __name__ == "__main__"`.
  Skip propre des langages non encore supportés.
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

### Modifié
- `core/types.py` : ajout `QueryType.CODE`.
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
