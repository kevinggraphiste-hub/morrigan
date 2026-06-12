"""
Assemblage du Danann servi par le runtime (CLI, bot Telegram).

Deux modes, dans cet ordre de priorité :

1. **Index compressé persisté** — si `index_path` (ou la variable d'env
   `MORRIGAN_INDEX`) pointe vers un dossier valide (`corpus.json` +
   `vectors.npz`), on le charge via `Danann.load_index`. Gros corpus
   servi avec une RAM réduite, **zéro réembedding** au démarrage. C'est
   le consommateur runtime des index produits par
   `scripts/build_compressed_index.py` et `scripts/ingest_wikipedia.py`
   (Phases 4-5).

2. **Ingestion à la volée** (comportement historique) — sinon, Danann en
   mémoire + ingestion de `data/knowledge` (réembedding du corpus curaté
   à chaque lancement).

Centralisé ici pour que CLI et bot partagent exactement la même logique
(avant, le CLI n'ingérait rien → Danann vide).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from modules.danann.store import Danann
from scripts.ingest_knowledge import ingest_directory

logger = logging.getLogger("morrigan.knowledge")

DEFAULT_KNOWLEDGE_DIR = Path("data/knowledge")
INDEX_ENV = "MORRIGAN_INDEX"
ANN_ENV = "MORRIGAN_ANN"               # "flat" (défaut) | "ivf"
IVF_PROBES_ENV = "MORRIGAN_IVF_PROBES"  # entier, optionnel (défaut ~C/8)
RERANKER_ENV = "MORRIGAN_RERANKER"      # "on" | "off" (défaut)
SHARD_BY_ENV = "MORRIGAN_SHARD_BY"      # clé de métadonnée (ex. "language")


def _is_valid_index(path: Path) -> bool:
    """Un dossier d'index est valide s'il contient les deux fichiers."""
    return (path / "corpus.json").exists() and (path / "vectors.npz").exists()


def _retrieval_opts(use_reranker: bool | None) -> dict:
    """Options retrieval du runtime, résolues depuis l'env (audit 2026-06-12).

    - Reranker **OFF par défaut** : le cross-encoder ms-marco (anglais)
      n'apporte pas de gain fiable mesuré en FR, pour ~300 ms+ par requête
      (cf. docs/audit-retrieval-2026-06-12.md). `MORRIGAN_RERANKER=on`
      pour le réactiver (ré-évaluation Phase 2D).
    - `MORRIGAN_ANN=ivf` active la recherche sous-linéaire (IVF),
      `MORRIGAN_IVF_PROBES` règle le compromis recall/latence.
    - Mini-RAG fragmenté **ON par défaut** (`MORRIGAN_SHARD_BY=language`) :
      recherche restreinte au shard routé (centroïde top-1, repli
      monolithique si le routeur hésite). Gain mesuré = qualité sur le
      corpus code (15/16 vs 13/16, corrige les pièges cross-langage type
      « tableau » FR). Sans risque sur les autres index : <2 valeurs de
      `language` ou index non-int8 → désactivation propre par Danann.
      `MORRIGAN_SHARD_BY=off` (ou `none`) pour désactiver explicitement.
    """
    if use_reranker is None:
        use_reranker = os.environ.get(RERANKER_ENV, "off").strip().lower() == "on"
    ann = os.environ.get(ANN_ENV, "flat").strip().lower() or "flat"
    if ann not in ("flat", "ivf"):
        logger.warning("%s=%r inconnu — repli sur 'flat'", ANN_ENV, ann)
        ann = "flat"
    probes_raw = os.environ.get(IVF_PROBES_ENV, "").strip()
    ivf_probes: int | None = None
    if probes_raw:
        try:
            ivf_probes = max(1, int(probes_raw))
        except ValueError:
            logger.warning("%s=%r invalide — ignoré", IVF_PROBES_ENV, probes_raw)
    shard_by: str | None = os.environ.get(SHARD_BY_ENV, "language").strip() or "language"
    if shard_by.lower() in ("off", "none"):
        shard_by = None
    return {
        "use_reranker": use_reranker, "ann": ann,
        "ivf_probes": ivf_probes, "shard_by": shard_by,
    }


def build_danann(
    *,
    knowledge_dir: Path | str = DEFAULT_KNOWLEDGE_DIR,
    index_path: str | None = None,
    use_reranker: bool | None = None,
) -> Danann:
    """Construit le Danann du runtime (voir docstring du module).

    `index_path` l'emporte sur la variable d'env `MORRIGAN_INDEX`. Si
    l'index pointé est invalide, on logge un warning et on retombe sur
    l'ingestion de `knowledge_dir` (dégradation gracieuse, jamais
    d'exception au boot). `use_reranker=None` = résolu depuis l'env
    (`MORRIGAN_RERANKER`, défaut off — cf. _retrieval_opts).
    """
    opts = _retrieval_opts(use_reranker)
    index_path = index_path or os.environ.get(INDEX_ENV)
    if index_path:
        p = Path(index_path)
        if _is_valid_index(p):
            logger.info("Danann : chargement de l'index persisté %s", p)
            return Danann.load_index(p, **opts)
        logger.warning(
            "%s=%s invalide (corpus.json/vectors.npz absents) — "
            "fallback sur l'ingestion de %s",
            INDEX_ENV, p, knowledge_dir,
        )

    danann = Danann(backend="memory", **opts)
    kdir = Path(knowledge_dir)
    if kdir.exists():
        total = ingest_directory(danann, kdir)
        logger.info("Danann : %d chunks ingérés depuis %s", total, kdir)
    else:
        logger.warning("%s introuvable — Danann vide", kdir)
    return danann
