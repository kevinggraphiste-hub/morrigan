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


def _is_valid_index(path: Path) -> bool:
    """Un dossier d'index est valide s'il contient les deux fichiers."""
    return (path / "corpus.json").exists() and (path / "vectors.npz").exists()


def build_danann(
    *,
    knowledge_dir: Path | str = DEFAULT_KNOWLEDGE_DIR,
    index_path: str | None = None,
    use_reranker: bool = True,
) -> Danann:
    """Construit le Danann du runtime (voir docstring du module).

    `index_path` l'emporte sur la variable d'env `MORRIGAN_INDEX`. Si
    l'index pointé est invalide, on logge un warning et on retombe sur
    l'ingestion de `knowledge_dir` (dégradation gracieuse, jamais
    d'exception au boot).
    """
    index_path = index_path or os.environ.get(INDEX_ENV)
    if index_path:
        p = Path(index_path)
        if _is_valid_index(p):
            logger.info("Danann : chargement de l'index persisté %s", p)
            return Danann.load_index(p, use_reranker=use_reranker)
        logger.warning(
            "%s=%s invalide (corpus.json/vectors.npz absents) — "
            "fallback sur l'ingestion de %s",
            INDEX_ENV, p, knowledge_dir,
        )

    danann = Danann(backend="memory", use_reranker=use_reranker)
    kdir = Path(knowledge_dir)
    if kdir.exists():
        total = ingest_directory(danann, kdir)
        logger.info("Danann : %d chunks ingérés depuis %s", total, kdir)
    else:
        logger.warning("%s introuvable — Danann vide", kdir)
    return danann
