"""
Ingestion de Wikipédia FR à l'échelle (Phase 5).

Stream le dataset `wikimedia/wikipedia` (FR) — sans télécharger tout le
dump — chunk les articles, les indexe dans un Danann **compressé**
(int8 par défaut, compression Phase 4), et persiste l'index sur disque.
Démontre la thèse Phase 5 : un gros corpus encyclopédique réel tenu sur
PC modeste, sans réembedding ni float32 matérialisé au chargement.

Usage :
    .venv-uv/bin/python scripts/ingest_wikipedia.py --max-articles 500
    .venv-uv/bin/python scripts/ingest_wikipedia.py \
        --max-articles 20000 --compression binary --output data/models/index_wiki

Le streaming évite de télécharger les ~20 Go du dump : on ne tire que
les `--max-articles` premiers articles.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.danann.store import Danann  # noqa: E402
from scripts.ingest_knowledge import chunk_text  # noqa: E402

logger = logging.getLogger("morrigan.ingest_wiki")

DEFAULT_OUTPUT = Path("data/models/index_wiki")
# Articles trop courts (homonymies, ébauches) : peu utiles, on filtre.
MIN_ARTICLE_CHARS = 400


def _iter_articles(config: str, max_articles: int):
    """Itère sur les articles Wikipédia en streaming (titre, texte)."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    for row in itertools.islice(ds, max_articles * 3):  # marge pour le filtre
        text = (row.get("text") or "").strip()
        if len(text) >= MIN_ARTICLE_CHARS:
            yield row.get("title", "?"), text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="20231101.fr",
                        help="Config du dataset wikimedia/wikipedia (snapshot.langue).")
    parser.add_argument("--max-articles", type=int, default=500)
    parser.add_argument("--compression", default="int8",
                        choices=["none", "int8", "binary"])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch", type=int, default=256,
                        help="Taille de lot d'indexation (chunks).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    danann = Danann(compression=args.compression, use_reranker=False)

    t0 = time.time()
    n_articles = 0
    batch_texts: list[str] = []
    batch_meta: list[dict] = []

    def _flush() -> None:
        if batch_texts:
            danann.index(batch_texts, batch_meta)
            batch_texts.clear()
            batch_meta.clear()

    print(f"Streaming wikimedia/wikipedia [{args.config}] — "
          f"{args.max_articles} articles max…")
    try:
        for title, text in _iter_articles(args.config, args.max_articles):
            for ck in chunk_text(text):
                batch_texts.append(ck)
                batch_meta.append({"domain": "wikipedia", "source": title})
                if len(batch_texts) >= args.batch:
                    _flush()
            n_articles += 1
            if n_articles % 100 == 0:
                print(f"  {n_articles} articles, {danann.count()} chunks…")
            if n_articles >= args.max_articles:
                break
        _flush()
    except Exception as exc:  # réseau / dataset
        logger.error("Erreur d'ingestion : %s", exc)
        _flush()  # ne pas perdre le lot en cours d'indexation
        if danann.count() == 0:
            sys.exit(f"Aucun chunk ingéré ({exc}).")
        print(f"⚠️  interrompu ({exc}) — on sauve ce qui a été ingéré.")

    elapsed = time.time() - t0
    out = Path(args.output)
    danann.save_index(out)

    mem = danann.memory_bytes()
    float_equiv = danann.count() * 384 * 4
    ratio = float_equiv / mem if mem else 0.0

    print("─" * 60)
    print(f"Articles ingérés : {n_articles}")
    print(f"Chunks indexés   : {danann.count()} en {elapsed:.0f}s")
    print(f"Compression      : {args.compression}")
    print(f"Index RAM        : {mem/1e6:.1f} MB  (float32 équiv ~{float_equiv/1e6:.1f} MB → ×{ratio:.1f})")
    print(f"Sauvé            : {out}/")
    print("─" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
