"""
Construit un index Danann COMPRESSÉ à partir d'un corpus, et le sauve
sur disque (Phase 4).

C'est l'outil « gros corpus sur PC modeste » : on ingère un répertoire
de documents, on quantize l'index (int8 ~4× / binary ~4.5× moins de
RAM), et on persiste sur disque. À la lecture (`Danann.load_index`),
aucun float32 n'est matérialisé et le corpus n'est pas réembeddé.

Usage :
    .venv-uv/bin/python scripts/build_compressed_index.py \
        --source data/knowledge --output data/models/index_default \
        --compression int8

Pour un très gros corpus (ex: dump Wikipedia FR extrait en .txt),
pointer --source sur le dossier et augmenter --patterns au besoin.
L'ingestion est incrémentale (fichier par fichier).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.danann.store import Danann  # noqa: E402
from scripts.ingest_knowledge import ingest_directory  # noqa: E402

logger = logging.getLogger("morrigan.build_index")

DEFAULT_SOURCE = Path("data/knowledge")
DEFAULT_OUTPUT = Path("data/models/index_default")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--compression", default="int8", choices=["none", "int8", "binary"]
    )
    parser.add_argument(
        "--min-chunks", type=int, default=1,
        help="Échec si moins de chunks indexés (garde-fou).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    source = Path(args.source)
    if not source.exists():
        sys.exit(f"Source introuvable : {source}")

    # Reranker off pendant la construction (inutile, gourmand).
    danann = Danann(compression=args.compression, use_reranker=False)

    t0 = time.time()
    total = ingest_directory(danann, source)
    build_s = time.time() - t0

    if total < args.min_chunks:
        sys.exit(f"ÉCHEC : {total} chunks indexés < {args.min_chunks}.")

    out = Path(args.output)
    danann.save_index(out)

    mem = danann.memory_bytes()
    # Estimation float32 équivalente pour situer le gain.
    dim = 384
    float_equiv = danann.count() * dim * 4
    ratio = float_equiv / mem if mem else 0.0

    print("─" * 60)
    print(f"Corpus      : {source}")
    print(f"Chunks      : {danann.count()} (ingestion {build_s:.1f}s)")
    print(f"Compression : {args.compression}")
    print(f"Index RAM   : {mem/1024:.1f} KB  (float32 équiv ~{float_equiv/1024:.1f} KB → ×{ratio:.1f})")
    print(f"Sauvé       : {out}/  (corpus.json + vectors.npz)")
    print("─" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
