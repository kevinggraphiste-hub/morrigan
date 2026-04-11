"""
MORRIGAN — Script d'ingestion de corpus.

Lit des fichiers texte/markdown depuis data/knowledge/,
les decoupe en chunks, et les indexe dans Danann.

Usage :
    python scripts/ingest_knowledge.py
    python scripts/ingest_knowledge.py --source data/knowledge/ --backend memory
    python scripts/ingest_knowledge.py --backend supabase
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, ".")

from modules.danann.store import Danann


# Configuration du chunking
CHUNK_SIZE = 400       # Caracteres par chunk
CHUNK_OVERLAP = 50     # Overlap entre chunks


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def split_into_sentences(text: str) -> List[str]:
    """Decoupe en phrases (approximation simple)."""
    # Nettoyer et splitter sur points, points d'exclamation, d'interrogation
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Decoupe un texte en chunks, en respectant les limites de phrases.
    """
    sentences = split_into_sentences(text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        # Si la phrase seule depasse, on la coupe brutalement
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            # Decoupe grossiere
            for i in range(0, len(sentence), chunk_size - overlap):
                chunks.append(sentence[i : i + chunk_size].strip())
            continue

        # Si on peut ajouter la phrase au chunk courant
        if len(current) + len(sentence) + 1 <= chunk_size:
            current += (" " if current else "") + sentence
        else:
            # Fermer le chunk courant, commencer un nouveau
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 20]  # Ignorer les micro-chunks


def load_file(path: Path) -> Tuple[str, dict]:
    """Charge un fichier et retourne son contenu + metadata."""
    content = path.read_text(encoding="utf-8")
    metadata = {
        "source": path.name,
        "path": str(path),
        "size": len(content),
    }
    return content, metadata


def ingest_directory(
    danann: Danann,
    source_dir: Path,
    extensions: Tuple[str, ...] = (".txt", ".md"),
) -> int:
    """Ingere tous les fichiers d'un repertoire dans Danann."""
    total_chunks = 0
    files_processed = 0

    for ext in extensions:
        for file_path in source_dir.rglob(f"*{ext}"):
            print(f"  Lecture: {file_path.name}")
            content, base_metadata = load_file(file_path)

            chunks = chunk_text(content)
            if not chunks:
                continue

            metadatas = [
                {**base_metadata, "chunk_index": i}
                for i in range(len(chunks))
            ]

            inserted = danann.index(chunks, metadatas)
            total_chunks += inserted
            files_processed += 1
            print(f"    -> {len(chunks)} chunks indexes")

    print()
    print(f"Total: {files_processed} fichier(s), {total_chunks} chunks")
    return total_chunks


def main():
    parser = argparse.ArgumentParser(description="Ingere un corpus dans Danann")
    parser.add_argument(
        "--source",
        default="data/knowledge",
        help="Repertoire source contenant les fichiers .txt/.md",
    )
    parser.add_argument(
        "--backend",
        choices=["memory", "supabase"],
        default="memory",
        help="Backend Danann (memory ou supabase)",
    )
    args = parser.parse_args()

    setup_logging()

    print("=" * 60)
    print("  MORRIGAN — Ingestion de corpus dans Danann")
    print("=" * 60)
    print()
    print(f"Source  : {args.source}")
    print(f"Backend : {args.backend}")
    print()

    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"ERREUR: le repertoire {source_dir} n'existe pas.")
        sys.exit(1)

    # Initialiser Danann avec le backend choisi
    danann = Danann(
        backend=args.backend,
        supabase_url=os.getenv("SUPABASE_URL", ""),
        supabase_key=os.getenv("SUPABASE_KEY", ""),
    )

    # Ingestion
    total = ingest_directory(danann, source_dir)

    # Test rapide
    if total > 0:
        print()
        print("--- Test rapide ---")
        test_queries = ["Morrigan", "reseau de neurones", "mythologie celtique"]
        for q in test_queries:
            results = danann.search(q, top_k=2)
            if results:
                top = results[0]
                print(f"  '{q}' -> [{top[1]:.2f}] {top[0][:80]}...")

    print()
    print("Ingestion terminee.")


if __name__ == "__main__":
    main()
