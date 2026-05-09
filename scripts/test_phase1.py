"""
MORRIGAN — Test Phase 2 end-to-end sur corpus reel.

Ingere data/knowledge/*.md dans Danann (memory backend) avec metadonnees
riches, puis fait passer une batterie de requetes a travers le pipeline
complet avec reranker cross-encoder.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from core.dagda import AnDagda
from modules.brigid.model import Brigid
from modules.ogham.engine import Ogham
from modules.danann.store import Danann
from modules.scathach.generator import Scathach
from modules.cauldron.memory import Cauldron
from scripts.ingest_knowledge import ingest_directory


TEST_QUERIES = [
    # Factuel
    ("Qu'est-ce que TCP ?", "factuel"),
    ("Qu'est-ce que RWKV ?", "factuel"),
    ("Qui est la Morrigan ?", "factuel"),
    ("Qu'est-ce que l'Ogham ?", "factuel"),
    # Comparaison
    ("Compare TCP et UDP", "comparaison"),
    ("Quelle est la difference entre Mamba et les Transformers ?", "comparaison"),
    # Explication
    ("Explique les Liquid Neural Networks", "explication"),
    ("Explique le projet Morrigan", "explication"),
    # Hors corpus (doit tomber dans not_found)
    ("Qui a gagne la Coupe du Monde 2022 ?", "hors_corpus"),
    # Conversation
    ("Salut, comment ca va ?", "conversation"),
]


async def main():
    logging.basicConfig(
        level=logging.WARNING,  # reduire le bruit
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 70)
    print("  MORRIGAN — Test Phase 2 end-to-end (corpus reel + reranker)")
    print("=" * 70)

    # 1. Init modules (avec reranker actif)
    dagda = AnDagda()
    brigid = Brigid()
    ogham = Ogham()
    danann = Danann(backend="memory", use_reranker=True)
    scathach = Scathach()
    cauldron = Cauldron()

    dagda.register_module("brigid", brigid)
    dagda.register_module("ogham", ogham)
    dagda.register_module("danann", danann)
    dagda.register_module("scathach", scathach)
    dagda.register_module("cauldron", cauldron)

    await dagda.initialize()

    # 2. Ingestion corpus reel (Phase 2 : metadonnees riches)
    print()
    print("--- Ingestion data/knowledge/ (Phase 2 — metadonnees riches) ---")
    source = Path("data/knowledge")
    if not source.exists():
        print(f"ERREUR: {source} introuvable.")
        sys.exit(1)
    total = ingest_directory(danann, source)
    print(f"Corpus indexe: {total} chunks")

    # Afficher les metadonnees du premier chunk pour validation
    if danann.metadata:
        print()
        print("--- Exemple de metadonnees (chunk 0) ---")
        for key, val in danann.metadata[0].items():
            print(f"  {key:15s}: {val}")
    print()

    # 3. Test retrieval direct (sans pipeline) pour valider le reranker
    print("=" * 70)
    print("  TEST RETRIEVAL DIRECT (cosine + reranker)")
    print("=" * 70)
    test_retrieval_queries = [
        "Qu'est-ce que TCP ?",
        "Compare TCP et UDP",
        "Liquid Neural Networks",
    ]
    for q in test_retrieval_queries:
        results = danann.search(q, top_k=3)
        print(f"\n  Query: {q}")
        for j, (text, score, meta) in enumerate(results):
            cosine = meta.get("score_cosine", score)
            rerank = meta.get("score_reranker", "N/A")
            domain = meta.get("domain", "?")
            ctype = meta.get("type", "?")
            section = meta.get("section", "?")
            print(
                f"    [{j+1}] score={score:.3f} "
                f"(cosine={cosine:.3f}, reranker={rerank}) "
                f"| {domain}/{ctype} | {section}"
            )
            print(f"        {text[:100]}...")

    # 4. Batterie de requetes pipeline complet
    print()
    print("=" * 70)
    print("  TEST PIPELINE COMPLET")
    print("=" * 70)

    latencies = []
    for i, (query, expected_type) in enumerate(TEST_QUERIES, 1):
        print("=" * 70)
        print(f"[{i:02d}/{len(TEST_QUERIES)}] ({expected_type}) {query}")
        print("-" * 70)

        t0 = time.perf_counter()
        response = await dagda.process(query, session_id="phase2")
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)

        print(response)
        print(f"(latence: {dt:.0f} ms)")
        print()

        cauldron.add_turn("phase2", "user", query)
        cauldron.add_turn("phase2", "morrigan", response)

    # 5. Resume
    print("=" * 70)
    print("RESUME — Phase 2")
    print("=" * 70)
    avg = sum(latencies) / len(latencies)
    print(f"Requetes    : {len(TEST_QUERIES)}")
    print(f"Latence moy : {avg:.0f} ms")
    print(f"Latence max : {max(latencies):.0f} ms")
    print(f"Latence min : {min(latencies):.0f} ms")
    print(f"Cauldron    : {len(cauldron.get_history('phase2'))} tours memorises")
    print(f"Reranker    : {'actif' if danann.reranker else 'inactif'}")
    print(f"Chunks idx  : {danann.count()}")
    print()
    print("Test Phase 2 termine.")


if __name__ == "__main__":
    asyncio.run(main())
