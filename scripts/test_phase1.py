"""
MORRIGAN — Test Phase 1 end-to-end sur corpus reel.

Ingere data/knowledge/*.md dans Danann (memory backend),
puis fait passer une batterie de requetes a travers le pipeline
complet (An Dagda -> Brigid -> Ogham -> Danann -> Scathach).
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
    "Qu'est-ce que TCP ?",
    "Qu'est-ce que RWKV ?",
    "Qui est la Morrigan ?",
    "Qu'est-ce que l'Ogham ?",
    # Comparaison
    "Compare TCP et UDP",
    "Quelle est la difference entre Mamba et les Transformers ?",
    # Explication
    "Explique les Liquid Neural Networks",
    "Explique le projet Morrigan",
    # Hors corpus (doit tomber dans not_found)
    "Qui a gagne la Coupe du Monde 2022 ?",
    # Conversation
    "Salut, comment ca va ?",
]


async def main():
    logging.basicConfig(
        level=logging.WARNING,  # reduire le bruit
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 70)
    print("  MORRIGAN — Test Phase 1 end-to-end (corpus reel)")
    print("=" * 70)

    # 1. Init modules
    dagda = AnDagda()
    brigid = Brigid()
    ogham = Ogham()
    danann = Danann(backend="memory")
    scathach = Scathach()
    cauldron = Cauldron()

    dagda.register_module("brigid", brigid)
    dagda.register_module("ogham", ogham)
    dagda.register_module("danann", danann)
    dagda.register_module("scathach", scathach)
    dagda.register_module("cauldron", cauldron)

    await dagda.initialize()

    # 2. Ingestion corpus reel
    print()
    print("--- Ingestion data/knowledge/ ---")
    source = Path("data/knowledge")
    if not source.exists():
        print(f"ERREUR: {source} introuvable.")
        sys.exit(1)
    total = ingest_directory(danann, source)
    print(f"Corpus indexe: {total} chunks")
    print()

    # 3. Batterie de requetes
    latencies = []
    for i, query in enumerate(TEST_QUERIES, 1):
        print("=" * 70)
        print(f"[{i:02d}/{len(TEST_QUERIES)}] {query}")
        print("-" * 70)

        t0 = time.perf_counter()
        response = await dagda.process(query, session_id="phase1")
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)

        print(response)
        print(f"(latence: {dt:.0f} ms)")
        print()

        cauldron.add_turn("phase1", "user", query)
        cauldron.add_turn("phase1", "morrigan", response)

    # 4. Resume
    print("=" * 70)
    print("RESUME")
    print("=" * 70)
    avg = sum(latencies) / len(latencies)
    print(f"Requetes    : {len(TEST_QUERIES)}")
    print(f"Latence moy : {avg:.0f} ms")
    print(f"Latence max : {max(latencies):.0f} ms")
    print(f"Latence min : {min(latencies):.0f} ms")
    print(f"Cauldron    : {len(cauldron.get_history('phase1'))} tours memorises")
    print()
    print("Test Phase 1 termine.")


if __name__ == "__main__":
    asyncio.run(main())
