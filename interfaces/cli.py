"""
Interface CLI pour Morrigan.

Permet de tester Morrigan directement depuis le terminal.
"""

import asyncio
import logging
import sys

# Ajout du path racine pour les imports
sys.path.insert(0, ".")

from core.dagda import AnDagda
from core.env import load_env
from core.knowledge import build_danann
from modules.brigid.model import Brigid
from modules.ogham.engine import Ogham
from modules.scathach.generator import Scathach
from modules.cauldron.memory import Cauldron


def setup_logging() -> None:
    """Configure le logging pour la phase de recherche."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


async def main() -> None:
    """Boucle principale CLI."""
    setup_logging()
    load_env()  # charge .env si présent (HF_TOKEN, Supabase…)
    logger = logging.getLogger("morrigan.cli")

    print("=" * 60)
    print("  MORRIGAN — Architecture IA Modulaire v0.1")
    print("  Phases 1-3 — génération neuronale RWKV (streaming)")
    print("=" * 60)
    print()

    # Initialiser l'orchestrateur
    dagda = AnDagda()

    # Enregistrer les modules. Scáthach en backend RWKV pour la vraie
    # génération + streaming (fallback template si modèle absent).
    dagda.register_module("brigid", Brigid())
    dagda.register_module("ogham", Ogham())
    # Danann : index persisté (MORRIGAN_INDEX) si présent, sinon
    # ingestion de data/knowledge. Cf. core.knowledge.build_danann.
    dagda.register_module("danann", build_danann())
    dagda.register_module("scathach", Scathach(backend="rwkv"))
    dagda.register_module("cauldron", Cauldron())

    await dagda.initialize()

    print("\nMorrigan est prête. Tapez 'quit' pour quitter.\n")

    while True:
        try:
            user_input = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() in ("/stats", "stats"):
            print(f"\n{dagda.format_stats()}\n")
            continue

        # Streaming : la réponse s'affiche au fil de l'eau (le 1er token
        # arrive vite, ce qui masque la latence d'un 1.6B sur CPU).
        print("\nMorrigan > ", end="", flush=True)
        async for piece in dagda.process_stream(user_input):
            print(piece, end="", flush=True)
        print("\n")

    print("\nMorrigan se repose. À bientôt.")


if __name__ == "__main__":
    asyncio.run(main())
