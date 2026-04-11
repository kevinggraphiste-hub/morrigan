"""
MORRIGAN — Test d'integration du pipeline complet.

Peuple Danann avec des connaissances, puis teste le pipeline
end-to-end avec plusieurs types de requetes.
"""

import asyncio
import logging
import sys

sys.path.insert(0, ".")

from core.dagda import AnDagda
from modules.brigid.model import Brigid
from modules.ogham.engine import Ogham
from modules.danann.store import Danann
from modules.scathach.generator import Scathach
from modules.cauldron.memory import Cauldron


# Base de connaissances de test
KNOWLEDGE = [
    "TCP est un protocole reseau fiable qui garantit la livraison des paquets dans l'ordre.",
    "UDP est un protocole reseau rapide mais sans garantie de livraison.",
    "HTTP utilise TCP pour assurer une communication web fiable.",
    "DNS utilise UDP pour une resolution rapide des noms de domaine.",
    "TCP etablit une connexion avant de transmettre des donnees (handshake en 3 temps).",
    "UDP envoie des datagrammes sans etablir de connexion prealable.",
    "Brigid est la deesse celtique de la forge, de la poesie et de la guerison.",
    "Le Dagda est le pere des Tuatha De Danann et possede un chaudron inepuisable.",
    "Scathach est une guerriere legendaire qui forme les heros.",
    "Les Liquid Neural Networks sont inspires du ver C. elegans et ses 302 neurones.",
    "Les LNN peuvent atteindre des performances elevees avec peu de parametres.",
    "Le modele CfC (Closed-form Continuous-time) est une version efficace des LNN.",
]


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  MORRIGAN — Test d'integration pipeline complet")
    print("=" * 60)
    print()

    # 1. Initialisation
    dagda = AnDagda()
    brigid = Brigid()
    ogham = Ogham()
    danann = Danann()
    scathach = Scathach()
    cauldron = Cauldron()

    dagda.register_module("brigid", brigid)
    dagda.register_module("ogham", ogham)
    dagda.register_module("danann", danann)
    dagda.register_module("scathach", scathach)
    dagda.register_module("cauldron", cauldron)

    await dagda.initialize()

    # 2. Peupler Danann
    print()
    print("--- Peuplement de Danann ---")
    danann.index(KNOWLEDGE)
    print()

    # 3. Tests
    test_queries = [
        "Qu'est-ce que TCP ?",
        "Compare TCP et UDP",
        "Qui est Brigid ?",
        "Explique les LNN",
        "Salut, comment ca va ?",
    ]

    for i, query in enumerate(test_queries, 1):
        print()
        print("=" * 60)
        print(f"TEST {i}/{len(test_queries)}: {query}")
        print("=" * 60)
        response = await dagda.process(query, session_id="test")
        print()
        print("REPONSE:")
        print(response)
        print()

        # Memoriser dans Cauldron
        cauldron.add_turn("test", "user", query)
        cauldron.add_turn("test", "morrigan", response)

    print()
    print("=" * 60)
    print("Test d'integration termine.")
    print(f"Cauldron a memorise {len(cauldron.get_history('test'))} tours.")


if __name__ == "__main__":
    asyncio.run(main())
