"""
BRIGID — Architecture LNN (Liquid Neural Network).

Utilise ncps (Neural Circuit Policies) pour créer un réseau
Liquid Time-Constant compact et efficient.
"""

import logging
from typing import Any, Dict

from core.types import ModuleInput, ModuleOutput, MorriganModule

logger = logging.getLogger("morrigan.brigid")


class Brigid(MorriganModule):
    """
    Module neuronal de Morrigan.

    Phase 0 : Squelette avec réponse placeholder.
    Phase 2 : Réseau LTC/CfC réel pour classification et créativité.
    """

    def __init__(self):
        self.initialized = False
        logger.info("Brigid (LNN) — initialisée en mode squelette")

    async def process(self, input: ModuleInput) -> ModuleOutput:
        """Traite un input via le réseau LNN."""
        logger.info("Brigid traite: %s", input.query[:60])

        # Phase 0 : placeholder
        return ModuleOutput(
            result={"patterns": [], "classification": "unknown"},
            confidence=0.1,
            metadata={"phase": 0, "note": "Squelette — LNN non entraîné"},
        )

    async def health_check(self) -> bool:
        """Vérifie que Brigid est opérationnelle."""
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Brigid",
            "type": "neural_network",
            "architecture": "LTC/CfC (Liquid Neural Network)",
            "capabilities": [
                "intent_classification",
                "semantic_encoding",
                "creative_association",
            ],
            "phase": 0,
        }
