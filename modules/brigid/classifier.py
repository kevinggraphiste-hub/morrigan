"""
BRIGID — Classificateur d'intention.

Sous-module dédié à la classification des inputs utilisateur
(intention, sentiment, type de requête).
"""

import logging
from typing import Dict, List

logger = logging.getLogger("morrigan.brigid.classifier")


class IntentClassifier:
    """
    Classificateur d'intention.

    Phase 0 : Basé sur des heuristiques.
    Phase 2 : Réseau LNN entraîné.
    """

    def classify(self, text: str) -> Dict[str, float]:
        """
        Retourne un dictionnaire {intention: score}.

        Phase 0 : heuristiques simples.
        """
        logger.debug("Classification de: %s", text[:60])

        # Placeholder — sera remplacé par le réseau LNN
        return {
            "factual": 0.25,
            "creative": 0.25,
            "reasoning": 0.25,
            "conversation": 0.25,
        }
