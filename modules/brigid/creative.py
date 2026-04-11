"""
BRIGID — Génération associative / créativité.

Sous-module pour les patterns associatifs, métaphores, analogies.
"""

import logging
from typing import List

logger = logging.getLogger("morrigan.brigid.creative")


class CreativeEngine:
    """
    Moteur de créativité basé sur les associations.

    Phase 0 : Associations statiques.
    Phase 2 : Réseau LNN pour les associations dynamiques.
    """

    def associate(self, concept: str) -> List[str]:
        """Génère des associations à partir d'un concept."""
        logger.debug("Associations pour: %s", concept)

        # Placeholder
        return [f"[association pour '{concept}' — à implémenter]"]
