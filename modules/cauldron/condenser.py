"""
CAULDRON — Condensation / Résumé.

Après N tours, résume la conversation pour libérer la mémoire
et extrait les faits nouveaux pour DANANN.
"""

import logging
from typing import Dict, List

logger = logging.getLogger("morrigan.cauldron.condenser")


class ConversationCondenser:
    """
    Condenseur de conversations.

    Phase 0 : Extraction simple (premiers/derniers messages).
    Phase 2 : Résumé extractif puis abstractif.
    """

    def condense(self, history: List[Dict[str, str]]) -> str:
        """Condense un historique en résumé."""
        if not history:
            return ""

        # Phase 0 : concaténation des contenus clés
        key_points = [
            turn["content"][:100]
            for turn in history
            if turn["role"] == "user"
        ]
        return " | ".join(key_points[-5:])

    def extract_facts(self, history: List[Dict[str, str]]) -> List[str]:
        """Extrait les faits nouveaux d'une conversation (pour DANANN)."""
        # Phase 0 : placeholder
        return []
