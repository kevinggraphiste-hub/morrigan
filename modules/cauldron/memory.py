"""
CAULDRON — Gestionnaire de mémoire de travail.

Gère les différentes couches de mémoire :
- Immédiate (tour en cours)
- Court terme (conversation)
- Épisodique (sessions)
- Long terme → délégué à DANANN
"""

import logging
from typing import Any, Dict, List

from core.types import ModuleInput, ModuleOutput, MorriganModule

logger = logging.getLogger("morrigan.cauldron")


class Cauldron(MorriganModule):
    """
    Mémoire de travail de Morrigan — le Chaudron inépuisable.
    """

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        logger.info("Cauldron — mémoire de travail (max_history=%d)", max_history)

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Ajoute un tour à l'historique de la session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({"role": role, "content": content})

        # Garder seulement les N derniers tours
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retourne l'historique d'une session."""
        return self.sessions.get(session_id, [])

    async def process(self, input: ModuleInput) -> ModuleOutput:
        """Fournit le contexte conversationnel."""
        session_id = input.context.get("session_id", "default")
        history = self.get_history(session_id)

        logger.info(
            "Cauldron — session '%s', %d tours en mémoire",
            session_id,
            len(history),
        )

        return ModuleOutput(
            result={"history": history, "turn_count": len(history)},
            confidence=1.0,
            metadata={"session_id": session_id},
        )

    async def health_check(self) -> bool:
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Cauldron",
            "type": "working_memory",
            "capabilities": [
                "conversation_history",
                "session_management",
                "context_retrieval",
            ],
            "phase": 0,
        }
