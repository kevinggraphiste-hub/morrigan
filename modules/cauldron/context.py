"""
CAULDRON — Contexte conversationnel.

Gère le contexte actif : intentions en cours, tâches multi-tours,
état de la conversation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("morrigan.cauldron.context")


@dataclass
class ConversationContext:
    """État courant d'une conversation."""
    session_id: str
    current_intent: Optional[str] = None
    active_task: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    turn_number: int = 0
