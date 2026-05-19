"""
Morrigan — Types de données partagés.

Définit les interfaces et structures de données que tous les modules
doivent respecter pour communiquer via l'orchestrateur An Dagda.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class QueryType(Enum):
    """Types de requêtes reconnus par An Dagda."""
    FACTUAL = "factual"             # Question factuelle → DANANN + OGHAM
    REASONING = "reasoning"         # Raisonnement logique → OGHAM principal
    CREATIVE = "creative"           # Tâche créative → BRIGID + SCÁTHACH
    CONVERSATION = "conversation"   # Conversation courante → CAULDRON + SCÁTHACH
    COMPLEX = "complex"             # Multi-étapes → tous les modules
    CODE = "code"                   # Bloc de code à vérifier → MORRIGAN-CODE + SCÁTHACH
    UNKNOWN = "unknown"


@dataclass
class ModuleInput:
    """Input standard pour tous les modules Morrigan."""
    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleOutput:
    """Output standard pour tous les modules Morrigan."""
    result: Any
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True si aucune erreur."""
        return len(self.errors) == 0


class MorriganModule(ABC):
    """Interface abstraite que tous les modules Morrigan doivent implémenter."""

    @abstractmethod
    async def process(self, input: ModuleInput) -> ModuleOutput:
        """Traite un input et retourne un output."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Vérifie que le module est opérationnel."""
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Retourne les capacités du module."""
        pass


@dataclass
class RoutingDecision:
    """Décision de routage prise par An Dagda."""
    query_type: QueryType
    modules: List[str]              # Noms des modules à activer, dans l'ordre
    priority: str = "normal"        # "low", "normal", "high"
    reasoning: str = ""             # Pourquoi ce routage (debug/logging)
    domain_hint: Optional[str] = None  # Phase 2 : indice de domaine (reseau, ia, mythologie, projet, code)
