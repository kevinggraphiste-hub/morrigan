"""
Morrigan — Pipeline de traitement.

Gère le flux de données entre les modules,
le logging du pipeline, et les métriques.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.types import ModuleInput, ModuleOutput

logger = logging.getLogger("morrigan.pipeline")


@dataclass
class PipelineStep:
    """Un pas dans le pipeline de traitement."""
    module_name: str
    input: ModuleInput
    output: ModuleOutput | None = None
    duration_ms: float = 0.0
    error: str | None = None


@dataclass
class PipelineTrace:
    """Trace complète d'un pipeline de traitement (pour debug/analyse)."""
    session_id: str
    query: str
    steps: List[PipelineStep] = field(default_factory=list)
    total_duration_ms: float = 0.0

    def summary(self) -> Dict[str, Any]:
        """Résumé du pipeline pour le logging."""
        return {
            "session_id": self.session_id,
            "query": self.query[:80],
            "steps": len(self.steps),
            "total_ms": round(self.total_duration_ms, 2),
            "modules": [s.module_name for s in self.steps],
            "errors": [s.error for s in self.steps if s.error],
        }
