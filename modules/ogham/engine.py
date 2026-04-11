"""
OGHAM — Moteur symbolique.

Phase 0 : Structuration de reponses + base de regles pyDatalog.
Phase 2 : Raisonnement formel avance (chainage complexe, ASP).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.types import ModuleInput, ModuleOutput, MorriganModule

logger = logging.getLogger("morrigan.ogham")


class Ogham(MorriganModule):
    """
    Moteur symbolique de Morrigan.

    Combine :
    - Structuration de reponses (templates de raisonnement)
    - Base de regles pyDatalog pour les deductions
    """

    def __init__(self, rules_path: str = "config/ogham_rules.yaml"):
        self.rules: Dict[str, Any] = {}
        self.rules_path = rules_path
        self._datalog_ready = False
        self._load_rules()
        self._init_datalog()
        logger.info("Ogham — moteur symbolique initialise")

    def _load_rules(self) -> None:
        """Charge les regles depuis le fichier YAML."""
        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                self.rules = yaml.safe_load(f) or {}
            logger.info(
                "Regles chargees: %d templates de structure",
                len(self.rules.get("structure_templates", {})),
            )
        except FileNotFoundError:
            logger.warning("Fichier de regles %s introuvable", self.rules_path)
            self.rules = {}

    def _init_datalog(self) -> None:
        """Initialise la base pyDatalog avec les termes de base."""
        try:
            from pyDatalog import pyDatalog
            # Termes standard pour le raisonnement
            pyDatalog.create_terms(
                "X, Y, Z, "
                "is_a, has_property, related_to, "
                "implies, contradicts"
            )
            self._datalog_ready = True
            logger.info("pyDatalog initialise — termes de base charges")
        except Exception as e:
            logger.error("Erreur initialisation pyDatalog: %s", e)

    def detect_structure_type(self, query: str) -> str:
        """Detecte le type de structuration approprie."""
        q = query.lower()
        if any(w in q for w in ["compare", "difference", "vs", "versus"]):
            return "comparison"
        if any(w in q for w in ["explique", "qu'est-ce", "c'est quoi", "comment"]):
            return "explanation"
        if any(w in q for w in ["analyse", "pourquoi", "raison"]):
            return "analysis"
        return "explanation"  # default

    async def process(self, input: ModuleInput) -> ModuleOutput:
        """Applique le raisonnement symbolique a l'input."""
        logger.info("Ogham traite: %s", input.query[:60])

        previous = input.context.get("previous_results", {})

        # 1. Detecter le type de structure
        structure_type = self.detect_structure_type(input.query)

        # 2. Recuperer le template associe
        templates = self.rules.get("structure_templates", {})
        template = templates.get(structure_type, {"sections": ["content"]})

        # 3. Construire la structure
        structure = {
            "query": input.query,
            "type": structure_type,
            "sections": template.get("sections", []),
            "knowledge_chunks": [],
        }

        # 4. Si Danann a fourni des connaissances, les integrer
        if "danann" in previous:
            danann_output = previous["danann"]
            if danann_output.success and danann_output.result:
                chunks = danann_output.result.get("chunks", [])
                structure["knowledge_chunks"] = chunks

        confidence = 0.7 if structure["knowledge_chunks"] else 0.4

        return ModuleOutput(
            result=structure,
            confidence=confidence,
            metadata={
                "structure_type": structure_type,
                "datalog_ready": self._datalog_ready,
                "chunks_used": len(structure["knowledge_chunks"]),
            },
        )

    async def health_check(self) -> bool:
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Ogham",
            "type": "symbolic_engine",
            "capabilities": [
                "rule_based_reasoning",
                "response_structuring",
                "logical_deduction",
                "comparison",
            ],
            "datalog_ready": self._datalog_ready,
            "templates": list(self.rules.get("structure_templates", {}).keys()),
        }
