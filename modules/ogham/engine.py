"""
OGHAM — Moteur symbolique.

Phase 0 : Structuration de réponses + base de règles pyDatalog.
Phase 2 (cette version) : utilisation du Knowledge Graph pour
    enrichir les réponses comparaison / explication / analyse avec
    des faits structurés extraits du corpus.

Ogham reste **dégradé-gracieux** : sans KG (fichier `data/models/
knowledge_graph.json` absent), il fonctionne comme avant — templates
de structure + chunks Danann passés tels quels. Avec un KG chargé, il
ajoute `compare` / `facts` dans la réponse pour le module Scáthach
en aval.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from core.types import ModuleInput, ModuleOutput, MorriganModule
from modules.ogham.extractor import slugify
from modules.ogham.knowledge_graph import KnowledgeGraph

logger = logging.getLogger("morrigan.ogham")


# Patterns pour extraire les entités d'intérêt depuis la query.
# Volontairement simples : Ogham ne fait pas de NLP profond ici, il
# délègue ce travail à l'extracteur (PR 2) qui a déjà peuplé le KG.

# « Compare X et Y » / « Différence entre X et Y » / « X vs Y ».
_COMPARE_PATTERNS = [
    re.compile(
        r"\bcompare(?:r|z)?\s+(.+?)\s+(?:et|à|avec|versus|vs\.?)\s+(.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:différence|difference)s?\s+entre\s+(.+?)\s+et\s+(.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\S+)\s+(?:vs|versus)\s+(\S+)\.?\??$",
        re.IGNORECASE,
    ),
]

# « Qu'est-ce que X » / « C'est quoi X » / « Définition de X » /
# « Parle-moi de X » / « Qui est X ».
_DEFINITION_PATTERNS = [
    re.compile(r"\bqu['e]?est-?ce\s+(?:que|qu['e])?\s*(.+?)\??\s*$", re.IGNORECASE),
    re.compile(r"\bc['e]?est\s+quoi\s+(.+?)\??\s*$", re.IGNORECASE),
    re.compile(r"\bdéfinition\s+(?:de|d['e])\s+(.+?)\.?$", re.IGNORECASE),
    re.compile(r"\bparle[\s\-]moi\s+de\s+(.+?)\.?$", re.IGNORECASE),
    re.compile(r"\bqui\s+(?:est|était|sont)\s+(.+?)\??\s*$", re.IGNORECASE),
]


def _strip_articles(label: str) -> str:
    """Enlève les déterminants/articles français en tête : « un TCP » → « TCP »."""
    return re.sub(
        r"^\s*(?:un|une|le|la|les|des|du|de\s+la|de\s+l['e]|l['e])\s+",
        "",
        label,
        flags=re.IGNORECASE,
    ).strip()


def _slug_candidates(label: str) -> List[str]:
    """Génère des ids candidats pour résoudre un label dans le KG.

    Stratégie : on essaie d'abord le slug complet, puis chaque mot pris
    individuellement (utile quand l'utilisateur dit « les protocoles TCP »
    et que seul `tcp` est dans le KG).
    """
    cleaned = _strip_articles(label.strip())
    candidates = [slugify(cleaned)]
    # Mots individuels (en gardant l'ordre, sans doublons).
    for word in cleaned.split():
        slug = slugify(word)
        if slug not in candidates and slug != "unknown":
            candidates.append(slug)
    return candidates


class Ogham(MorriganModule):
    """
    Moteur symbolique de Morrigan.

    Phase 2 : combine
      - Structuration de réponses (templates `comparison` / `explanation`
        / `analysis` depuis `config/ogham_rules.yaml`)
      - Knowledge graph (PR 1-3) consulté pour faits structurés
      - Base pyDatalog initialisée (pour usage futur PR 5+)
    """

    # Chemin par défaut du KG construit par scripts/build_knowledge_graph.py.
    DEFAULT_KG_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "data" / "models" / "knowledge_graph.json"
    )

    def __init__(
        self,
        rules_path: str = "config/ogham_rules.yaml",
        kg_path: Optional[Path] = None,
    ) -> None:
        self.rules: Dict[str, Any] = {}
        self.rules_path = rules_path
        self._datalog_ready = False
        self._kg_path = kg_path or self.DEFAULT_KG_PATH
        self._kg: Optional[KnowledgeGraph] = None
        self._kg_load_error: Optional[str] = None
        self._load_rules()
        self._init_datalog()
        logger.info(
            "Ogham — moteur symbolique initialisé (KG sera chargé depuis %s au 1er appel)",
            self._kg_path,
        )

    # ─── Chargement ─────────────────────────────────────────────

    def _load_rules(self) -> None:
        """Charge les regles depuis le fichier YAML."""
        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                self.rules = yaml.safe_load(f) or {}
            logger.info(
                "Règles chargées: %d templates de structure",
                len(self.rules.get("structure_templates", {})),
            )
        except FileNotFoundError:
            logger.warning("Fichier de règles %s introuvable", self.rules_path)
            self.rules = {}

    def _init_datalog(self) -> None:
        """Initialise la base pyDatalog avec les termes de base."""
        try:
            from pyDatalog import pyDatalog
            pyDatalog.create_terms(
                "X, Y, Z, is_a, has_property, related_to, implies, contradicts"
            )
            self._datalog_ready = True
            logger.info("pyDatalog initialisé — termes de base chargés")
        except Exception as e:
            logger.error("Erreur initialisation pyDatalog: %s", e)

    def _try_load_kg(self) -> bool:
        """Charge le KG au 1er appel. Renvoie True si dispo."""
        if self._kg is not None:
            return True
        if self._kg_load_error is not None:
            return False
        try:
            self._kg = KnowledgeGraph.load(self._kg_path)
            logger.info(
                "KG chargé : %d entités, %d triplets depuis %s",
                len(self._kg), self._kg.relation_count, self._kg_path,
            )
            return True
        except (FileNotFoundError, ValueError) as e:
            self._kg_load_error = str(e)
            logger.warning(
                "KG non disponible (%s) — Ogham fonctionnera en mode dégradé "
                "(templates de structure sans faits KG).",
                e,
            )
            return False

    # ─── Détection de structure & d'entités ─────────────────────

    def detect_structure_type(self, query: str) -> str:
        """Détecte le type de structuration approprié (heuristique simple)."""
        q = query.lower()
        if any(w in q for w in ["compare", "comparé", "différence", "vs", "versus"]):
            return "comparison"
        if any(w in q for w in ["explique", "qu'est-ce", "c'est quoi", "comment"]):
            return "explanation"
        if any(w in q for w in ["analyse", "pourquoi", "raison"]):
            return "analysis"
        return "explanation"  # défaut

    def extract_comparison_entities(
        self, query: str
    ) -> Optional[Tuple[str, str]]:
        """Si la query est une comparaison, renvoie les labels des 2 entités."""
        for pattern in _COMPARE_PATTERNS:
            m = pattern.search(query)
            if m:
                return _strip_articles(m.group(1)), _strip_articles(m.group(2))
        return None

    def extract_definition_entity(self, query: str) -> Optional[str]:
        """Si la query est une définition, renvoie le label de l'entité."""
        for pattern in _DEFINITION_PATTERNS:
            m = pattern.search(query)
            if m:
                return _strip_articles(m.group(1))
        return None

    def _resolve_in_kg(self, label: str) -> Optional[str]:
        """Tente plusieurs ids candidats pour résoudre `label` dans le KG."""
        if self._kg is None:
            return None
        for candidate in _slug_candidates(label):
            if candidate in self._kg:
                return candidate
        return None

    # ─── Process ────────────────────────────────────────────────

    async def process(self, input: ModuleInput) -> ModuleOutput:
        """Applique le raisonnement symbolique à l'input.

        Pipeline :
          1. Détecte le type de structure (comparison/explanation/analysis)
          2. Récupère le template associé (sections)
          3. Si KG dispo et requête identifiable → ajoute compare/facts
          4. Intègre les chunks Danann fournis (rétrocompat)
          5. Renvoie un ModuleOutput avec `type="structured_response"`
        """
        logger.info("Ogham traite: %s", input.query[:60])

        previous = input.context.get("previous_results", {})
        kg_loaded = self._try_load_kg()

        # 1-2. Structure de réponse + sections.
        structure_type = self.detect_structure_type(input.query)
        templates = self.rules.get("structure_templates", {})
        template = templates.get(structure_type, {"sections": ["content"]})

        structure: Dict[str, Any] = {
            "type": "structured_response",        # contrat stable, attendu par les consommateurs (Scáthach)
            "structure_type": structure_type,     # comparison/explanation/analysis
            "query": input.query,
            "sections": template.get("sections", []),
            "knowledge_chunks": [],
            "compare": None,
            "facts": None,
        }

        # 3. Enrichissement via KG.
        compare = None
        facts = None
        entities_used: List[str] = []

        if kg_loaded:
            if structure_type == "comparison":
                pair = self.extract_comparison_entities(input.query)
                if pair is not None:
                    a_id = self._resolve_in_kg(pair[0])
                    b_id = self._resolve_in_kg(pair[1])
                    if a_id and b_id:
                        compare = self._kg.compare(a_id, b_id)
                        entities_used = [a_id, b_id]
            else:
                label = self.extract_definition_entity(input.query)
                if label is not None:
                    eid = self._resolve_in_kg(label)
                    if eid is not None:
                        facts = self._kg.facts_about(eid)
                        entities_used = [eid]

        # Convertit en JSON-safe pour l'aval (Scáthach n'utilise pas les
        # dataclasses, juste des dicts/strings).
        if compare is not None:
            structure["compare"] = {
                "a": entities_used[0],
                "b": entities_used[1],
                "common_neighbors": [
                    {"id": e.id, "label": e.label, "predicates": preds}
                    for e, preds in compare["common_neighbors"]
                ],
                "a_only": [
                    {"id": e.id, "label": e.label, "predicate": p}
                    for e, p in compare["a_only"]
                ],
                "b_only": [
                    {"id": e.id, "label": e.label, "predicate": p}
                    for e, p in compare["b_only"]
                ],
                "direct_relations": [
                    {
                        "subject": r.subject_id,
                        "predicate": r.predicate,
                        "object": r.object_id,
                    }
                    for r in compare["direct_relations"]
                ],
            }
        if facts is not None:
            structure["facts"] = {
                "entity": entities_used[0],
                "relations": [
                    {
                        "subject": r.subject_id,
                        "predicate": r.predicate,
                        "object": r.object_id,
                        "confidence": r.confidence,
                    }
                    for r in facts
                ],
            }

        # 4. Chunks Danann.
        if "danann" in previous:
            danann_output = previous["danann"]
            if danann_output.success and danann_output.result:
                structure["knowledge_chunks"] = danann_output.result.get("chunks", [])

        # 5. Confidence : on monte si on a vraiment des faits KG, on
        #    reste prudent sur les chunks seuls.
        if compare is not None or facts is not None:
            confidence = 0.85
        elif structure["knowledge_chunks"]:
            confidence = 0.7
        else:
            confidence = 0.4

        return ModuleOutput(
            result=structure,
            confidence=confidence,
            metadata={
                "structure_type": structure_type,
                "datalog_ready": self._datalog_ready,
                "kg_loaded": kg_loaded,
                "kg_load_error": self._kg_load_error,
                "entities_used": entities_used,
                "chunks_used": len(structure["knowledge_chunks"]),
            },
        )

    async def health_check(self) -> bool:
        """Healthy même sans KG (mode dégradé acceptable)."""
        self._try_load_kg()  # log clair au 1er run
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
                "knowledge_graph_query",
            ],
            "datalog_ready": self._datalog_ready,
            "templates": list(self.rules.get("structure_templates", {}).keys()),
            "kg_loaded": self._kg is not None,
        }
