"""Tests d'intégration Ogham ↔ KnowledgeGraph (PR 4 du chantier KG).

Couvre les nouveaux comportements de `Ogham.process` :
  - dégradation gracieuse quand le KG n'est pas disponible
  - détection des requêtes « compare X et Y » + appel à `kg.compare`
  - détection des requêtes « qu'est-ce X » / « qui est X » + appel à
    `kg.facts_about`
  - propagation propre dans `ModuleOutput.metadata`

Fixture session-scopée : construit un mini-KG en mémoire pour les
tests qui ont besoin de données. On évite de dépendre du KG produit
par `scripts/build_knowledge_graph.py` (qui est gitignoré et donc
absent sur main fresh / CI tests).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from core.types import ModuleInput
from modules.ogham.engine import Ogham
from modules.ogham.knowledge_graph import Entity, KnowledgeGraph, Relation


@pytest.fixture
def small_kg_file(tmp_path: Path) -> Path:
    """Crée un mini-KG TCP/UDP/IP/Brigid et le sauve en JSON."""
    kg = KnowledgeGraph()
    for eid, label, t in [
        ("tcp", "TCP", "acronym"),
        ("udp", "UDP", "acronym"),
        ("ip", "IP", "acronym"),
        ("protocole", "Protocole", "concept"),
        ("fiable", "Fiable", "concept"),
        ("rapide", "Rapide", "concept"),
        ("brigid", "Brigid", "concept"),
        ("deesse", "Déesse", "concept"),
    ]:
        kg.add_entity(Entity(id=eid, label=label, type=t))

    kg.add_relation(Relation(subject_id="tcp", predicate="is_a", object_id="protocole"))
    kg.add_relation(Relation(subject_id="udp", predicate="is_a", object_id="protocole"))
    kg.add_relation(Relation(subject_id="tcp", predicate="uses", object_id="ip"))
    kg.add_relation(Relation(subject_id="udp", predicate="uses", object_id="ip"))
    kg.add_relation(Relation(subject_id="tcp", predicate="is_a", object_id="fiable"))
    kg.add_relation(Relation(subject_id="udp", predicate="is_a", object_id="rapide"))
    kg.add_relation(Relation(subject_id="brigid", predicate="is_a", object_id="deesse"))

    out = tmp_path / "kg.json"
    kg.save(out)
    return out


# ─── Dégradation gracieuse sans KG ─────────────────────────────────


def test_process_without_kg_still_works(tmp_path: Path):
    """Ogham sans KG : type='structured_response' quand même."""
    ogham = Ogham(kg_path=tmp_path / "nope.json")
    out = asyncio.run(ogham.process(ModuleInput(query="Compare TCP et UDP")))
    assert out.success
    assert out.result["type"] == "structured_response"
    assert out.result["compare"] is None  # pas de KG → pas de compare
    assert out.metadata["kg_loaded"] is False
    assert out.metadata["kg_load_error"]  # message explicite stocké


def test_process_without_kg_has_lower_confidence(tmp_path: Path):
    """Sans KG ni chunks, confidence reste basse (0.4)."""
    ogham = Ogham(kg_path=tmp_path / "nope.json")
    out = asyncio.run(ogham.process(ModuleInput(query="Compare TCP et UDP")))
    assert out.confidence < 0.6


# ─── Avec KG : compare ─────────────────────────────────────────────


def test_process_compare_uses_kg(small_kg_file: Path):
    ogham = Ogham(kg_path=small_kg_file)
    out = asyncio.run(ogham.process(ModuleInput(query="Compare TCP et UDP")))
    assert out.success
    assert out.result["structure_type"] == "comparison"
    assert out.metadata["kg_loaded"] is True
    assert out.metadata["entities_used"] == ["tcp", "udp"]

    compare = out.result["compare"]
    assert compare is not None
    assert compare["a"] == "tcp"
    assert compare["b"] == "udp"

    # `protocole` et `ip` sont communs aux deux.
    common_ids = {n["id"] for n in compare["common_neighbors"]}
    assert "protocole" in common_ids
    assert "ip" in common_ids

    # `fiable` est spécifique à TCP, `rapide` à UDP.
    a_only_ids = {n["id"] for n in compare["a_only"]}
    b_only_ids = {n["id"] for n in compare["b_only"]}
    assert "fiable" in a_only_ids
    assert "rapide" in b_only_ids


def test_compare_confidence_higher_when_kg_hits(small_kg_file: Path):
    """KG qui trouve qq chose → confidence remonte (0.85)."""
    ogham = Ogham(kg_path=small_kg_file)
    out = asyncio.run(ogham.process(ModuleInput(query="Compare TCP et UDP")))
    assert out.confidence >= 0.8


def test_compare_pattern_difference_entre(small_kg_file: Path):
    """« Différence entre X et Y » doit aussi déclencher kg.compare."""
    ogham = Ogham(kg_path=small_kg_file)
    out = asyncio.run(ogham.process(ModuleInput(query="Différence entre TCP et UDP ?")))
    assert out.result["compare"] is not None
    assert out.result["compare"]["a"] == "tcp"
    assert out.result["compare"]["b"] == "udp"


def test_compare_unknown_entity_falls_back(small_kg_file: Path):
    """Compare X et Y où X inconnu → pas de compare, mais type stable."""
    ogham = Ogham(kg_path=small_kg_file)
    out = asyncio.run(ogham.process(ModuleInput(query="Compare Quetzalcoatl et UDP")))
    assert out.success
    assert out.result["type"] == "structured_response"
    assert out.result["compare"] is None  # X non résolu → on n'invente pas


# ─── Avec KG : définition / facts_about ────────────────────────────


def test_process_definition_uses_kg(small_kg_file: Path):
    ogham = Ogham(kg_path=small_kg_file)
    out = asyncio.run(ogham.process(ModuleInput(query="Qui est Brigid ?")))
    assert out.success
    assert out.result["structure_type"] == "explanation"
    assert out.metadata["entities_used"] == ["brigid"]

    facts = out.result["facts"]
    assert facts is not None
    assert facts["entity"] == "brigid"
    # Brigid is_a deesse dans notre mini-KG.
    assert any(
        r["subject"] == "brigid" and r["predicate"] == "is_a" and r["object"] == "deesse"
        for r in facts["relations"]
    )


def test_definition_pattern_qu_est_ce(small_kg_file: Path):
    ogham = Ogham(kg_path=small_kg_file)
    out = asyncio.run(ogham.process(ModuleInput(query="Qu'est-ce que TCP ?")))
    facts = out.result["facts"]
    assert facts is not None
    assert facts["entity"] == "tcp"


def test_definition_pattern_parle_moi_de(small_kg_file: Path):
    ogham = Ogham(kg_path=small_kg_file)
    out = asyncio.run(ogham.process(ModuleInput(query="Parle-moi de Brigid.")))
    facts = out.result["facts"]
    assert facts is not None
    assert facts["entity"] == "brigid"


def test_definition_unknown_entity_yields_none(small_kg_file: Path):
    """Définition d'un truc absent du KG → facts=None, pas de crash."""
    ogham = Ogham(kg_path=small_kg_file)
    out = asyncio.run(ogham.process(ModuleInput(query="Qui est Hermès ?")))
    assert out.success
    assert out.result["type"] == "structured_response"
    assert out.result["facts"] is None


# ─── Capabilities ──────────────────────────────────────────────────


def test_capabilities_advertise_kg(small_kg_file: Path):
    ogham = Ogham(kg_path=small_kg_file)
    # Force le chargement.
    asyncio.run(ogham.health_check())
    caps = ogham.get_capabilities()
    assert "knowledge_graph_query" in caps["capabilities"]
    assert caps["kg_loaded"] is True


def test_capabilities_without_kg(tmp_path: Path):
    ogham = Ogham(kg_path=tmp_path / "nope.json")
    caps = ogham.get_capabilities()
    assert "knowledge_graph_query" in caps["capabilities"]  # déclarée même si non chargée
    assert caps["kg_loaded"] is False
