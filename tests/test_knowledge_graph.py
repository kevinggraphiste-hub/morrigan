"""Tests du KnowledgeGraph (PR 1 du chantier KG Ogham).

Couvre le modèle de données, les invariants de mutation, les requêtes
(neighbors, compare, facts_about) et la sérialisation JSON. Pas
d'extraction ici (PR 2), pas d'intégration Ogham (PR 4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

# networkx est dans requirements.txt depuis Phase 1 ; pas de
# importorskip nécessaire (mais ça ne coûte rien d'être défensif).
pytest.importorskip("networkx")

from modules.ogham.knowledge_graph import (
    Entity,
    KnowledgeGraph,
    Relation,
)


# ─── Entity / Relation : validation ────────────────────────────────


def test_entity_rejects_empty_id():
    with pytest.raises(ValueError, match="id ne peut être vide"):
        Entity(id="", label="X")


def test_entity_rejects_empty_label():
    with pytest.raises(ValueError, match="label ne peut être vide"):
        Entity(id="x", label="")


def test_entity_default_type_is_concept():
    e = Entity(id="x", label="X")
    assert e.type == "concept"
    assert e.attributes == {}


def test_relation_confidence_must_be_in_range():
    with pytest.raises(ValueError, match="confidence"):
        Relation(subject_id="a", predicate="p", object_id="b", confidence=1.5)
    with pytest.raises(ValueError, match="confidence"):
        Relation(subject_id="a", predicate="p", object_id="b", confidence=-0.1)


def test_relation_predicate_must_be_non_empty():
    with pytest.raises(ValueError, match="predicate"):
        Relation(subject_id="a", predicate="", object_id="b")


# ─── KG : structure et accès ──────────────────────────────────────


def test_empty_kg():
    kg = KnowledgeGraph()
    assert len(kg) == 0
    assert kg.relation_count == 0
    assert kg.entities() == []
    assert kg.relations() == []
    assert "anything" not in kg


def test_add_and_get_entity():
    kg = KnowledgeGraph()
    e = Entity(id="tcp", label="TCP", type="protocol")
    kg.add_entity(e)
    assert "tcp" in kg
    got = kg.get_entity("tcp")
    assert got is not None
    assert got.label == "TCP"
    assert got.type == "protocol"


def test_add_entity_merges_attributes():
    kg = KnowledgeGraph()
    kg.add_entity(Entity(id="tcp", label="TCP", attributes={"layer": 4}))
    kg.add_entity(Entity(id="tcp", label="TCP", attributes={"port": "*"}))
    got = kg.get_entity("tcp")
    assert got.attributes == {"layer": 4, "port": "*"}


def test_entities_filter_by_type():
    kg = KnowledgeGraph()
    kg.add_entity(Entity(id="tcp", label="TCP", type="protocol"))
    kg.add_entity(Entity(id="brigid", label="Brigid", type="deity"))
    protocols = kg.entities(type="protocol")
    assert len(protocols) == 1
    assert protocols[0].id == "tcp"


# ─── add_relation : invariants et agrégation ──────────────────────


def test_add_relation_creates_missing_entities():
    kg = KnowledgeGraph()
    kg.add_relation(Relation(
        subject_id="tcp", predicate="is_a", object_id="protocol"
    ))
    assert "tcp" in kg and "protocol" in kg
    assert kg.relation_count == 1


def test_add_duplicate_relation_aggregates():
    kg = KnowledgeGraph()
    r = Relation(
        subject_id="tcp", predicate="is_a", object_id="protocol",
        confidence=0.6, source="doc_a",
    )
    kg.add_relation(r)
    kg.add_relation(Relation(
        subject_id="tcp", predicate="is_a", object_id="protocol",
        confidence=0.9, source="doc_b",
    ))
    # Toujours 1 triplet distinct, mais count=2
    assert kg.relation_count == 1
    # Lecture via neighbors : un seul prédicat
    out = kg.neighbors("tcp")
    assert len(out) == 1
    # Confidence max retenue
    rels = kg.relations(subject_id="tcp")
    assert len(rels) == 1
    assert rels[0].confidence == 0.9


def test_relations_with_different_predicates_coexist():
    kg = KnowledgeGraph()
    kg.add_relation(Relation(subject_id="tcp", predicate="is_a", object_id="protocol"))
    kg.add_relation(Relation(subject_id="tcp", predicate="competes_with", object_id="udp"))
    assert kg.relation_count == 2
    assert {r.predicate for r in kg.relations(subject_id="tcp")} == {
        "is_a", "competes_with",
    }


# ─── Requêtes : neighbors, facts_about, compare ───────────────────


def _build_demo_kg() -> KnowledgeGraph:
    """Petit KG d'exemple inspiré du domaine réseau, partagé par les tests."""
    kg = KnowledgeGraph()
    for eid, label, t in [
        ("tcp", "TCP", "protocol"),
        ("udp", "UDP", "protocol"),
        ("ip", "IP", "protocol"),
        ("transport", "Couche Transport", "concept"),
        ("connectionless", "Sans connexion", "property"),
        ("reliable", "Fiable", "property"),
    ]:
        kg.add_entity(Entity(id=eid, label=label, type=t))

    kg.add_relation(Relation(subject_id="tcp", predicate="is_a", object_id="transport"))
    kg.add_relation(Relation(subject_id="udp", predicate="is_a", object_id="transport"))
    kg.add_relation(Relation(subject_id="tcp", predicate="uses", object_id="ip"))
    kg.add_relation(Relation(subject_id="udp", predicate="uses", object_id="ip"))
    kg.add_relation(Relation(subject_id="tcp", predicate="has_property", object_id="reliable"))
    kg.add_relation(Relation(subject_id="udp", predicate="has_property", object_id="connectionless"))
    return kg


def test_neighbors_returns_pairs():
    kg = _build_demo_kg()
    out = kg.neighbors("tcp")
    # 3 voisins : transport, ip, reliable
    ids = sorted(n.id for n, _ in out)
    assert ids == ["ip", "reliable", "transport"]


def test_neighbors_filter_by_predicate():
    kg = _build_demo_kg()
    out = kg.neighbors("tcp", predicate="uses")
    assert len(out) == 1
    assert out[0][0].id == "ip"


def test_neighbors_unknown_entity_returns_empty():
    kg = _build_demo_kg()
    assert kg.neighbors("inconnu") == []


def test_facts_about_returns_in_and_out():
    kg = _build_demo_kg()
    facts = kg.facts_about("transport")
    # `transport` est uniquement object (cible de "is_a"), pas subject
    assert all(r.object_id == "transport" for r in facts)
    assert len(facts) == 2  # tcp et udp


def test_compare_finds_common_and_differences():
    kg = _build_demo_kg()
    result = kg.compare("tcp", "udp")

    # Voisins communs : transport (is_a) et ip (uses)
    common_ids = sorted(e.id for e, _ in result["common_neighbors"])
    assert common_ids == ["ip", "transport"]

    # a_only de tcp : reliable
    a_only_ids = sorted(e.id for e, _ in result["a_only"])
    assert a_only_ids == ["reliable"]

    # b_only de udp : connectionless
    b_only_ids = sorted(e.id for e, _ in result["b_only"])
    assert b_only_ids == ["connectionless"]

    # Pas de relation directe tcp ↔ udp dans la démo
    assert result["direct_relations"] == []


def test_compare_picks_up_direct_relation():
    kg = KnowledgeGraph()
    kg.add_relation(Relation(subject_id="tcp", predicate="competes_with", object_id="udp"))
    result = kg.compare("tcp", "udp")
    assert len(result["direct_relations"]) == 1
    assert result["direct_relations"][0].predicate == "competes_with"


# ─── Persistance JSON ─────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path: Path):
    kg = _build_demo_kg()
    # Ajoute des duplicates avec sources pour tester la préservation.
    kg.add_relation(Relation(
        subject_id="tcp", predicate="is_a", object_id="transport",
        source="doc_a",
    ))
    kg.add_relation(Relation(
        subject_id="tcp", predicate="is_a", object_id="transport",
        source="doc_b",
    ))

    path = tmp_path / "kg.json"
    kg.save(path)
    assert path.exists()

    loaded = KnowledgeGraph.load(path)
    assert len(loaded) == len(kg)
    assert loaded.relation_count == kg.relation_count

    # Le compteur d'agrégation est préservé.
    rels = loaded.relations(subject_id="tcp", predicate="is_a", object_id="transport")
    assert len(rels) == 1  # 1 triplet distinct
    # On vérifie via la structure interne que le count est bien à 3
    # (1 dans _build_demo_kg + 2 ajouts).
    edge = loaded._graph.edges["tcp", "transport"]
    assert edge["predicates"]["is_a"]["count"] == 3
    assert set(edge["predicates"]["is_a"]["sources"]) == {"doc_a", "doc_b"}


def test_save_produces_readable_json(tmp_path: Path):
    kg = KnowledgeGraph()
    kg.add_entity(Entity(id="brigid", label="Brigid", type="deity"))
    path = tmp_path / "kg.json"
    kg.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert any(e["id"] == "brigid" and e["label"] == "Brigid" for e in data["entities"])


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        KnowledgeGraph.load(tmp_path / "nope.json")


def test_load_wrong_schema_version_raises(tmp_path: Path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema_version": 42, "entities": [], "edges": []}))
    with pytest.raises(ValueError, match="Schema version"):
        KnowledgeGraph.load(path)
