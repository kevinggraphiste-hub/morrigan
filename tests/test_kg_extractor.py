"""Tests de l'extracteur KG (PR 2 du chantier knowledge graph).

Couvre slugify, split_sentences, extract_entities, extract_relations
et populate_graph (intégration avec KnowledgeGraph).
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from modules.ogham.extractor import (
    extract_entities,
    extract_relations,
    populate_graph,
    slugify,
    split_sentences,
)
from modules.ogham.knowledge_graph import KnowledgeGraph


# ─── slugify ───────────────────────────────────────────────────────


def test_slugify_basic():
    assert slugify("Brigid") == "brigid"
    assert slugify("TCP") == "tcp"


def test_slugify_strips_accents():
    assert slugify("Cúchulainn") == "cuchulainn"
    assert slugify("Scáthach") == "scathach"


def test_slugify_multi_word():
    assert slugify("Empire Romain") == "empire_romain"
    assert slugify("Marie Curie") == "marie_curie"


def test_slugify_collapses_spaces_and_punctuation():
    assert slugify("Saint-Pierre") == "saint_pierre"
    assert slugify("  Hello   World  ") == "hello_world"


def test_slugify_empty_becomes_unknown():
    assert slugify("---") == "unknown"
    assert slugify("") == "unknown"


# ─── split_sentences ──────────────────────────────────────────────


def test_split_sentences_basic():
    text = "Brigid est une déesse. Cúchulainn est un héros. Ils sont liés."
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("Brigid")
    assert sentences[-1].startswith("Ils")


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_no_terminal_punct():
    # Pas de découpe possible → renvoie la phrase entière.
    assert split_sentences("juste une phrase sans fin") == [
        "juste une phrase sans fin"
    ]


# ─── extract_entities ─────────────────────────────────────────────


def test_extract_acronyms():
    text = "TCP et UDP sont des protocoles de transport. HTTP utilise TCP."
    ents = extract_entities(text)
    ids = {e.id for e in ents}
    assert "tcp" in ids
    assert "udp" in ids
    assert "http" in ids


def test_extract_acronym_type():
    ents = extract_entities("TCP est un protocole.")
    tcp = next(e for e in ents if e.id == "tcp")
    assert tcp.type == "acronym"


def test_extract_title_case():
    text = "Marie Curie a découvert le radium. Empire Romain a précédé Byzance."
    ents = extract_entities(text)
    ids = {e.id for e in ents}
    assert "marie_curie" in ids
    assert "empire_romain" in ids
    assert "byzance" in ids


def test_extract_preserves_label_accents():
    ents = extract_entities("Cúchulainn est un héros celtique.")
    cuchu = next(e for e in ents if e.id == "cuchulainn")
    assert cuchu.label == "Cúchulainn"  # accent préservé dans label


def test_extract_dedups_by_id():
    text = "Brigid est forte. Brigid règne. Brigid inspire."
    ents = extract_entities(text)
    brigids = [e for e in ents if e.id == "brigid"]
    assert len(brigids) == 1


def test_extract_filters_stopwords():
    """Les déterminants/conjonctions capitalisés ne sont pas extraits."""
    text = "Le ciel est bleu. Et puis Brigid arrive."
    ents = extract_entities(text)
    ids = {e.id for e in ents}
    assert "le" not in ids
    assert "et" not in ids
    assert "puis" not in ids
    # Mais Brigid (vraie entité) est extraite.
    assert "brigid" in ids


def test_extract_ignores_single_char_after_slug():
    """Un slug d'1 lettre n'est pas une entité utile."""
    ents = extract_entities("A est B. C est D.")
    assert ents == []


# ─── extract_relations : patterns explicites ─────────────────────


def test_relation_is_a():
    rels = extract_relations("TCP est un protocole de transport.")
    is_a = [r for r in rels if r.predicate == "is_a"]
    assert len(is_a) >= 1
    assert any(r.subject_id == "tcp" and r.object_id == "protocole" for r in is_a)


def test_relation_is_a_multi_word():
    rels = extract_relations("Brigid est une déesse celtique.")
    is_a = [r for r in rels if r.predicate == "is_a"]
    assert any(r.subject_id == "brigid" and r.object_id == "deesse" for r in is_a)


def test_relation_has():
    rels = extract_relations("Le système possède une mémoire.")
    has = [r for r in rels if r.predicate == "has"]
    # Système est filtré comme stopword potentiel ? Non, "système" non capitalisé
    # mais le pattern attend une entité capitalisée. Vérifions sur capitalisé.
    rels2 = extract_relations("Morrigan possède une mémoire vectorielle.")
    has2 = [r for r in rels2 if r.predicate == "has"]
    assert any(r.subject_id == "morrigan" for r in has2)


def test_relation_uses():
    rels = extract_relations("Morrigan utilise un embedder MiniLM.")
    uses = [r for r in rels if r.predicate == "uses"]
    assert any(r.subject_id == "morrigan" for r in uses)


def test_relation_source_propagated():
    rels = extract_relations("TCP est un protocole.", source="doc_reseau.md")
    assert all(r.source == "doc_reseau.md" for r in rels)


def test_relation_no_self_loop():
    """X est un X ne doit pas produire de boucle."""
    rels = extract_relations("Brigid est Brigid.")
    self_loops = [r for r in rels if r.subject_id == r.object_id]
    assert self_loops == []


# ─── extract_relations : co-occurrence ───────────────────────────


def test_co_occurrence_in_same_sentence():
    rels = extract_relations("Brigid et Cúchulainn sont des figures celtiques.")
    co = [r for r in rels if r.predicate == "co_occurs_with"]
    # Au moins brigid ↔ cuchulainn devrait être co-occurent.
    pair = {("brigid", "cuchulainn"), ("cuchulainn", "brigid")}
    assert any((r.subject_id, r.object_id) in pair for r in co)


def test_co_occurrence_has_low_confidence():
    rels = extract_relations("Brigid et Cúchulainn règnent.")
    for r in rels:
        if r.predicate == "co_occurs_with":
            assert r.confidence < 0.5  # confidence faible par design


def test_co_occurrence_does_not_cross_sentences():
    """Brigid et Cúchulainn dans deux phrases distinctes ne co-occurent pas."""
    text = "Brigid règne sur le feu. Cúchulainn protège l'Ulster."
    rels = extract_relations(text)
    co = [
        r for r in rels
        if r.predicate == "co_occurs_with"
        and {r.subject_id, r.object_id} == {"brigid", "cuchulainn"}
    ]
    assert co == []


# ─── populate_graph (intégration KG) ──────────────────────────────


def test_populate_graph_returns_counts():
    kg = KnowledgeGraph()
    n_ents, n_rels = populate_graph(
        kg,
        "TCP est un protocole. UDP est un protocole. TCP utilise IP.",
        source="reseau.md",
    )
    assert n_ents >= 3  # TCP, UDP, IP au minimum
    assert n_rels >= 2
    assert "tcp" in kg
    assert "udp" in kg
    assert "ip" in kg


def test_populate_graph_aggregates_duplicates():
    """Deux passes du même texte → 1 triplet distinct (count=2)."""
    kg = KnowledgeGraph()
    text = "TCP est un protocole."
    populate_graph(kg, text, source="a")
    populate_graph(kg, text, source="b")
    # Un seul triplet distinct.
    is_a = [r for r in kg.relations(predicate="is_a") if r.subject_id == "tcp"]
    assert len(is_a) == 1
    # Le count interne est passé à 2 et les deux sources sont gardées.
    edge_data = kg._graph.edges["tcp", "protocole"]["predicates"]["is_a"]
    assert edge_data["count"] == 2
    assert set(edge_data["sources"]) == {"a", "b"}


def test_populate_graph_preserves_entity_metadata():
    """Une entité ajoutée d'abord (par extract_entities) garde son label/type."""
    kg = KnowledgeGraph()
    populate_graph(kg, "TCP est un protocole.")
    tcp = kg.get_entity("tcp")
    assert tcp is not None
    assert tcp.label == "TCP"
    assert tcp.type == "acronym"
