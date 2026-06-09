"""Garde-fous des bugs de correction (backlog audit 2026-05-31).

Quatre bugs distincts, un test (ou groupe) chacun :
  1. KG.from_dict perdait des sources quand `count < len(sources)`.
  2. An Dagda choisissait un domaine arbitraire en cas d'égalité de hits.
  3. Le filtre domaine de Danann vidait la fenêtre → faux « je ne sais pas ».
  4. process_stream n'enregistrait pas la requête (/stats) si le stream levait.
"""

import asyncio
import sys

sys.path.insert(0, ".")

from core.dagda import AnDagda
from core.types import ModuleInput, ModuleOutput
from modules.danann.store import Danann
from modules.ogham.knowledge_graph import KnowledgeGraph, Relation


# ─── Bug 1 : KG.from_dict — restauration verbatim, zéro source perdue ───


def _is_a_pred(kg: KnowledgeGraph) -> dict:
    return kg.to_dict()["edges"][0]["predicates"]["is_a"]


def test_from_dict_roundtrip_is_stable():
    kg = KnowledgeGraph()
    kg.add_relation(Relation("tcp", "is_a", "proto", confidence=0.5, source="docA"))
    kg.add_relation(Relation("tcp", "is_a", "proto", confidence=0.9, source="docB"))
    kg.add_relation(Relation("tcp", "is_a", "proto", confidence=0.7, source="docA"))

    once = KnowledgeGraph.from_dict(kg.to_dict())
    twice = KnowledgeGraph.from_dict(once.to_dict())

    for tag, g in (("orig", kg), ("once", once), ("twice", twice)):
        p = _is_a_pred(g)
        assert p["count"] == 3, tag
        assert p["confidence"] == 0.9, tag
        assert p["sources"] == ["docA", "docB"], tag


def test_from_dict_preserves_sources_when_count_missing():
    """Fichier externe sans `count` : toutes les sources doivent survivre
    (l'ancien replay n'en gardait qu'une — sources[0])."""
    payload = {
        "schema_version": 1,
        "entities": [],
        "edges": [{
            "subject": "tcp", "object": "proto",
            "predicates": {"is_a": {"confidence": 0.9, "sources": ["a", "b", "c"]}},
        }],
    }
    p = _is_a_pred(KnowledgeGraph.from_dict(payload))
    assert p["sources"] == ["a", "b", "c"]
    assert p["count"] == 3  # borné à >= nb sources distinctes


def test_from_dict_preserves_sources_when_count_underestimates():
    """count < len(sources) : l'ancien code droppait les sources au-delà de count."""
    payload = {
        "schema_version": 1,
        "entities": [],
        "edges": [{
            "subject": "tcp", "object": "proto",
            "predicates": {"is_a": {"count": 2, "confidence": 0.9,
                                    "sources": ["a", "b", "c", "d"]}},
        }],
    }
    p = _is_a_pred(KnowledgeGraph.from_dict(payload))
    assert p["sources"] == ["a", "b", "c", "d"]
    assert p["count"] == 4


# ─── Bug 2 : domain_hint — égalité ⇒ None (pas de choix arbitraire) ───


def test_domain_hint_tie_returns_none():
    """Une query qui matche autant un domaine qu'un autre ne doit PAS
    trancher arbitrairement (l'ancien max() suivait l'ordre du dict)."""
    dagda = AnDagda()
    # "rwkv" → ia, "python" → code : un hit chacun → égalité.
    hint = dagda._detect_domain_hint("rwkv en python")
    assert hint is None


def test_domain_hint_clear_winner_kept():
    """Pas de régression : un domaine nettement majoritaire reste détecté."""
    dagda = AnDagda()
    hint = dagda._detect_domain_hint("transformer llm embedding neurone")
    assert hint == "ia"


# ─── Bug 3 : filtre domaine best-effort (repli si vide) ───


def test_domain_filter_falls_back_when_empty():
    """Un domain_hint sans aucun chunk correspondant ne doit pas vider la
    fenêtre de candidats (sinon RAG strict → faux « je ne sais pas »)."""
    danann = Danann()
    danann.index(
        ["TCP est un protocole réseau fiable.", "UDP est un protocole sans connexion."],
        [{"domain": "reseau"}, {"domain": "reseau"}],
    )
    # Aucun chunk "mythologie" → l'ancien filtre dur renvoyait [].
    results = danann.search("protocole fiable", domain="mythologie")
    assert results, "le filtre domaine ne doit jamais vider le retrieval"


def test_domain_filter_applies_when_matches_exist():
    """Pas de régression : quand des chunks du domaine existent, on filtre."""
    danann = Danann()
    danann.index(
        ["TCP protocole réseau.", "Le Dagda est un dieu celtique."],
        [{"domain": "reseau"}, {"domain": "mythologie"}],
    )
    results = danann.search("dieu celtique", domain="mythologie")
    assert results
    assert all(meta.get("domain") == "mythologie" for _, _, meta in results)


# ─── Bug 4 : process_stream enregistre la requête même si le stream lève ───


class _OkModule:
    async def process(self, _inp: ModuleInput) -> ModuleOutput:
        return ModuleOutput(result="contexte")


class _BoomScathach:
    """Dernier module : commence à streamer puis lève en cours de route."""

    last_generated_by = None

    async def process(self, _inp: ModuleInput) -> ModuleOutput:
        return ModuleOutput(result="x")

    async def stream(self, _inp: ModuleInput):
        yield "début…"
        raise RuntimeError("génération plantée")


def test_process_stream_records_query_on_failure():
    dagda = AnDagda()
    # "Salut" → CONVERSATION → modules ["cauldron", "scathach"].
    dagda.register_module("cauldron", _OkModule())
    dagda.register_module("scathach", _BoomScathach())

    async def _run():
        pieces = []
        async for piece in dagda.process_stream("Salut, comment ça va ?"):
            pieces.append(piece)
        return pieces

    raised = None
    try:
        asyncio.run(_run())
    except RuntimeError as e:
        raised = e

    assert raised is not None and "plantée" in str(raised)
    # Le bug : sans try/finally, queries restait à 0.
    assert dagda.stats["queries"] == 1
    assert dagda.last_routing is not None
