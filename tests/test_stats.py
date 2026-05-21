"""Tests de l'observabilité An Dagda (/stats)."""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, ".")

from core.dagda import AnDagda
from core.types import ModuleInput, ModuleOutput
from modules.brigid.model import IntentClassification
from modules.scathach.generator import Scathach


async def _collect(agen):
    return [x async for x in agen]


class FakeBrigid:
    """Brigid factice : renvoie une classification contrôlée."""

    def __init__(self, label: str, confidence: float) -> None:
        labels = ["factual", "reasoning", "creative", "conversation", "complex", "code"]
        # Probas jouet : confidence sur le label, reste réparti.
        rest = (1.0 - confidence) / (len(labels) - 1)
        self._probas = {lbl: (confidence if lbl == label else rest) for lbl in labels}
        self._classif = IntentClassification(label, confidence, self._probas)

    def classify_intent(self, query):
        return self._classif

    # MorriganModule-ish (pas utilisé pour le routage stream ici)
    def get_capabilities(self):
        return {"name": "Brigid"}


class FakeStreamScathach(Scathach):
    """Scáthach factice qui ne charge pas de modèle (backend rwkv simulé)."""


# ─── État initial ──────────────────────────────────────────────────


def test_stats_empty_at_start():
    d = AnDagda()
    s = d.format_stats()
    assert "Requêtes traitées   : 0" in s
    assert d.stats["queries"] == 0


# ─── Comptage après requêtes (process) ────────────────────────────


def test_record_query_updates_counters():
    d = AnDagda()
    # Scáthach template (pas de modèle) pour avoir un generated_by.
    d.register_module("cauldron", _Cauldron())
    d.register_module("scathach", Scathach(backend="template"))

    asyncio.run(d.process("Bonjour"))
    asyncio.run(d.process("Salut"))

    assert d.stats["queries"] == 2
    assert d.stats["by_type"].get("conversation", 0) >= 1
    # generated_by template comptabilisé
    assert d.stats["by_generated_by"].get("template", 0) >= 1
    assert d.last_routing is not None


def test_format_stats_shows_last_query():
    d = AnDagda()
    d.register_module("cauldron", _Cauldron())
    d.register_module("scathach", Scathach(backend="template"))
    asyncio.run(d.process("Bonjour"))
    s = d.format_stats()
    assert "Dernière requête" in s
    assert "conversation" in s
    assert "généré par" in s


# ─── Brigid capturé dans la trace ─────────────────────────────────


def test_brigid_classification_captured_in_stats():
    d = AnDagda()
    d.register_module("brigid", FakeBrigid("factual", 0.82))
    d.register_module("danann", _Empty())
    d.register_module("ogham", _Empty())
    d.register_module("scathach", Scathach(backend="template"))

    asyncio.run(d.process("Qui est Alan Turing ?"))
    s = d.format_stats()
    # Routage via Brigid + ligne brigid avec probas.
    assert "brigid" in s.lower()
    assert d._last_brigid is not None
    assert d._last_brigid.label == "factual"


def test_brigid_reset_when_fence_shortcut():
    """Un fence markdown court-circuite Brigid → pas de trace brigid."""
    d = AnDagda()
    d.register_module("brigid", FakeBrigid("factual", 0.99))
    d.register_module("scathach", Scathach(backend="template"))
    from modules.morrigan_code.module import MorriganCode
    d.register_module("morrigan_code", MorriganCode())

    asyncio.run(d.process("Vérifie :\n```python\nprint(1)\n```"))
    assert d._last_brigid is None  # Brigid pas consultée (fence prioritaire)
    assert d.last_routing.query_type.value == "code"


# ─── Stats via process_stream ─────────────────────────────────────


def test_stats_recorded_via_stream():
    d = AnDagda()
    d.register_module("cauldron", _Cauldron())
    d.register_module("scathach", Scathach(backend="template"))
    asyncio.run(_collect(d.process_stream("Bonjour")))
    assert d.stats["queries"] == 1
    assert d.last_generated_by == "template"


# ─── Doubles helpers minimalistes ─────────────────────────────────


class _Empty:
    """Module factice neutre (health/process no-op)."""

    async def process(self, inp):
        return ModuleOutput(result=None, confidence=0.0)

    async def health_check(self):
        return True

    def get_capabilities(self):
        return {"name": "empty"}


class _Cauldron:
    """Cauldron factice : fournit un previous_results non vide pour la conv."""

    async def process(self, inp):
        return ModuleOutput(result={"turn_count": 0}, confidence=0.5)

    async def health_check(self):
        return True

    def get_capabilities(self):
        return {"name": "cauldron"}
