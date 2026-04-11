"""Tests d'intégration — pipeline complet."""

import asyncio
import sys

sys.path.insert(0, ".")

from core.dagda import AnDagda
from modules.brigid.model import Brigid
from modules.ogham.engine import Ogham
from modules.danann.store import Danann
from modules.scathach.generator import Scathach
from modules.cauldron.memory import Cauldron


def _create_dagda() -> AnDagda:
    """Crée un An Dagda avec tous les modules."""
    dagda = AnDagda()
    dagda.register_module("brigid", Brigid())
    dagda.register_module("ogham", Ogham())
    dagda.register_module("danann", Danann())
    dagda.register_module("scathach", Scathach())
    dagda.register_module("cauldron", Cauldron())
    return dagda


def test_full_pipeline():
    """Le pipeline complet doit retourner une réponse."""
    dagda = _create_dagda()
    response = asyncio.run(dagda.process("Qui est Alan Turing ?"))
    assert response is not None
    assert len(response) > 0


def test_creative_pipeline():
    """Le pipeline créatif doit fonctionner."""
    dagda = _create_dagda()
    response = asyncio.run(dagda.process("Écris un haiku"))
    assert response is not None


def test_conversation_pipeline():
    """Le pipeline conversationnel doit fonctionner."""
    dagda = _create_dagda()
    response = asyncio.run(dagda.process("Salut !"))
    assert response is not None
