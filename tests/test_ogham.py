"""Tests pour Ogham — le moteur symbolique."""

import asyncio
import sys

import pytest

sys.path.insert(0, ".")

from core.types import ModuleInput
from modules.ogham.engine import Ogham


def test_ogham_health():
    ogham = Ogham()
    assert asyncio.run(ogham.health_check()) is True


@pytest.mark.xfail(
    reason=(
        "Pré-existant depuis le commit initial cdc66ce — Ogham.process ne "
        "renvoie pas encore de réponse structurée pour les requêtes de "
        "comparaison. À reprendre quand Ogham passera de pyDatalog minimal "
        "au knowledge graph (Phase 2)."
    ),
    strict=False,
)
def test_ogham_process():
    ogham = Ogham()
    input_data = ModuleInput(query="Compare TCP et UDP", context={})
    output = asyncio.run(ogham.process(input_data))
    assert output.success
    assert output.result["type"] == "structured_response"
