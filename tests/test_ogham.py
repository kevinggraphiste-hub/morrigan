"""Tests pour Ogham — le moteur symbolique."""

import asyncio
import sys

sys.path.insert(0, ".")

from core.types import ModuleInput
from modules.ogham.engine import Ogham


def test_ogham_health():
    ogham = Ogham()
    assert asyncio.run(ogham.health_check()) is True


def test_ogham_process():
    """Test rétabli (PR 4 du chantier KG, ex-xfail depuis cdc66ce).

    Avec ou sans KG chargé, Ogham.process doit toujours renvoyer un
    `ModuleOutput.result["type"] == "structured_response"` — c'est le
    contrat stable vis-à-vis des consommateurs (Scáthach).
    """
    ogham = Ogham()
    input_data = ModuleInput(query="Compare TCP et UDP", context={})
    output = asyncio.run(ogham.process(input_data))
    assert output.success
    assert output.result["type"] == "structured_response"
