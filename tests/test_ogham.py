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
    ogham = Ogham()
    input_data = ModuleInput(query="Compare TCP et UDP", context={})
    output = asyncio.run(ogham.process(input_data))
    assert output.success
    assert output.result["type"] == "structured_response"
