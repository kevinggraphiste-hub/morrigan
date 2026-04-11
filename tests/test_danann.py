"""Tests pour Danann — la mémoire vectorielle."""

import asyncio
import sys

sys.path.insert(0, ".")

from core.types import ModuleInput
from modules.danann.store import Danann


def test_danann_health():
    danann = Danann()
    assert asyncio.run(danann.health_check()) is True


def test_danann_process():
    danann = Danann()
    input_data = ModuleInput(query="Qu'est-ce que TCP ?")
    output = asyncio.run(danann.process(input_data))
    assert output.success
