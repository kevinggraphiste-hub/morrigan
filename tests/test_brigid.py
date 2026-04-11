"""Tests pour Brigid — le réseau LNN."""

import asyncio
import sys

sys.path.insert(0, ".")

from core.types import ModuleInput
from modules.brigid.model import Brigid


def test_brigid_health():
    brigid = Brigid()
    assert asyncio.run(brigid.health_check()) is True


def test_brigid_process():
    brigid = Brigid()
    input_data = ModuleInput(query="Test query")
    output = asyncio.run(brigid.process(input_data))
    assert output.success
    assert output.metadata["phase"] == 0


def test_brigid_capabilities():
    brigid = Brigid()
    caps = brigid.get_capabilities()
    assert caps["name"] == "Brigid"
    assert "intent_classification" in caps["capabilities"]
