"""Tests pour Scáthach — le module langage."""

import asyncio
import sys

sys.path.insert(0, ".")

from core.types import ModuleInput
from modules.scathach.generator import Scathach


def test_scathach_health():
    scathach = Scathach()
    assert asyncio.run(scathach.health_check()) is True


def test_scathach_template_generation():
    scathach = Scathach(backend="template")
    input_data = ModuleInput(query="Bonjour", context={"previous_results": {}})
    output = asyncio.run(scathach.process(input_data))
    assert output.success
    assert "Morrigan" in str(output.result)
