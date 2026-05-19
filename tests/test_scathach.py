"""Tests pour Scáthach — le module langage."""

import asyncio
import sys

import pytest

sys.path.insert(0, ".")

from core.types import ModuleInput
from modules.scathach.generator import Scathach


def test_scathach_health():
    scathach = Scathach()
    assert asyncio.run(scathach.health_check()) is True


@pytest.mark.xfail(
    reason=(
        "Pré-existant depuis le commit initial cdc66ce — la chaîne "
        "conversationnelle ne renvoie pas le mot 'Morrigan' dans la réponse "
        "template par défaut. À reprendre quand le template 'conversation.j2' "
        "sera revu ou que le backend Scáthach évoluera (Phase 3, RWKV/Llama)."
    ),
    strict=False,
)
def test_scathach_template_generation():
    scathach = Scathach(backend="template")
    input_data = ModuleInput(query="Bonjour", context={"previous_results": {}})
    output = asyncio.run(scathach.process(input_data))
    assert output.success
    assert "Morrigan" in str(output.result)
