"""Tests pour Brigid — le réseau LNN.

Tests basiques de l'interface MorriganModule (health + capabilities).
Les vraies validations d'inférence + intégration An Dagda vivent dans
`tests/test_brigid_inference.py` (gated par torch/ncps).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, ".")

from core.types import ModuleInput
from modules.brigid.model import Brigid


def test_brigid_health():
    """health_check renvoie True même sans checkpoint (mode dégradé OK)."""
    # On force un chemin inexistant pour ne pas dépendre du state du repo.
    brigid = Brigid(checkpoint_path=Path("/nonexistent/brigid_no.pt"))
    assert asyncio.run(brigid.health_check()) is True


def test_brigid_process_without_checkpoint_degrades_cleanly():
    """Sans checkpoint, process renvoie un output dégradé (pas d'exception).

    Comportement attendu pour PR C : Brigid n'a pas le checkpoint
    (dev avant entraînement, env sans data/models/), An Dagda
    fallback aux heuristiques. process() doit signaler proprement,
    pas planter.
    """
    brigid = Brigid(checkpoint_path=Path("/nonexistent/brigid_no.pt"))
    output = asyncio.run(brigid.process(ModuleInput(query="Test query")))
    assert not output.success  # errors non vide
    assert output.confidence == 0.0
    assert output.metadata["phase"] == 2
    assert output.metadata["loaded"] is False


def test_brigid_capabilities():
    brigid = Brigid(checkpoint_path=Path("/nonexistent/brigid_no.pt"))
    caps = brigid.get_capabilities()
    assert caps["name"] == "Brigid"
    assert "intent_classification" in caps["capabilities"]
    # Sans checkpoint chargé : phase 1 (entre squelette 0 et trained 2).
    assert caps["phase"] == 1
    assert caps["checkpoint_loaded"] is False
