"""Tests pour An Dagda — l'orchestrateur."""

import asyncio
import pytest
import sys

sys.path.insert(0, ".")

from core.dagda import AnDagda
from core.types import QueryType


def test_classify_factual():
    """An Dagda doit classifier les questions factuelles."""
    dagda = AnDagda()
    result = dagda.classify_query("Qui est le président de la France ?")
    assert result.query_type == QueryType.FACTUAL


def test_classify_creative():
    """An Dagda doit classifier les tâches créatives."""
    dagda = AnDagda()
    result = dagda.classify_query("Écris-moi un poème sur la mer")
    assert result.query_type == QueryType.CREATIVE


def test_classify_reasoning():
    """An Dagda doit classifier les demandes de raisonnement."""
    dagda = AnDagda()
    result = dagda.classify_query("Explique la différence entre TCP et UDP")
    assert result.query_type == QueryType.REASONING


def test_classify_conversation():
    """Les inputs non classifiés → conversation."""
    dagda = AnDagda()
    result = dagda.classify_query("Salut, comment ça va ?")
    assert result.query_type == QueryType.CONVERSATION


def test_register_module():
    """An Dagda doit pouvoir enregistrer des modules."""
    from modules.scathach.generator import Scathach

    dagda = AnDagda()
    dagda.register_module("scathach", Scathach())
    assert "scathach" in dagda.modules
