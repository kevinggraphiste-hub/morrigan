"""Tests pour Danann — la mémoire vectorielle."""

import asyncio
import math
import sys

sys.path.insert(0, ".")

from core.types import ModuleInput
from modules.danann.embeddings import EmbeddingEngine
from modules.danann.store import Danann


def test_danann_health():
    danann = Danann()
    assert asyncio.run(danann.health_check()) is True


def test_danann_process():
    danann = Danann()
    input_data = ModuleInput(query="Qu'est-ce que TCP ?")
    output = asyncio.run(danann.process(input_data))
    assert output.success


def test_embeddings_are_l2_normalized():
    """Garde-fou : Danann doit produire des vecteurs L2-normalisés.

    Tout le module (quantization int8/binary, ann IVF, store mode `none`)
    suppose norme = 1 pour assimiler produit scalaire et cosinus. Une
    régression ici (oubli de normalize_embeddings) fausse silencieusement
    le ranking RAG. Voir aussi modules/brigid/embedder.py (même contrat).
    """
    engine = EmbeddingEngine()
    vectors = engine.encode(["Qu'est-ce que TCP ?", "Le réseau IP", "Python"])
    assert len(vectors) == 3
    for vec in vectors:
        norm = math.sqrt(sum(x * x for x in vec))
        assert math.isclose(norm, 1.0, abs_tol=1e-5), f"norme L2 = {norm}, attendu 1.0"
