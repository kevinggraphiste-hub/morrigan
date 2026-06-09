"""Garde-fou mutualisation de l'embedder : Danann et Brigid partagent UNE seule
instance SentenceTransformer en RAM (via core.embedder_cache), pas deux.

Levier #1 RAM sur machine modeste : sans le cache partagé, EmbeddingEngine
(Danann) et IntentEmbedder (Brigid) chargeaient chacun le même modèle.
"""

import pytest

pytest.importorskip("sentence_transformers")

from core.embedder_cache import _canonical, get_sentence_transformer  # noqa: E402
from modules.brigid.embedder import EMBED_MODEL_NAME, IntentEmbedder  # noqa: E402
from modules.danann.embeddings import EmbeddingEngine  # noqa: E402


def test_canonical_unifies_aliases():
    """Un nom nu et son équivalent org-préfixé désignent la même clé de cache."""
    assert _canonical("all-MiniLM-L6-v2") == _canonical(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def test_cache_returns_same_instance():
    """Deux appels au cache pour le même modèle renvoient le même objet."""
    a = get_sentence_transformer(EMBED_MODEL_NAME)
    b = get_sentence_transformer(EMBED_MODEL_NAME)
    assert a is b


def test_danann_and_brigid_share_one_model():
    """Le modèle chargé par Danann EST le même objet que celui de Brigid
    (Danann et Brigid utilisent par défaut le même nom canonique e5)."""
    engine = EmbeddingEngine()  # défaut e5-small, même nom que Brigid
    engine.load()
    brigid = IntentEmbedder()
    brigid._ensure_loaded()
    assert engine.model is not None
    assert engine.model is brigid._model  # une seule instance en RAM
