"""Garde-fou mutualisation MiniLM : Danann et Brigid partagent UNE seule
instance SentenceTransformer en RAM (via core.embedder_cache), pas deux.

Levier #1 RAM sur machine modeste : sans le cache partagé, EmbeddingEngine
(Danann) et IntentEmbedder (Brigid) chargeaient chacun le même MiniLM.
"""

import pytest

pytest.importorskip("sentence_transformers")

from core.embedder_cache import _canonical, get_sentence_transformer  # noqa: E402
from modules.brigid.embedder import IntentEmbedder  # noqa: E402
from modules.danann.embeddings import EmbeddingEngine  # noqa: E402


def test_canonical_unifies_aliases():
    """`all-MiniLM-L6-v2` et `sentence-transformers/all-MiniLM-L6-v2` = 1 clé."""
    assert _canonical("all-MiniLM-L6-v2") == _canonical(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def test_cache_returns_same_instance_for_aliases():
    """Le cache renvoie le même objet pour les deux alias du modèle."""
    a = get_sentence_transformer("all-MiniLM-L6-v2")
    b = get_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
    assert a is b


def test_danann_and_brigid_share_one_model():
    """Le modèle chargé par Danann EST le même objet que celui de Brigid."""
    engine = EmbeddingEngine()  # défaut "all-MiniLM-L6-v2"
    engine.load()
    brigid = IntentEmbedder()
    brigid._ensure_loaded()
    assert engine.model is not None
    assert engine.model is brigid._model  # une seule instance MiniLM en RAM
