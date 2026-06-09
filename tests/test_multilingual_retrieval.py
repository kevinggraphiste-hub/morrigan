"""Garde-fou Phase 2A : retrieval cross-lingue (requête FR → passage EN).

C'est la raison d'être de la bascule vers `intfloat/multilingual-e5-small` :
le corpus de docs code est majoritairement anglophone, les requêtes utilisateur
sont en français. On vérifie qu'une question FR retrouve bien le passage EN
pertinent (et pas un distracteur), préfixes e5 `query:`/`passage:` appliqués.

Utilise le vrai embedder (CPU) — skip si sentence-transformers absent.
"""

import sys

import pytest

sys.path.insert(0, ".")

from core.embedder_cache import text_prompt_prefix


# ─── Unitaire : préfixes e5 (pas de modèle) ───────────────────────────


def test_prefix_e5_asymmetric():
    m = "intfloat/multilingual-e5-small"
    assert text_prompt_prefix(m, "query") == "query: "
    assert text_prompt_prefix(m, "passage") == "passage: "


def test_prefix_empty_for_non_e5():
    # MiniLM (ou tout modèle hors famille e5) → aucun préfixe.
    assert text_prompt_prefix("all-MiniLM-L6-v2", "query") == ""
    assert text_prompt_prefix("sentence-transformers/all-MiniLM-L6-v2", "passage") == ""


# ─── Intégration : FR → EN (vrai modèle e5) ───────────────────────────


def test_french_query_retrieves_english_passage():
    pytest.importorskip("sentence_transformers")
    from modules.danann.store import Danann

    danann = Danann(use_reranker=False)
    danann.index(
        [
            "TCP is a reliable, connection-oriented network protocol.",
            "Photosynthesis converts sunlight into chemical energy in plants.",
        ],
        [{"domain": "reseau"}, {"domain": "bio"}],
    )

    # Requête en français → doit ramener le passage réseau (EN) en tête.
    results = danann.search("Qu'est-ce que le protocole réseau TCP ?", top_k=2)
    assert results, "le retrieval cross-lingue ne doit rien renvoyer de vide"
    top_text = results[0][0]
    assert "TCP" in top_text, f"top chunk inattendu (cross-lingue KO) : {top_text!r}"


def test_french_query_retrieves_english_code_passage():
    pytest.importorskip("sentence_transformers")
    from modules.danann.store import Danann

    danann = Danann(use_reranker=False)
    danann.index(
        [
            "In Python, a list comprehension builds a list from an iterable.",
            "Le Dagda est une divinité de la mythologie celtique irlandaise.",
        ],
        [{"domain": "code"}, {"domain": "mythologie"}],
    )

    results = danann.search("comment créer une liste en Python ?", top_k=2)
    assert results
    assert "Python" in results[0][0]
