"""Tests de l'intégration IVF dans Danann (Phase 4 PR 4)."""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

pytest.importorskip("sentence_transformers")

from modules.danann.store import Danann

CORPUS = [
    "TCP est un protocole de transport fiable qui garantit la livraison.",
    "UDP est un protocole de transport rapide sans garantie.",
    "La déesse Brigid est associée à la forge et à la poésie.",
    "Le dieu Dagda possède un chaudron d'abondance.",
    "Un réseau neuronal liquide utilise des dynamiques à temps continu.",
    "Le transformer repose sur l'attention multi-têtes.",
    "Le protocole HTTP transporte les pages web sur TCP.",
    "Cúchulainn est un héros guerrier de la mythologie celtique.",
]


def test_ivf_combines_with_int8():
    """IVF + int8 : construit sans erreur et sert des résultats pertinents."""
    d = Danann(ann="ivf", compression="int8", use_reranker=False)
    d.index(CORPUS)
    res = d.search("Qu'est-ce que le protocole TCP ?", top_k=1)
    assert res
    assert "TCP" in res[0][0]


def test_ivf_int8_stores_no_float32():
    """L'IVF compressé ne matérialise pas de float32 (re-score via int8)."""
    d = Danann(ann="ivf", compression="int8", use_reranker=False)
    d.index(CORPUS)
    d.search("réseau", top_k=1)
    assert d._ivf is not None
    assert d._ivf.vectors is None       # aucun float32 conservé
    assert d._ivf.int8 is not None      # re-score via les codes int8
    assert d.embeddings is None         # mode compressé : pas d'embeddings float


def test_ivf_combines_with_binary():
    """IVF + binary : l'IVF se bâtit sur l'int8 sous-jacent (rerank)."""
    d = Danann(ann="ivf", compression="binary", use_reranker=False)
    d.index(CORPUS)
    res = d.search("héros guerrier celtique", top_k=1)
    assert res
    assert "Cúchulainn" in res[0][0]


def test_ivf_int8_top1_matches_flat_float():
    """Sur requêtes nettes, IVF+int8 ≈ flat float (quantization quasi sans perte)."""
    flat = Danann(ann="flat", use_reranker=False)
    flat.index(CORPUS)
    ivf8 = Danann(ann="ivf", compression="int8", use_reranker=False)
    ivf8.index(CORPUS)
    for q in ["protocole réseau fiable", "héros celtique", "attention multi-têtes"]:
        assert flat.search(q, top_k=1)[0][0] == ivf8.search(q, top_k=1)[0][0], q


def test_invalid_ann_rejected():
    with pytest.raises(ValueError, match="ann"):
        Danann(ann="bogus")


def test_ivf_search_returns_relevant():
    d = Danann(ann="ivf", use_reranker=False)
    d.index(CORPUS)
    results = d.search("Qu'est-ce que le protocole TCP ?", top_k=1)
    assert results
    assert "TCP" in results[0][0]


def test_ivf_built_lazily_and_reported():
    d = Danann(ann="ivf", use_reranker=False)
    d.index(CORPUS)
    assert d._ivf is None  # pas encore construit
    d.search("réseau neuronal", top_k=1)
    assert d._ivf is not None  # construit au 1er search
    caps = d.get_capabilities()
    assert caps["ann"] == "ivf"


def test_ivf_invalidated_on_new_index():
    d = Danann(ann="ivf", use_reranker=False)
    d.index(CORPUS[:4])
    d.search("Brigid", top_k=1)
    assert d._ivf is not None
    d.index(CORPUS[4:])         # nouvel index → IVF invalidé
    assert d._ivf is None
    # Re-construit et trouve le nouveau contenu.
    res = d.search("transformer attention", top_k=1)
    assert d._ivf is not None
    assert "transformer" in res[0][0].lower()


def test_ivf_top1_matches_flat_on_clear_queries():
    flat = Danann(ann="flat", use_reranker=False)
    flat.index(CORPUS)
    ivf = Danann(ann="ivf", use_reranker=False)
    ivf.index(CORPUS)
    for q in ["protocole réseau fiable", "héros celtique", "attention multi-têtes"]:
        # n_probe par défaut peut rater ; on sonde large via top_k élevé.
        assert flat.search(q, top_k=1)[0][0] == ivf.search(q, top_k=1)[0][0], q
