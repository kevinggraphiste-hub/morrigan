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


def test_ivf_requires_no_compression():
    with pytest.raises(ValueError, match="ivf"):
        Danann(ann="ivf", compression="int8")


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
