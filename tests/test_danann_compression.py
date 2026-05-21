"""Tests de la compression d'index Danann (Phase 4 PR 2).

Utilise le vrai embedder MiniLM (CPU) sur un petit corpus thématique.
Reranker désactivé pour isoler l'effet de la quantization.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

pytest.importorskip("sentence_transformers")
pytest.importorskip("numpy")

from modules.danann.store import Danann

# Corpus jouet : 6 chunks, 3 thèmes nets.
CORPUS = [
    "TCP est un protocole de transport fiable qui garantit la livraison des paquets.",
    "UDP est un protocole de transport rapide sans garantie de livraison.",
    "La déesse Brigid est associée à la forge, la poésie et la guérison.",
    "Le dieu Dagda possède un chaudron d'abondance dans la mythologie celtique.",
    "Un réseau neuronal liquide utilise des dynamiques à temps continu.",
    "Le transformer repose sur le mécanisme d'attention multi-têtes.",
]
META = [{"domain": d} for d in ["reseau", "reseau", "myth", "myth", "ia", "ia"]]


def _danann(compression: str) -> Danann:
    d = Danann(compression=compression, use_reranker=False)
    d.index(CORPUS, META)
    return d


# ─── Construction / config ─────────────────────────────────────────


def test_invalid_compression_rejected():
    with pytest.raises(ValueError, match="compression"):
        Danann(compression="bogus")


def test_compressed_does_not_keep_float32():
    d = _danann("int8")
    assert d.embeddings is None       # float32 jeté
    assert d._int8 is not None
    assert len(d.chunks) == len(CORPUS)


def test_binary_keeps_both_indexes():
    d = _danann("binary")
    assert d._binary is not None
    assert d._int8 is not None        # int8 = étage fin de rerank
    assert d.embeddings is None


# ─── Mémoire ───────────────────────────────────────────────────────


def test_int8_uses_less_memory_than_none():
    none = _danann("none")
    int8 = _danann("int8")
    assert 0 < int8.memory_bytes() < none.memory_bytes()
    # ~4× plus compact (codes int8 + petits scales par-vecteur).
    assert int8.memory_bytes() <= none.memory_bytes() // 3


def test_binary_uses_less_memory_than_none():
    none = _danann("none")
    binary = _danann("binary")
    assert 0 < binary.memory_bytes() < none.memory_bytes()


# ─── Qualité de recherche ──────────────────────────────────────────


@pytest.mark.parametrize("compression", ["none", "int8", "binary"])
def test_search_returns_relevant_chunk(compression):
    d = _danann(compression)
    results = d.search("Qu'est-ce que le protocole TCP ?", top_k=1)
    assert results
    top_text = results[0][0]
    # Le meilleur résultat doit parler de TCP, quelle que soit la compression.
    assert "TCP" in top_text


@pytest.mark.parametrize("compression", ["int8", "binary"])
def test_compressed_top1_matches_exact(compression):
    """Le top-1 compressé doit matcher le top-1 exact sur des requêtes nettes."""
    exact = _danann("none")
    comp = _danann(compression)
    for q in [
        "protocole réseau fiable",
        "déesse celtique de la forge",
        "réseau neuronal à temps continu",
    ]:
        e_top = exact.search(q, top_k=1)[0][0]
        c_top = comp.search(q, top_k=1)[0][0]
        assert c_top == e_top, f"[{compression}] '{q}' : {c_top!r} != {e_top!r}"


def test_domain_filter_works_compressed():
    d = _danann("int8")
    results = d.search("protocole", top_k=5, domain="reseau")
    assert results
    assert all(meta.get("domain") == "reseau" for _, _, meta in results)


def test_capabilities_report_compression():
    d = _danann("int8")
    caps = d.get_capabilities()
    assert caps["compression"] == "int8"
    assert "vector_quantization" in caps["capabilities"]
    assert caps["index_memory_bytes"] > 0


def test_empty_compressed_search_returns_empty():
    d = Danann(compression="int8", use_reranker=False)
    assert d.search("rien") == []
