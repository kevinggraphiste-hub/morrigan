"""Tests de la quantization vectorielle Danann (Phase 4 PR 1)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from modules.danann.quantization import (
    BinaryIndex,
    Int8Index,
    exact_search,
    recall_at_k,
    two_stage_search,
)


def _normalized(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Matrice (n, d) de vecteurs L2-normalisés (comme MiniLM)."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, d)).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


# ─── Int8 ──────────────────────────────────────────────────────────


def test_int8_build_shapes_and_dtype():
    emb = _normalized(50, 384)
    idx = Int8Index.build(emb)
    assert idx.codes.shape == (50, 384)
    assert idx.codes.dtype == np.int8
    assert len(idx) == 50


def test_int8_is_4x_smaller_than_float32():
    emb = _normalized(100, 384)
    idx = Int8Index.build(emb)
    # Les codes int8 = exactement 4× plus petits ; memory_bytes inclut
    # en plus le(s) scale(s) (8 octets pour le scalaire global).
    assert idx.codes.nbytes == emb.nbytes // 4
    assert idx.memory_bytes() <= emb.nbytes // 4 + 16


def test_int8_dequantize_is_close():
    emb = _normalized(20, 384)
    idx = Int8Index.build(emb)
    deq = idx.dequantize()
    # Erreur de quantization bornée par le pas (scale).
    assert np.max(np.abs(deq - emb)) <= idx.scale + 1e-6


def test_int8_recall_is_high():
    """La recherche int8 retrouve l'essentiel du top-k exact."""
    emb = _normalized(500, 384, seed=1)
    idx = Int8Index.build(emb)
    recalls = []
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = emb[rng.integers(0, 500)] + 0.01 * rng.standard_normal(384).astype(np.float32)
        a_idx, _ = idx.search(q, 10)
        e_idx, _ = exact_search(emb, q, 10)
        recalls.append(recall_at_k(a_idx, e_idx))
    # int8 = quasi sans perte → recall moyen très élevé.
    assert np.mean(recalls) >= 0.9


# ─── Binary ────────────────────────────────────────────────────────


def test_binary_is_much_smaller():
    emb = _normalized(100, 384)
    idx = BinaryIndex.build(emb)
    # 384 bits = 48 octets/vecteur vs 1536 octets en float32 → 32×.
    assert idx.memory_bytes() == 100 * (384 // 8)
    assert idx.memory_bytes() < emb.nbytes // 30


def test_binary_search_returns_k():
    emb = _normalized(200, 384)
    idx = BinaryIndex.build(emb)
    q = emb[0]
    a_idx, scores = idx.search(q, 5)
    assert len(a_idx) == 5
    # Le vecteur identique à lui-même doit être bien classé (score haut).
    assert 0 in a_idx.tolist()


def test_binary_recall_lower_but_reasonable():
    """Binary seul : recall plus faible (filtre grossier), mais > hasard."""
    emb = _normalized(500, 384, seed=3)
    idx = BinaryIndex.build(emb)
    recalls = []
    rng = np.random.default_rng(4)
    for _ in range(20):
        q = emb[rng.integers(0, 500)] + 0.05 * rng.standard_normal(384).astype(np.float32)
        a_idx, _ = idx.search(q, 10)
        e_idx, _ = exact_search(emb, q, 10)
        recalls.append(recall_at_k(a_idx, e_idx))
    # Bien au-dessus du hasard (10/500 = 2%), mais imparfait.
    assert np.mean(recalls) >= 0.3


# ─── Recherche 2 étages ────────────────────────────────────────────


def test_two_stage_recovers_recall():
    """Le re-score float après filtre binaire récupère un recall élevé."""
    emb = _normalized(500, 384, seed=5)
    coarse = BinaryIndex.build(emb)
    recalls_coarse, recalls_2stage = [], []
    rng = np.random.default_rng(6)
    for _ in range(20):
        q = emb[rng.integers(0, 500)] + 0.05 * rng.standard_normal(384).astype(np.float32)
        e_idx, _ = exact_search(emb, q, 10)
        c_idx, _ = coarse.search(q, 10)
        t_idx, _ = two_stage_search(coarse, emb, q, 10)
        recalls_coarse.append(recall_at_k(c_idx, e_idx))
        recalls_2stage.append(recall_at_k(t_idx, e_idx))
    # Le 2 étages doit faire nettement mieux que le binaire seul
    # (c'est l'invariant clé) et atteindre un recall élevé.
    assert np.mean(recalls_2stage) > np.mean(recalls_coarse)
    assert np.mean(recalls_2stage) >= 0.8


def test_two_stage_empty_index():
    emb = _normalized(1, 384)
    coarse = BinaryIndex.build(emb)
    idx, scores = two_stage_search(coarse, emb, emb[0], 10)
    assert len(idx) == 1  # un seul vecteur disponible


# ─── Helpers / garde-fous ──────────────────────────────────────────


def test_exact_search_topk_ordering():
    emb = _normalized(30, 16, seed=7)
    q = emb[5]
    idx, scores = exact_search(emb, q, 5)
    assert idx[0] == 5  # le vecteur identique est le 1er
    assert list(scores) == sorted(scores, reverse=True)


def test_build_rejects_1d():
    with pytest.raises(ValueError):
        Int8Index.build(np.zeros(384, dtype=np.float32))


def test_recall_at_k_edge_cases():
    assert recall_at_k(np.array([1, 2, 3]), np.array([])) == 1.0
    assert recall_at_k(np.array([1, 2]), np.array([1, 2])) == 1.0
    assert recall_at_k(np.array([9, 8]), np.array([1, 2])) == 0.0
