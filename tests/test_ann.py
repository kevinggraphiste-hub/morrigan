"""Tests de l'index IVF (modules/danann/ann.py) — ANN pure NumPy."""

from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from modules.danann.ann import IVFIndex
from modules.danann.quantization import exact_search, recall_at_k


def _normalized(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, d)).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


# ─── Construction ──────────────────────────────────────────────────


def test_build_default_clusters_sqrt():
    emb = _normalized(400, 64)
    idx = IVFIndex.build(emb)
    # ~sqrt(400) = 20 cellules.
    assert 10 <= idx.n_clusters <= 30
    assert len(idx) == 400


def test_lists_partition_all_vectors():
    emb = _normalized(300, 32)
    idx = IVFIndex.build(emb)
    total = sum(lst.size for lst in idx.lists)
    assert total == 300  # chaque vecteur dans exactement une cellule


def test_build_rejects_1d():
    with pytest.raises(ValueError):
        IVFIndex.build(np.zeros(64, dtype=np.float32))


# ─── Recherche ─────────────────────────────────────────────────────


def test_search_returns_k():
    emb = _normalized(500, 64, seed=1)
    idx = IVFIndex.build(emb)
    q = emb[0]
    res_idx, scores = idx.search(q, 10)
    assert len(res_idx) == 10
    assert list(scores) == sorted(scores, reverse=True)


def test_recall_high_with_enough_probes():
    """Avec assez de probes, l'IVF approche la recherche exacte."""
    emb = _normalized(600, 64, seed=2)
    idx = IVFIndex.build(emb, n_clusters=24)
    recalls = []
    rng = np.random.default_rng(3)
    for _ in range(20):
        base = emb[rng.integers(0, 600)]
        q = base + 0.03 * rng.standard_normal(64).astype(np.float32)
        a_idx, _ = idx.search(q, 10, n_probe=12)  # sonde la moitié
        e_idx, _ = exact_search(emb, q, 10)
        recalls.append(recall_at_k(a_idx, e_idx))
    assert np.mean(recalls) >= 0.8


def test_sublinearity_few_probes_scan_less():
    """Sonder peu de cellules scanne nettement moins que tout le corpus."""
    emb = _normalized(900, 64, seed=4)
    idx = IVFIndex.build(emb, n_clusters=30, n_probe=3)
    scanned = idx.candidates_scanned(n_probe=3)
    assert scanned < 900 // 2  # bien moins que la moitié du corpus


def test_more_probes_better_recall():
    emb = _normalized(500, 64, seed=5)
    idx = IVFIndex.build(emb, n_clusters=25)
    rng = np.random.default_rng(6)
    r_low, r_high = [], []
    for _ in range(15):
        q = emb[rng.integers(0, 500)] + 0.04 * rng.standard_normal(64).astype(np.float32)
        e_idx, _ = exact_search(emb, q, 10)
        r_low.append(recall_at_k(idx.search(q, 10, n_probe=2)[0], e_idx))
        r_high.append(recall_at_k(idx.search(q, 10, n_probe=15)[0], e_idx))
    assert np.mean(r_high) >= np.mean(r_low)


def test_search_self_is_top1():
    emb = _normalized(200, 64, seed=7)
    idx = IVFIndex.build(emb, n_clusters=10, n_probe=10)
    # En sondant toutes les cellules, le vecteur identique est top-1.
    res_idx, _ = idx.search(emb[42], 1, n_probe=10)
    assert res_idx[0] == 42
